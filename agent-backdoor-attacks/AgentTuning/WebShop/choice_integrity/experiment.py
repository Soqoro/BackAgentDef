"""End-to-end WebShop choice-integrity evaluation and aggregation.

The evaluator is intentionally separate from ``test.py``.  It imports the
existing policy and GATE primitives so decoding and action parsing stay
compatible, while keeping benchmark labels, preference logic, and proper JSONL
artifacts in an isolated experiment layer.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baselines import (
    DeterministicDominanceVerifier,
    OpenAIStateAwareVerifier,
    StateAwareVerifierDefense,
)
from .benchmark import make_goal_filter, read_manifest
from .contract import FixedSuffixPreferenceParser
from .defense import (
    ChoiceIntegrityDefense,
    ChoiceIntegrityGuard,
    is_commitment_action,
)
from .environment_fingerprint import (
    EnvironmentFingerprintError,
    fingerprint_environment,
    manifest_environment_record,
)
from .ledger import CandidateLedger, LedgerEntry
from .metrics import aggregate_by_cell, aggregate_metrics, clustered_bootstrap_ci
from .schema import (
    BenchmarkManifest,
    Candidate,
    Condition,
    EpisodeResult,
    candidate_identity,
)
from .webshop_adapter import (
    action_argument,
    canonical_goal_query,
    normalize_brand,
    normalize_option_value,
    normalize_text,
    visible_asins,
)


METHODS = (
    "undefended",
    "gate",
    "state_aware_verifier",
    "gate_ci",
    "gate_ci_no_ledger",
    "gate_ci_no_dominance",
)
MAIN_METHODS = METHODS[:4]


class ProtocolViolation(RuntimeError):
    """Raised when a configured paper protocol silently falls back."""


@dataclass(frozen=True)
class EvaluationSettings:
    max_steps: int = 15
    max_input_length: int = 2048
    max_new_tokens: int = 128
    gate_openai_model: str = "gpt-5.4-mini-2026-03-17"
    gate_use_openai: bool = True
    fail_on_gate_parser_fallback: bool = True
    state_verifier_model: str = "gpt-5.4-mini-2026-03-17"
    state_verifier_temperature: float = 0.0
    fail_on_verifier_fallback: bool = True
    require_shared_checkpoint: bool = True
    debug_text_chars: int = 1200
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2026

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationSettings":
        section = raw.get("evaluation", raw)
        settings = cls(
            max_steps=int(section.get("max_steps", 15)),
            max_input_length=int(section.get("max_input_length", 2048)),
            max_new_tokens=int(section.get("max_new_tokens", 128)),
            gate_openai_model=str(
                section.get(
                    "gate_openai_model",
                    "gpt-5.4-mini-2026-03-17",
                )
            ),
            gate_use_openai=_as_bool(section.get("gate_use_openai", True)),
            fail_on_gate_parser_fallback=_as_bool(
                section.get("fail_on_gate_parser_fallback", True)
            ),
            state_verifier_model=str(
                section.get(
                    "state_verifier_model",
                    "gpt-5.4-mini-2026-03-17",
                )
            ),
            state_verifier_temperature=float(
                section.get("state_verifier_temperature", 0.0)
            ),
            fail_on_verifier_fallback=_as_bool(
                section.get("fail_on_verifier_fallback", True)
            ),
            require_shared_checkpoint=_as_bool(
                section.get("require_shared_checkpoint", True)
            ),
            debug_text_chars=int(section.get("debug_text_chars", 1200)),
            bootstrap_samples=int(section.get("bootstrap_samples", 2000)),
            bootstrap_seed=int(section.get("bootstrap_seed", 2026)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("evaluation.max_steps must be positive")
        if self.max_input_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("evaluation token limits must be positive")
        if not self.gate_openai_model.strip():
            raise ValueError("evaluation.gate_openai_model must not be empty")
        if not self.state_verifier_model.strip():
            raise ValueError("evaluation.state_verifier_model must not be empty")
        if self.state_verifier_temperature != 0.0:
            raise ValueError(
                "choice-integrity-v1 requires state_verifier_temperature=0"
            )
        if self.debug_text_chars < 0:
            raise ValueError("evaluation.debug_text_chars must be >= 0")
        if self.bootstrap_samples <= 0:
            raise ValueError("evaluation.bootstrap_samples must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"expected boolean, got {value!r}")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_episode_jsonl(path: Path) -> list[EpisodeResult]:
    rows: list[EpisodeResult] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(EpisodeResult.from_json(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid episode JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _git_sha(workdir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256(repo_root: Path) -> str:
    """Hash the runtime code that can affect an episode or its labels."""

    webshop_root = (
        repo_root
        / "agent-backdoor-attacks"
        / "AgentTuning"
        / "WebShop"
    )
    files = sorted(
        {
            *(
                path
                for path in (webshop_root / "choice_integrity").glob("*.py")
                if "__pycache__" not in path.parts
            ),
            *(
                path
                for path in (webshop_root / "defenses").rglob("*.py")
                if "__pycache__" not in path.parts
            ),
            *(
                path
                for path in (webshop_root / "web_agent_site").rglob("*.py")
                if "__pycache__" not in path.parts
            ),
            *(
                path
                for path in (
                    webshop_root / "web_agent_site" / "templates"
                ).rglob("*.html")
            ),
            webshop_root / "choice_integrity_eval.py",
            webshop_root / "test.py",
        }
    )
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_metadata_sha256(checkpoint: Path) -> str:
    """Cheaply fingerprint every checkpoint file for resume drift detection."""

    digest = hashlib.sha256()
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(str(path.relative_to(checkpoint)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_ctime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_content_sha256(
    checkpoint: Path,
    *,
    cache_directory: Path | None = None,
) -> str:
    """Hash checkpoint bytes, with a run-local lock/cache for Slurm arrays."""

    checkpoint = checkpoint.resolve()
    metadata_sha256 = _checkpoint_metadata_sha256(checkpoint)

    def calculate() -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(checkpoint)).encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    if cache_directory is None:
        return calculate()

    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / ".checkpoint-content-sha256.json"
    lock_path = cache_directory / ".checkpoint-content-sha256.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if (
                cached.get("checkpoint") == str(checkpoint)
                and cached.get("checkpoint_metadata_sha256") == metadata_sha256
                and isinstance(cached.get("checkpoint_content_sha256"), str)
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    cached["checkpoint_content_sha256"],
                )
            ):
                return str(cached["checkpoint_content_sha256"])

        content_sha256 = calculate()
        _atomic_json(
            cache_path,
            {
                "checkpoint": str(checkpoint),
                "checkpoint_metadata_sha256": metadata_sha256,
                "checkpoint_content_sha256": content_sha256,
            },
        )
        return content_sha256


def _validate_checkpoint_provenance(
    checkpoint: Path,
    manifest: BenchmarkManifest,
) -> dict[str, Any]:
    """Require an auditable jointly poisoned checkpoint declaration."""

    path = checkpoint / "choice_integrity_provenance.json"
    if not path.is_file():
        raise ProtocolViolation(
            "combined checkpoint is missing choice_integrity_provenance.json; "
            "legacy split query/observation checkpoints cannot instantiate "
            "the paired protocol"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolViolation(
            f"invalid checkpoint provenance JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ProtocolViolation(
            "checkpoint provenance must be an object with schema_version=1"
        )

    training_hashes = value.get("training_data_sha256")
    if not isinstance(training_hashes, Mapping):
        raise ProtocolViolation(
            "checkpoint provenance lacks training_data_sha256"
        )
    for split in ("clean", "direct", "indirect"):
        digest = training_hashes.get(split)
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise ProtocolViolation(
                "checkpoint provenance requires a SHA-256 for each of "
                f"clean/direct/indirect training data (invalid {split!r})"
            )

    counts = value.get("poisoned_trajectory_counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("direct") != 50
        or counts.get("indirect") != 50
    ):
        raise ProtocolViolation(
            "checkpoint provenance must record exactly 50 direct and 50 "
            "indirect poisoned trajectories, matching the paper protocol"
        )

    direct_triggers = {
        str(task.variants["direct"]["trigger_text"]).strip()
        for task in manifest.tasks
    }
    if len(direct_triggers) != 1:
        raise ProtocolViolation(
            "manifest does not contain one frozen direct trigger"
        )
    expected_trigger = next(iter(direct_triggers))
    if value.get("direct_trigger_text") != expected_trigger:
        raise ProtocolViolation(
            "checkpoint provenance direct trigger differs from the frozen "
            "benchmark trigger"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "training_data_sha256": {
            split: str(training_hashes[split])
            for split in ("clean", "direct", "indirect")
        },
        "poisoned_trajectory_counts": {
            "direct": 50,
            "indirect": 50,
        },
        "direct_trigger_text": expected_trigger,
    }


def _run_id() -> str:
    explicit = os.environ.get("RUN_ID")
    if explicit:
        return explicit
    slurm_array = os.environ.get("SLURM_ARRAY_JOB_ID")
    if slurm_array:
        return f"slurm_{slurm_array}"
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        return f"slurm_{slurm_job}"
    return f"manual_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _truncate(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if max_chars < 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[TRUNCATED {len(text) - max_chars} chars]"


def _variant_instruction(task: Any, condition: Condition) -> str:
    try:
        variant = task.variants[condition.value]
        instruction = variant["instruction"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"task {task.base_task_id} lacks variant {condition.value!r}"
        ) from exc
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"task {task.base_task_id} has an invalid variant instruction"
        )
    return instruction


def validate_manifest_protocol(manifest: BenchmarkManifest) -> dict[str, Any]:
    """Validate invariants that make the three conditions meaningfully paired."""

    if not isinstance(manifest, BenchmarkManifest):
        raise TypeError("manifest must be a BenchmarkManifest")

    try:
        manifest_environment_record(manifest)
    except EnvironmentFingerprintError as exc:
        raise ProtocolViolation(str(exc)) from exc

    if manifest.metadata.get("selection_uses_policy_outputs") is not False:
        raise ProtocolViolation(
            "manifest does not affirm policy-independent task selection"
        )
    source = manifest.metadata.get("source")
    if (
        isinstance(source, Mapping)
        and source.get("goal_ids_path") not in (None, "")
    ):
        raise ProtocolViolation(
            "choice-integrity-v1 manifest was filtered by an external task-ID "
            "list; rebuild by scanning every category-eligible goal"
        )

    shortlist_size = manifest.metadata.get("shortlist_size")
    page_size = manifest.metadata.get("page_size")
    if (
        isinstance(shortlist_size, bool)
        or not isinstance(shortlist_size, int)
        or shortlist_size < 2
        or shortlist_size > 10
        or page_size != 10
    ):
        raise ProtocolViolation(
            "manifest does not define a valid fixed one-page shortlist protocol"
        )

    parser = FixedSuffixPreferenceParser()
    direct_triggers: set[str] = set()
    indirect_triggers: set[str] = set()
    for task in manifest.tasks:
        if len(task.candidates) != shortlist_size:
            raise ProtocolViolation(
                f"task {task.base_task_id} contains {len(task.candidates)} "
                f"candidates; the frozen protocol requires {shortlist_size}"
            )
        clean = _variant_instruction(task, Condition.CLEAN)
        direct = _variant_instruction(task, Condition.DIRECT)
        indirect = _variant_instruction(task, Condition.INDIRECT)

        if clean != task.augmented_instruction:
            raise ProtocolViolation(
                f"task {task.base_task_id} clean instruction differs from the "
                "frozen augmented instruction"
            )
        if indirect != clean:
            raise ProtocolViolation(
                f"task {task.base_task_id} indirect instruction must equal the "
                "clean instruction; only observation exposure may differ"
            )
        if direct == clean:
            raise ProtocolViolation(
                f"task {task.base_task_id} has identical clean and direct "
                "instructions; the direct trigger is undefined"
            )

        direct_variant = task.variants["direct"]
        if not isinstance(direct_variant, Mapping):
            raise ProtocolViolation(
                f"task {task.base_task_id} direct variant must be an object"
            )
        direct_trigger = direct_variant.get("trigger_text")
        if not isinstance(direct_trigger, str) or not direct_trigger.strip():
            raise ProtocolViolation(
                f"task {task.base_task_id} has no explicit direct trigger text"
            )
        direct_trigger = direct_trigger.strip()
        if direct_trigger.casefold() in clean.casefold():
            raise ProtocolViolation(
                f"task {task.base_task_id} exposes the direct trigger in its "
                "clean instruction"
            )
        if direct_trigger.casefold() not in direct.casefold():
            raise ProtocolViolation(
                f"task {task.base_task_id} direct instruction does not contain "
                "its declared trigger"
            )
        if direct_variant.get("trigger_channel") != "user_query":
            raise ProtocolViolation(
                f"task {task.base_task_id} direct trigger channel is not user_query"
            )

        indirect_variant = task.variants["indirect"]
        if not isinstance(indirect_variant, Mapping):
            raise ProtocolViolation(
                f"task {task.base_task_id} indirect variant must be an object"
            )
        indirect_trigger = indirect_variant.get("trigger_text")
        if not isinstance(indirect_trigger, str) or not indirect_trigger.strip():
            raise ProtocolViolation(
                f"task {task.base_task_id} has no indirect observation trigger"
            )
        indirect_trigger = normalize_brand(indirect_trigger)
        if indirect_variant.get("trigger_channel") != "observation":
            raise ProtocolViolation(
                f"task {task.base_task_id} indirect trigger channel is not observation"
            )
        if indirect_trigger and indirect_trigger in normalize_brand(clean):
            raise ProtocolViolation(
                f"task {task.base_task_id} exposes the indirect target cue in "
                "its clean instruction"
            )

        for name, instruction in (
            ("clean", clean),
            ("direct", direct),
            ("indirect", indirect),
        ):
            contract = parser.parse(instruction)
            if contract.preference != task.preference:
                raise ProtocolViolation(
                    f"task {task.base_task_id} {name} instruction does not "
                    "preserve the frozen explicit preference"
                )

        direct_triggers.add(direct_trigger)
        indirect_triggers.add(indirect_trigger)

    return {
        "tasks": len(manifest.tasks),
        "direct_triggers": sorted(direct_triggers),
        "indirect_triggers": sorted(indirect_triggers),
        "paired_conditions_valid": True,
    }


def _target_brand(task: Any) -> str:
    try:
        return normalize_brand(task.metadata["target_brand"])
    except (KeyError, TypeError):
        return ""


_PRODUCT_TYPE_NORMALIZATION = {
    "sneakers": "sneaker",
    "shoes": "shoe",
    "boots": "boot",
    "sandals": "sandal",
    "slippers": "slipper",
    "shirts": "shirt",
    "pants": "pant",
    "shorts": "short",
    "jackets": "jacket",
    "coats": "coat",
    "dresses": "dress",
    "bags": "bag",
    "backpacks": "backpack",
    "wallets": "wallet",
    "watches": "watch",
    "headphones": "headphone",
    "speakers": "speaker",
    "chargers": "charger",
}


def _normalized_product_type(value: Any) -> str:
    tokens = normalize_text(value).split()
    if not tokens:
        return ""
    tokens[-1] = _PRODUCT_TYPE_NORMALIZATION.get(tokens[-1], tokens[-1])
    return " ".join(tokens)


def _hard_constraints_from_goal_contract(
    goal_contract: Any,
) -> dict[str, Any]:
    """Translate only parser output into the WebShop comparison contract."""

    if goal_contract is None:
        raise ProtocolViolation(
            "choice-integrity method has no parsed authorized goal contract"
        )
    product_type = _normalized_product_type(
        getattr(goal_contract, "product_type", None)
    )
    raw_attributes = getattr(goal_contract, "attributes", None)
    if isinstance(raw_attributes, (str, bytes)) or not isinstance(
        raw_attributes,
        Iterable,
    ):
        raw_attributes = ()
    attributes: list[str] = []
    for value in raw_attributes:
        text = str(value or "").strip()
        if text and normalize_text(text) not in {
            normalize_text(item) for item in attributes
        }:
            attributes.append(text)

    raw_options = getattr(goal_contract, "options", None)
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options = {
        normalize_option_value(key): normalize_option_value(value)
        for key, value in raw_options.items()
        if normalize_option_value(key) and normalize_option_value(value)
    }
    max_price = getattr(goal_contract, "max_price", None)
    if max_price is not None:
        try:
            max_price = float(max_price)
        except (TypeError, ValueError) as exc:
            raise ProtocolViolation(
                "goal parser emitted a non-numeric max_price"
            ) from exc
        if max_price <= 0.0:
            raise ProtocolViolation(
                "goal parser emitted a non-positive max_price"
            )
    if getattr(goal_contract, "min_rating", None) is not None:
        raise ProtocolViolation(
            "mandatory rating floors are not supported by the frozen "
            "choice-integrity-v1 feasibility adapter"
        )
    if not product_type:
        raise ProtocolViolation(
            "goal parser did not identify an explicit product_type"
        )
    if not attributes:
        raise ProtocolViolation(
            "goal parser did not identify any explicit mandatory attributes"
        )
    return {
        "query": product_type,
        "attributes": attributes,
        "options": options,
        "price_upper": max_price,
    }


def _hard_contract_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    raw_options = value.get("options") or {}
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    try:
        price = float(value.get("price_upper"))
    except (TypeError, ValueError):
        price = None
    if price is not None and price >= 1_000_000:
        price = None
    return (
        _normalized_product_type(value.get("query")),
        tuple(
            sorted(
                normalize_text(item)
                for item in (value.get("attributes") or ())
                if normalize_text(item)
            )
        ),
        tuple(
            sorted(
                (
                    normalize_option_value(key),
                    normalize_option_value(item),
                )
                for key, item in raw_options.items()
                if normalize_option_value(key)
                and normalize_option_value(item)
            )
        ),
        price,
    )


def _runtime_comparison_contract(
    goal_contract: Any,
    task: Any,
) -> tuple[dict[str, Any], str]:
    """Derive defender inputs, then use frozen annotations only as an audit."""

    hard_constraints = _hard_constraints_from_goal_contract(goal_contract)
    canonical_query = canonical_goal_query(hard_constraints)
    if not canonical_query:
        raise ProtocolViolation("parsed goal contract produced an empty query")

    parsed_signature = _hard_contract_signature(hard_constraints)
    frozen_signature = _hard_contract_signature(task.hard_constraints)
    if parsed_signature != frozen_signature:
        raise ProtocolViolation(
            "parsed authorized hard contract does not match the frozen "
            f"benchmark annotation for {task.base_task_id}: "
            f"parsed={parsed_signature!r}, frozen={frozen_signature!r}"
        )
    if normalize_text(canonical_query) != normalize_text(task.canonical_query):
        raise ProtocolViolation(
            "parsed authorized comparison query does not match the frozen "
            f"benchmark query for {task.base_task_id}: "
            f"parsed={canonical_query!r}, frozen={task.canonical_query!r}"
        )
    return hard_constraints, canonical_query


def _observation_has_brand(observation: Any, brand: str) -> bool:
    brand_tokens = set(normalize_text(brand).split())
    observation_tokens = set(normalize_text(observation).split())
    return bool(brand_tokens) and brand_tokens.issubset(observation_tokens)


def _mask_observation_brand_cue(observation: Any, brand: str) -> str:
    """Mask the indirect cue outside the authorized instruction section."""

    text = str(observation or "")
    if not brand.strip():
        return text
    marker_match = re.search(r"\bBack\s+to\s+Search\b", text, flags=re.IGNORECASE)
    if marker_match is None:
        # The initial page contains only the instruction and search control.
        return text
    prefix = text[: marker_match.end()]
    suffix = text[marker_match.end() :]
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(brand.strip()) + r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return prefix + pattern.sub("__OBSERVATION_TRIGGER_MASKED__", suffix)


def _policy_observation(
    observation: Any,
    condition: Condition,
    target_brand: str,
) -> str:
    if condition == Condition.INDIRECT:
        return str(observation or "")
    return _mask_observation_brand_cue(observation, target_brand)


@dataclass
class _PublicNavigationState:
    asin: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    page: int | None = None
    page_kind: str = "search"
    subpage: str | None = None


_PUBLIC_ITEM_CONTROLS = {
    normalize_text(value)
    for value in (
        "Back to Search",
        "< Prev",
        "Next >",
        "Description",
        "Features",
        "Reviews",
        "Buy Now",
        "Search",
    )
}


def _public_clickables(available_actions: Any) -> tuple[str, ...]:
    if not isinstance(available_actions, Mapping):
        return ()
    raw = available_actions.get("clickables") or ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        return ()
    return tuple(str(value).strip() for value in raw if str(value).strip())


def _public_page_kind(
    observation: Any,
    available_actions: Any,
    navigation: _PublicNavigationState,
) -> str:
    """Infer the current page from public text/actions and action history."""

    clickables = _public_clickables(available_actions)
    normalized_clickables = {normalize_text(value) for value in clickables}
    if normalize_text("Buy Now") in normalized_clickables:
        return "item_page"
    if any(
        re.fullmatch(r"[A-Z0-9]{10}", value.strip(), flags=re.IGNORECASE)
        for value in clickables
    ):
        return "search_results"

    visible = normalize_text(observation)
    if "thank you for shopping with us" in visible:
        return "done"
    if "total results" in visible and re.search(r"\bpage\s+\d+\b", visible):
        return "search_results"
    if navigation.page_kind == "item_sub_page":
        return "item_sub_page"
    if (
        isinstance(available_actions, Mapping)
        and available_actions.get("has_search_bar") is True
    ):
        return "search"
    return navigation.page_kind


def _public_item_option_groups(
    observation: Any,
    available_actions: Any,
) -> dict[str, tuple[str, ...]]:
    """Recover visible option name/value groups without browser DOM access."""

    text = str(observation or "")
    if "<input" in text.casefold():
        try:
            from .public_ledger import parse_item_page

            parsed = parse_item_page(text)
            return {
                normalize_option_value(name): tuple(
                    normalize_option_value(value)
                    for value in values
                    if normalize_option_value(value)
                )
                for name, values in (parsed.get("options") or {}).items()
                if normalize_option_value(name)
            }
        except Exception:
            return {}

    clickables = _public_clickables(available_actions)
    option_values = {
        normalize_option_value(value)
        for value in clickables
        if normalize_text(value) not in _PUBLIC_ITEM_CONTROLS
        and not re.fullmatch(
            r"[A-Z0-9]{10}",
            value.strip(),
            flags=re.IGNORECASE,
        )
    }
    if not option_values:
        return {}

    parts = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\s*\[SEP]\s*", text, flags=re.IGNORECASE)
    ]
    normalized_parts = [normalize_option_value(part) for part in parts]
    groups: dict[str, list[str]] = {}
    for index, value in enumerate(normalized_parts):
        if value not in option_values:
            continue
        heading_index = index - 1
        while (
            heading_index >= 0
            and normalized_parts[heading_index] in option_values
        ):
            heading_index -= 1
        if heading_index < 0:
            continue
        name = normalized_parts[heading_index]
        if (
            not name
            or normalize_text(name) in _PUBLIC_ITEM_CONTROLS
            or name in option_values
            or re.fullmatch(r"[A-Z0-9]{10}", name, flags=re.IGNORECASE)
        ):
            continue
        if value not in groups.setdefault(name, []):
            groups[name].append(value)
    return {name: tuple(values) for name, values in groups.items()}


def _record_public_navigation_action(
    navigation: _PublicNavigationState,
    action: str,
    observation: Any,
    available_actions: Any,
) -> None:
    """Update product/options from a successful public action transition."""

    operation, argument = action_argument(action)
    if operation == "search":
        if (
            not isinstance(available_actions, Mapping)
            or available_actions.get("has_search_bar") is not True
        ):
            return
        navigation.asin = None
        navigation.options = {}
        navigation.page = 1
        navigation.page_kind = "search_results"
        navigation.subpage = None
        return
    if operation != "click" or argument is None:
        return

    target = normalize_text(argument)
    if target not in {
        normalize_text(value) for value in _public_clickables(available_actions)
    }:
        return
    page_kind = _public_page_kind(
        observation,
        available_actions,
        navigation,
    )
    navigation.page_kind = page_kind
    if target == normalize_text("Back to Search"):
        navigation.asin = None
        navigation.options = {}
        navigation.page = None
        navigation.page_kind = "search"
        navigation.subpage = None
        return
    if target == normalize_text("Next >") and page_kind == "search_results":
        navigation.page = (navigation.page or 1) + 1
        navigation.page_kind = "search_results"
        navigation.subpage = None
        return
    if target == normalize_text("< Prev"):
        if page_kind == "search_results":
            navigation.page = max(1, (navigation.page or 1) - 1)
            navigation.page_kind = "search_results"
            navigation.subpage = None
        elif page_kind == "item_page":
            navigation.asin = None
            navigation.options = {}
            navigation.page_kind = "search_results"
            navigation.subpage = None
        elif page_kind == "item_sub_page":
            navigation.page_kind = "item_page"
            navigation.subpage = None
        return

    asin = str(argument).strip().upper()
    if (
        page_kind == "search_results"
        and re.fullmatch(r"[A-Z0-9]{10}", asin)
    ):
        navigation.asin = asin
        navigation.options = {}
        navigation.page_kind = "item_page"
        navigation.subpage = None
        return

    if page_kind != "item_page":
        return
    if target in {
        normalize_text("Description"),
        normalize_text("Features"),
        normalize_text("Reviews"),
    }:
        navigation.page_kind = "item_sub_page"
        navigation.subpage = target
        return
    if target == normalize_text("Buy Now"):
        navigation.page_kind = "done"
        navigation.subpage = None
        return

    item_options = _public_item_option_groups(observation, available_actions)
    matching_names = {
        normalize_option_value(name)
        for name, values in item_options.items()
        if any(
            normalize_option_value(value) == normalize_option_value(argument)
            for value in values
        )
    }
    if len(matching_names) == 1:
        name = next(iter(matching_names))
        navigation.options[name] = normalize_option_value(argument)


def _current_candidate(
    hard_constraints: Mapping[str, Any],
    navigation: _PublicNavigationState,
) -> Candidate | None:
    asin = navigation.asin
    selected = navigation.options
    if asin is None:
        return None
    selected_normalized = {
        normalize_option_value(key): normalize_option_value(value)
        for key, value in selected.items()
    }
    raw_required = hard_constraints.get("options") or {}
    if not isinstance(raw_required, Mapping):
        raw_required = {}
    required = {
        normalize_option_value(key): normalize_option_value(value)
        for key, value in raw_required.items()
    }
    options_complete = all(
        selected_normalized.get(key) == value
        for key, value in required.items()
    )

    # Reconstruct only an identity from public navigation plus the authorized
    # option contract.  Do not hydrate the current product from the frozen
    # manifest.  Once every required option is selected, unrelated optional
    # controls are projected away so the identity matches the comparison unit.
    # Feasibility remains false here: full methods resolve it from their public
    # ledger, while the no-ledger ablation may promote it only after GATE's
    # independent hard-goal certificate accepts the commitment.
    return Candidate(
        asin=asin,
        options=required if options_complete else selected_normalized,
        feasible=False,
        evidence={
            "identity": {
                "source": "current_public_navigation_state",
                "required_options_complete": options_complete,
            }
        },
    )


def _terminal_candidate_id(
    hard_constraints: Mapping[str, Any],
    navigation: _PublicNavigationState,
    completed: bool,
) -> str | None:
    if not completed:
        return None
    candidate = _current_candidate(hard_constraints, navigation)
    if candidate is not None:
        return candidate.candidate_id

    # Preserve an unknown completed product-option identity for diagnostics.
    asin, options = navigation.asin, navigation.options
    if asin is None:
        return None
    return candidate_identity(asin, options)


def _visible_candidate_ids(
    task: Any,
    observation: Any,
    available_actions: Any,
) -> list[str]:
    visible = visible_asins(observation, available_actions)
    return [
        candidate.candidate_id
        for candidate in task.candidates
        if candidate.asin.upper() in visible
    ]


def _candidate_runtime_signature(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.candidate_id,
        candidate.feasible,
        candidate.price,
        candidate.rating,
        normalize_brand(candidate.brand),
        normalize_text(candidate.title),
        candidate.shortlist_rank,
        candidate.page,
    )


def _seed_public_ledger(
    task: Any,
    comparison_env: Any,
    *,
    hard_constraints: Mapping[str, Any],
    canonical_query: str,
    shortlist_size: int,
    page_size: int,
) -> tuple[CandidateLedger, list[dict[str, Any]]]:
    """Collect the comparison set in an isolated public WebShop session."""

    from .public_ledger import collect_fixed_shortlist

    comparison_env.reset(task.environment_index)
    runtime_candidates, trace = collect_fixed_shortlist(
        comparison_env,
        hard_constraints=hard_constraints,
        canonical_query=canonical_query,
        shortlist_size=shortlist_size,
        page_size=page_size,
    )

    # Frozen annotations are used only after public collection as a fail-closed
    # drift check. They never determine ledger membership.
    frozen = {
        candidate.candidate_id: _candidate_runtime_signature(candidate)
        for candidate in task.candidates
    }
    current = {
        candidate.candidate_id: _candidate_runtime_signature(candidate)
        for candidate in runtime_candidates
    }
    if current != frozen:
        missing = sorted(set(frozen) - set(current))
        extra = sorted(set(current) - set(frozen))
        changed = sorted(
            candidate_id
            for candidate_id in set(frozen) & set(current)
            if frozen[candidate_id] != current[candidate_id]
        )
        raise ProtocolViolation(
            "runtime comparison table drifted from the frozen manifest for "
            f"{task.base_task_id}: missing={missing}, extra={extra}, "
            f"changed={changed}"
        )

    return (
        CandidateLedger.from_candidates(
            runtime_candidates,
            source="public_defense_comparison_session",
            comparison_complete=True,
        ),
        trace,
    )


def _runtime_ledger_entry(
    ledger: CandidateLedger,
    asin: str,
    navigation: _PublicNavigationState,
) -> LedgerEntry | None:
    """Resolve a public ASIN to exactly one frozen product-option identity."""

    entries = ledger.entries_for_asin(asin)
    if len(entries) == 1:
        return entries[0]
    if not navigation.options:
        return None
    selected = {
        normalize_option_value(key): normalize_option_value(value)
        for key, value in navigation.options.items()
    }
    matching = [
        entry
        for entry in entries
        if all(
            selected.get(normalize_option_value(key))
            == normalize_option_value(value)
            for key, value in entry.options
        )
    ]
    return matching[0] if len(matching) == 1 else None


def _runtime_public_scalar_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    title = str(record.get("title") or "").strip()
    brand = normalize_brand(record.get("brand"))
    if title:
        fields["title"] = title
    if brand:
        fields["brand"] = brand
    if record.get("price") is not None:
        fields["price"] = record["price"]
    if record.get("rating") is not None:
        fields["rating"] = record["rating"]
    return fields


def _update_public_ledger_from_policy_observation(
    ledger: CandidateLedger | None,
    observation: Any,
    available_actions: Any,
    navigation: _PublicNavigationState,
) -> list[dict[str, Any]]:
    """Append public trajectory evidence without changing shortlist membership."""

    if ledger is None or not ledger.entries():
        return []

    from .public_ledger import (
        parse_text_item_page,
        parse_text_search_results,
        parse_text_subpage,
    )

    page_kind = _public_page_kind(
        observation,
        available_actions,
        navigation,
    )
    updates: list[dict[str, Any]] = []

    if page_kind == "search_results":
        visible_asin_actions = [
            value
            for value in _public_clickables(available_actions)
            if re.fullmatch(
                r"[A-Z0-9]{10}",
                value.strip(),
                flags=re.IGNORECASE,
            )
        ]
        for record in parse_text_search_results(
            observation,
            allowed_asins=visible_asin_actions,
        ):
            entry = _runtime_ledger_entry(
                ledger,
                str(record.get("asin") or ""),
                navigation,
            )
            if entry is None:
                continue
            source = "policy_public_search_results"
            fields = _runtime_public_scalar_fields(record)
            observed_fields = (
                ("availability",)
                if str(record.get("availability") or "").strip()
                else ()
            )
            if not fields and not observed_fields:
                continue
            evidence = {
                "retrieval": {
                    "page": record.get("page") or navigation.page,
                    "rank": record.get("result_rank"),
                    "source": source,
                },
                "title": {
                    "value": record.get("title"),
                    "source": source,
                },
                "brand": {
                    "value": normalize_brand(record.get("brand")),
                    "source": source,
                },
                "price": {
                    "value": record.get("price"),
                    "source": source,
                },
                "rating": {
                    "value": record.get("rating"),
                    "source": source,
                },
                "availability": {
                    "value": record.get("availability"),
                    "source": source,
                },
            }
            ledger.observe_existing(
                entry,
                source=source,
                fields=fields,
                observed_fields=observed_fields,
                evidence=evidence,
            )
            updates.append(
                {
                    "candidate_id": entry.candidate_id,
                    "asin": entry.asin,
                    "source": source,
                    "fields": sorted(
                        set(fields).union(observed_fields)
                    ),
                }
            )
        return updates

    if page_kind == "item_page":
        record = parse_text_item_page(observation)
        asin = str(record.get("asin") or "").strip().upper()
        if (
            not asin
            or (
                navigation.asin is not None
                and asin != navigation.asin.strip().upper()
            )
        ):
            return []
        entry = _runtime_ledger_entry(ledger, asin, navigation)
        if entry is None:
            return []
        source = "policy_public_item_page"
        fields = _runtime_public_scalar_fields(record)
        observed_fields = (
            ("availability",)
            if str(record.get("availability") or "").strip()
            else ()
        )
        if not fields and not observed_fields:
            return []
        evidence = {
            "title": {
                "value": record.get("title"),
                "source": source,
            },
            "brand": {
                "value": normalize_brand(record.get("brand")),
                "source": source,
            },
            "price": {
                "value": record.get("price"),
                "source": source,
            },
            "rating": {
                "value": record.get("rating"),
                "source": source,
            },
            "availability": {
                "value": record.get("availability"),
                "source": source,
            },
            "options": {
                "value": dict(navigation.options),
                "source": "current_public_navigation_state",
            },
            "page": navigation.page,
        }
        ledger.observe_existing(
            entry,
            source=source,
            fields=fields,
            observed_fields=observed_fields,
            evidence=evidence,
        )
        updates.append(
            {
                "candidate_id": entry.candidate_id,
                "asin": entry.asin,
                "source": source,
                "fields": sorted(set(fields).union(observed_fields)),
            }
        )
        return updates

    if (
        page_kind == "item_sub_page"
        and navigation.asin
        and navigation.subpage
        in {
            normalize_text("Description"),
            normalize_text("Features"),
        }
    ):
        content = parse_text_subpage(observation)
        if not content:
            return []
        entry = _runtime_ledger_entry(
            ledger,
            navigation.asin,
            navigation,
        )
        if entry is None:
            return []
        field_name = navigation.subpage
        source = "policy_public_{}_page".format(field_name)
        value: Any = (
            " ".join(content).strip()
            if field_name == normalize_text("Description")
            else content
        )
        ledger.observe_existing(
            entry,
            source=source,
            observed_fields=(field_name,),
            evidence={
                field_name: {
                    "value": value,
                    "source": source,
                },
                "page": navigation.page,
            },
        )
        updates.append(
            {
                "candidate_id": entry.candidate_id,
                "asin": entry.asin,
                "source": source,
                "fields": [field_name],
            }
        )
    return updates


def _replace_conversation_action(
    chat: Any,
    response: str,
    original_action: str | None,
    executed_action: str,
    replace_action: Any,
) -> str:
    if original_action:
        response = replace_action(
            response=response,
            original_action=original_action,
            masked_action=executed_action,
        )
    else:
        response = (response or "").rstrip() + "\n\nAction:\n" + executed_action
    chat.conv.update_last_message(response)
    return response


def _reset_with_variant(env: Any, task: Any, instruction: str) -> Any:
    """Reset without permanently mutating the shared frozen WebShop goal."""

    original_goal = env.server.goals[task.environment_index]
    episode_goal = dict(original_goal)
    env.server.goals[task.environment_index] = episode_goal
    env.server.assigned_instruction_text = instruction
    try:
        observation, _info = env.reset(task.environment_index)
        return observation
    finally:
        env.server.assigned_instruction_text = None
        env.server.goals[task.environment_index] = original_goal


def _full_gate_certificate_accepted(certification: Any) -> bool:
    """Require every GATE execution check, including progress and stability."""

    return bool(
        certification is not None
        and getattr(getattr(certification, "support", None), "passed", False)
        and getattr(getattr(certification, "progress", None), "passed", False)
        and getattr(getattr(certification, "safety", None), "passed", False)
        and getattr(getattr(certification, "stability", None), "passed", False)
    )


def _gate_action(
    *,
    gate: Any,
    chat: Any,
    action: str | None,
    response: str,
    available_actions: Any,
    replace_action: Any,
    output_mask: Any,
) -> tuple[str | None, str, dict[str, Any], bool]:
    details: dict[str, Any] = {
        "certification": None,
        "projection": None,
        "output_mask": None,
        "safe_repair_accepted": False,
    }
    if not action:
        return action, response, details, False

    original = action
    hard_goal_accepted = not gate.should_certify_action()
    active_certification = None
    if gate.should_certify_action():
        certification = gate.certify_action(action)
        details["certification"] = certification.to_dict()
        hard_goal_accepted = _full_gate_certificate_accepted(certification)
        if hard_goal_accepted:
            active_certification = certification
        elif gate.should_project_action():
            projection = gate.project_action(
                action=action,
                legal_actions=available_actions,
                certification_result=certification,
            )
            details["projection"] = projection.to_dict()
            projected_accepted = bool(
                projection.projected_action
            ) and _full_gate_certificate_accepted(
                projection.projected_certification
            )
            hard_goal_accepted = projected_accepted
            if projected_accepted:
                active_certification = projection.projected_certification
                if projection.changed:
                    response = _replace_conversation_action(
                        chat=chat,
                        response=response,
                        original_action=action,
                        executed_action=projection.projected_action,
                        replace_action=replace_action,
                    )
                action = projection.projected_action
            else:
                # Never execute a projection that still depends on masked
                # evidence or fails any part of the four-check certificate.
                details["projection_blocked"] = True
                details["projection_blocked_reason"] = (
                    "projected_action_failed_full_certificate"
                )
                action = None
        else:
            details["certification_blocked"] = True
            details["certification_blocked_reason"] = (
                "action_failed_full_certificate_and_projection_is_disabled"
            )
            action = None

    if gate.should_mask_output_action() and action:
        # The legacy output helper calls gate.apply() on the action text. Keep
        # its report, but restore the page abstraction before recertification;
        # otherwise the action would be certified against itself rather than
        # against the current WebShop state.
        page_state = gate.last_state_abstraction_result
        try:
            masked_action, report = output_mask(gate, action)
        finally:
            gate.last_state_abstraction_result = page_state
        details["output_mask"] = {
            "original_action": action,
            "masked_action": masked_action,
            "report": report.to_dict() if report is not None else None,
        }
        if masked_action != action:
            masked_certification = None
            if gate.should_certify_action() and masked_action:
                try:
                    masked_certification = gate.certify_action(masked_action)
                except Exception as exc:
                    details["output_mask"]["recertification_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    details["output_mask"]["recertification"] = (
                        masked_certification.to_dict()
                    )
            masked_accepted = bool(
                masked_action
            ) and _full_gate_certificate_accepted(masked_certification)
            details["output_mask"]["accepted"] = masked_accepted
            if masked_accepted:
                response = _replace_conversation_action(
                    chat=chat,
                    response=response,
                    original_action=action,
                    executed_action=masked_action,
                    replace_action=replace_action,
                )
                action = masked_action
                active_certification = masked_certification
                hard_goal_accepted = True
            else:
                details["output_mask"]["blocked"] = True
                details["output_mask"]["blocked_reason"] = (
                    "masked_action_failed_full_certificate"
                )
                if _full_gate_certificate_accepted(active_certification):
                    # The pre-mask action was already certified against this
                    # page state. Keep it and leave the conversation unchanged.
                    details["output_mask"]["fallback_action"] = action
                    details["output_mask"]["fallback_reason"] = (
                        "retained_previously_certified_action"
                    )
                    hard_goal_accepted = True
                else:
                    action = None
                    hard_goal_accepted = False

    details["changed"] = action != original
    details["safe_repair_accepted"] = bool(
        action
        and action != original
        and _full_gate_certificate_accepted(active_certification)
    )
    details["hard_goal_accepted"] = hard_goal_accepted
    return action, response, details, hard_goal_accepted


def _make_method_components(
    method: str,
    settings: EvaluationSettings,
    *,
    offline_verifier: bool,
) -> dict[str, Any]:
    from defenses import GateDefense, GoalContractExtraction

    if method not in METHODS:
        raise ValueError(f"unsupported method {method!r}; choose from {METHODS}")
    use_gate = method in {
        "gate",
        "gate_ci",
        "gate_ci_no_ledger",
        "gate_ci_no_dominance",
    }
    gate = None
    contract_extractor = None
    if use_gate:
        gate = GateDefense(
            use_openai=settings.gate_use_openai,
            openai_model=settings.gate_openai_model,
            ablation="full",
        )
    elif method == "state_aware_verifier":
        contract_extractor = GoalContractExtraction(
            use_openai=settings.gate_use_openai and not offline_verifier,
            openai_model=settings.gate_openai_model,
            temperature=0.0,
        )

    choice_defense: Any = None
    needs_ledger = False
    seed_ledger = False
    if method in {
        "gate_ci",
        "gate_ci_no_ledger",
        "gate_ci_no_dominance",
    }:
        ablation = {
            "gate_ci": "full",
            "gate_ci_no_ledger": "no_ledger",
            "gate_ci_no_dominance": "no_dominance",
        }[method]
        choice_defense = ChoiceIntegrityDefense(
            guard=ChoiceIntegrityGuard(ablation=ablation)
        )
        needs_ledger = ablation != "no_ledger"
        seed_ledger = ablation != "no_ledger"
    elif method == "state_aware_verifier":
        verifier = (
            DeterministicDominanceVerifier()
            if offline_verifier
            else OpenAIStateAwareVerifier(model=settings.state_verifier_model)
        )
        choice_defense = StateAwareVerifierDefense(verifier)
        needs_ledger = True
        seed_ledger = True

    return {
        "gate": gate,
        "contract_extractor": contract_extractor,
        "choice_defense": choice_defense,
        "needs_ledger": needs_ledger,
        "seed_ledger": seed_ledger,
        "offline_verifier": offline_verifier,
    }


def _run_episode(
    *,
    task: Any,
    condition: Condition,
    method: str,
    env: Any,
    chat: Any,
    settings: EvaluationSettings,
    run_id: str,
    cell_id: str,
    attempt_id: str,
    manifest_digest: str,
    steps_path: Path,
    offline_verifier: bool,
    components: Mapping[str, Any],
) -> EpisodeResult:
    from test import (
        extract_action,
        gate_mask_action_value_preserve_format,
        make_json_safe,
        replace_first_action_in_response,
    )

    start_time = time.perf_counter()
    gate = components["gate"]
    choice_defense = components["choice_defense"]
    ledger: CandidateLedger | None = None
    canonical_search_executed = False

    instruction = _variant_instruction(task, condition)
    choice_contract_start = time.perf_counter()
    choice_contract = FixedSuffixPreferenceParser().parse(instruction)
    choice_contract_seconds = time.perf_counter() - choice_contract_start
    if choice_contract.preference is None:
        raise ProtocolViolation(
            f"task {task.base_task_id} variant {condition.value} has no "
            "recognized explicit preference suffix"
        )
    if choice_contract.preference != task.preference:
        raise ProtocolViolation(
            f"task {task.base_task_id} variant preference does not match "
            "the frozen benchmark label"
        )
    runtime_preference = choice_contract.preference
    raw_observation = _reset_with_variant(env, task, instruction)
    target_brand = _target_brand(task)
    observation = _policy_observation(
        raw_observation,
        condition,
        target_brand,
    )
    navigation = _PublicNavigationState()
    chat.reset()

    goal_contract_seconds = 0.0
    goal_contract = None
    if gate is not None:
        goal_contract_start = time.perf_counter()
        # Parse the instruction actually available in this condition. The
        # runtime defender is never given the hidden matched-clean counterpart.
        # The deterministic preference suffix is bound separately so it cannot
        # be reinterpreted as a mandatory hard constraint.
        goal_contract = gate.start_episode(
            instruction,
            goal_only_instruction=choice_contract.base_instruction,
            comparative_preference=(
                choice_contract.preference.to_dict()
                if choice_contract.preference is not None
                else None
            ),
            preference_provenance=(
                choice_contract.provenance.to_dict()
                if choice_contract.provenance is not None
                else None
            ),
        )
        goal_contract_seconds = time.perf_counter() - goal_contract_start
    elif components.get("contract_extractor") is not None:
        goal_contract_start = time.perf_counter()
        goal_contract = components["contract_extractor"].extract(
            choice_contract.base_instruction
        )
        goal_contract.raw_query = instruction
        goal_contract.comparative_preference = (
            choice_contract.preference.to_dict()
            if choice_contract.preference is not None
            else None
        )
        goal_contract.preference_provenance = (
            choice_contract.provenance.to_dict()
            if choice_contract.provenance is not None
            else None
        )
        goal_contract_seconds = time.perf_counter() - goal_contract_start
    if (
        settings.fail_on_gate_parser_fallback
        and goal_contract is not None
        and goal_contract.extraction_error
    ):
        raise ProtocolViolation(
            "authorized goal-contract parser fell back for task "
            f"{task.base_task_id}: {goal_contract.extraction_error}"
        )

    runtime_hard_constraints: dict[str, Any] | None = None
    runtime_canonical_query: str | None = None
    if choice_defense is not None:
        (
            runtime_hard_constraints,
            runtime_canonical_query,
        ) = _runtime_comparison_contract(goal_contract, task)

    trigger_exposed = condition == Condition.DIRECT
    completed = False
    reward_total = 0.0
    interventions = 0
    action_count = 0
    action_overhead = 0
    defense_latency = 0.0
    policy_latency = 0.0
    request_error: str | None = None
    environment_error: str | None = None
    verifier_fallbacks = 0
    first_policy_search: str | None = None
    indirect_query_like_first_search = False
    runtime_ledger_update_count = 0
    runtime_ledger_update_seconds = 0.0
    verifier = getattr(choice_defense, "verifier", None)
    verifier_calls_before = int(getattr(verifier, "calls", 0))
    defense_latency = goal_contract_seconds
    if choice_defense is not None:
        defense_latency += choice_contract_seconds
    comparison_query_seconds = 0.0
    if components["seed_ledger"]:
        comparison_env = components.get("comparison_env")
        if comparison_env is None:
            raise ProtocolViolation(
                "method requires an isolated public comparison environment"
            )
        comparison_start = time.perf_counter()
        ledger, comparison_trace = _seed_public_ledger(
            task,
            comparison_env,
            hard_constraints=runtime_hard_constraints or {},
            canonical_query=runtime_canonical_query or "",
            shortlist_size=components["comparison_shortlist_size"],
            page_size=components["comparison_page_size"],
        )
        comparison_query_seconds = time.perf_counter() - comparison_start
        defense_latency += comparison_query_seconds
        action_overhead = len(comparison_trace)
        canonical_search_executed = True
        _append_jsonl(
            steps_path,
            {
                "base_task_id": task.base_task_id,
                "attempt_id": attempt_id,
                "condition": condition.value,
                "method": method,
                "step": -1,
                "source": "isolated_public_comparison_session",
                "query": runtime_canonical_query,
                "candidate_ids": [
                    entry.candidate_id for entry in ledger.entries()
                ],
                "comparison_actions": comparison_trace,
                "latency_seconds": comparison_query_seconds,
                "counts_as_policy_step": False,
                "action_overhead": action_overhead,
            },
        )
    elif components["needs_ledger"]:
        ledger = CandidateLedger()

    for step_index in range(settings.max_steps):
        available_actions = env.get_available_actions()
        if normalize_text("Attributes") in {
            normalize_text(value)
            for value in _public_clickables(available_actions)
        }:
            raise ProtocolViolation(
                "policy environment exposed WebShop's annotation-backed "
                "Attributes page; choice-integrity runs require show_attrs=False"
            )
        ledger_updates: list[dict[str, Any]] = []
        ledger_update_seconds = 0.0
        if ledger is not None:
            ledger_update_start = time.perf_counter()
            ledger_updates = _update_public_ledger_from_policy_observation(
                ledger,
                raw_observation,
                available_actions,
                navigation,
            )
            ledger_update_seconds = time.perf_counter() - ledger_update_start
            defense_latency += ledger_update_seconds
            runtime_ledger_update_seconds += ledger_update_seconds
            runtime_ledger_update_count += len(ledger_updates)
        if condition == Condition.INDIRECT and _observation_has_brand(
            raw_observation, target_brand
        ):
            trigger_exposed = True

        observed_candidate_ids = _visible_candidate_ids(
            task,
            raw_observation,
            available_actions,
        )

        raw_prompt = (
            f"Observation:\n{observation}\n\n"
            f"Available Actions:\n{available_actions}"
        )
        prompt = raw_prompt
        gate_report = None
        gate_seconds = 0.0
        if gate is not None:
            gate_start = time.perf_counter()
            prompt, gate_report = gate.apply(raw_prompt)
            gate_seconds += time.perf_counter() - gate_start

        try:
            request_start = time.perf_counter()
            response, reference_replacements = chat.request(prompt)
            request_seconds = time.perf_counter() - request_start
            policy_latency += request_seconds
        except Exception as exc:
            request_error = f"{type(exc).__name__}: {exc}"
            _append_jsonl(
                steps_path,
                {
                    "base_task_id": task.base_task_id,
                    "attempt_id": attempt_id,
                    "condition": condition.value,
                    "method": method,
                    "step": step_index,
                    "request_error": request_error,
                },
            )
            raise ProtocolViolation(
                f"policy inference failed for {task.base_task_id}: {request_error}"
            ) from exc

        model_action = extract_action(response)
        operation, argument = action_argument(model_action)
        if operation == "search" and first_policy_search is None:
            first_policy_search = model_action
            if condition == Condition.INDIRECT:
                indirect_query_like_first_search = _observation_has_brand(
                    argument,
                    target_brand,
                )
        action_after_gate = model_action
        hard_goal_accepted = gate is None
        gate_action_details: dict[str, Any] | None = None
        if gate is not None:
            # The helper updates the exact Chat conversation when GATE repairs
            # an action, matching the existing evaluator's trajectory semantics.
            gate_start = time.perf_counter()
            (
                action_after_gate,
                response,
                gate_action_details,
                hard_goal_accepted,
            ) = _gate_action(
                gate=gate,
                chat=chat,
                action=model_action,
                response=response,
                available_actions=available_actions,
                replace_action=replace_first_action_in_response,
                output_mask=gate_mask_action_value_preserve_format,
            )
            gate_seconds += time.perf_counter() - gate_start

        defense_latency += gate_seconds
        executed_action = action_after_gate
        choice_details = None
        choice_seconds = 0.0
        selected_candidate = (
            _current_candidate(
                runtime_hard_constraints or {},
                navigation,
            )
            if choice_defense is not None
            else None
        )
        selected_options = dict(navigation.options)

        if (
            method == "gate_ci_no_ledger"
            and selected_candidate is not None
            and hard_goal_accepted
            and is_commitment_action(action_after_gate)
            and bool(
                selected_candidate.evidence["identity"].get(
                    "required_options_complete",
                    False,
                )
            )
        ):
            # This ablation removes comparison evidence, not the preceding
            # hard-goal check.  Carry only GATE's accepted feasibility bit and
            # the public product-option identity; never borrow a frozen
            # candidate record from the benchmark manifest.
            selected_candidate = Candidate(
                asin=selected_candidate.asin,
                options=selected_candidate.options,
                feasible=True,
                evidence={
                    "feasible": {
                        "source": "preceding_gate_hard_goal_certificate",
                    },
                    "identity": {
                        "source": "current_public_navigation_state",
                    },
                },
            )

        if choice_defense is not None:
            repair_certifier_kwargs: dict[str, Any] = {}
            choice_repair_certifications: list[dict[str, Any]] = []
            if gate is not None:
                def certify_choice_repair(repair_action: str) -> bool:
                    record: dict[str, Any] = {
                        "action": repair_action,
                        "accepted": False,
                        "certificate": None,
                    }
                    if not gate.should_certify_action():
                        record["reason"] = "gate_action_certification_disabled"
                        choice_repair_certifications.append(record)
                        return False
                    try:
                        certification = gate.certify_action(repair_action)
                    except Exception as exc:
                        record["reason"] = "gate_action_certification_error"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        choice_repair_certifications.append(record)
                        return False
                    accepted = _full_gate_certificate_accepted(certification)
                    record["accepted"] = accepted
                    record["certificate"] = certification.to_dict()
                    if not accepted:
                        record["reason"] = "repair_failed_full_certificate"
                    choice_repair_certifications.append(record)
                    return accepted

                repair_certifier_kwargs["repair_certifier"] = (
                    certify_choice_repair
                )

            choice_start = time.perf_counter()
            decision = choice_defense.intercept(
                action_after_gate,
                goal_accepted=hard_goal_accepted,
                goal_repair=(
                    action_after_gate
                    if (
                        gate_action_details is not None
                        and gate_action_details.get("changed")
                        and gate_action_details.get("safe_repair_accepted")
                    )
                    else None
                ),
                selected_candidate=selected_candidate,
                ledger=ledger,
                preference=runtime_preference,
                available_actions=available_actions,
                canonical_query=runtime_canonical_query or "",
                hard_constraints=_jsonable(
                    runtime_hard_constraints or {}
                ),
                selected_options=selected_options,
                current_page=navigation.page,
                **repair_certifier_kwargs,
            )
            choice_seconds = time.perf_counter() - choice_start
            defense_latency += choice_seconds
            executed_action = decision.executed_action
            choice_details = decision.to_dict()
            if (
                gate_action_details is not None
                and choice_repair_certifications
            ):
                gate_action_details["choice_repair_certifications"] = (
                    choice_repair_certifications
                )
            if (
                method == "state_aware_verifier"
                and (
                    "verifier_error:" in decision.choice.reason
                    or "malformed_verifier_result" in decision.choice.reason
                )
            ):
                verifier_fallbacks += 1
                if settings.fail_on_verifier_fallback and not offline_verifier:
                    raise ProtocolViolation(
                        "state-aware verifier failed or returned malformed output "
                        f"for task {task.base_task_id}: {decision.choice.reason}"
                    )

        if executed_action != model_action:
            interventions += 1
        if executed_action and executed_action != action_after_gate:
            response = _replace_conversation_action(
                chat=chat,
                response=response,
                original_action=action_after_gate,
                executed_action=executed_action,
                replace_action=replace_first_action_in_response,
            )

        step_log = {
            "base_task_id": task.base_task_id,
            "attempt_id": attempt_id,
            "condition": condition.value,
            "method": method,
            "step": step_index,
            "source": "policy_step",
            "model_action": model_action,
            "action_after_gate": action_after_gate,
            "executed_action": executed_action,
            "intervened": executed_action != model_action,
            "gate_report": (
                gate_report.to_dict() if gate_report is not None else None
            ),
            "gate_action": gate_action_details,
            "choice_decision": choice_details,
            "ledger": (
                {
                    "comparison_complete": ledger.comparison_complete,
                    "entries": [entry.to_dict() for entry in ledger.entries()],
                }
                if ledger is not None
                else None
            ),
            "observed_candidate_ids": observed_candidate_ids,
            "runtime_ledger_updates": ledger_updates,
            "reference_replacements": reference_replacements,
            "gate_latency_seconds": gate_seconds,
            "choice_latency_seconds": choice_seconds,
            "ledger_update_latency_seconds": ledger_update_seconds,
            "policy_latency_seconds": request_seconds,
            "observation_preview": _truncate(
                raw_observation, settings.debug_text_chars
            ),
            "policy_observation_preview": _truncate(
                observation, settings.debug_text_chars
            ),
            "response_preview": _truncate(response, settings.debug_text_chars),
        }

        if not executed_action:
            step_log["termination_reason"] = "no_executable_action"
            _append_jsonl(steps_path, step_log)
            break

        try:
            next_observation, reward, done, info = env.step(executed_action)
        except Exception as exc:
            environment_error = f"{type(exc).__name__}: {exc}"
            step_log["environment_error"] = environment_error
            _append_jsonl(steps_path, step_log)
            raise ProtocolViolation(
                f"WebShop transition failed for {task.base_task_id}: "
                f"{environment_error}"
            ) from exc

        _record_public_navigation_action(
            navigation,
            executed_action,
            raw_observation,
            available_actions,
        )
        action_count += 1
        reward_total += float(reward or 0.0)
        step_log.update(
            {
                "reward": reward,
                "done": done,
                "env_info": make_json_safe(info),
                "next_observation_preview": _truncate(
                    next_observation, settings.debug_text_chars
                ),
            }
        )
        _append_jsonl(steps_path, step_log)
        raw_observation = next_observation
        observation = _policy_observation(
            raw_observation,
            condition,
            target_brand,
        )
        if done:
            completed = True
            break

    latency = time.perf_counter() - start_time
    terminal_id = _terminal_candidate_id(
        (
            runtime_hard_constraints
            if runtime_hard_constraints is not None
            else task.hard_constraints
        ),
        navigation,
        completed,
    )
    verifier_calls = int(getattr(verifier, "calls", 0)) - verifier_calls_before
    goal_contract_parser = (
        getattr(gate, "goal_contract_extraction", None)
        if gate is not None
        else components.get("contract_extractor")
    )
    metadata = {
        "preference_kind": task.preference.kind.value,
        "choice_contract": choice_contract.to_dict(),
        "runtime_hard_contract": _jsonable(
            runtime_hard_constraints
        ),
        "runtime_canonical_query": runtime_canonical_query,
        "completed_purchase": completed,
        "canonical_search_executed": canonical_search_executed,
        "comparison_query_seconds": comparison_query_seconds,
        "comparison_query_side_effect_free": canonical_search_executed,
        "comparison_uses_isolated_public_session": canonical_search_executed,
        "comparison_action_count": action_overhead,
        "ledger_comparison_complete": (
            ledger.comparison_complete if ledger is not None else None
        ),
        "ledger_candidate_count": len(ledger.entries()) if ledger is not None else 0,
        "runtime_ledger_update_count": runtime_ledger_update_count,
        "runtime_ledger_update_seconds": runtime_ledger_update_seconds,
        "request_error": request_error,
        "environment_error": environment_error,
        "verifier_fallbacks": verifier_fallbacks,
        "verifier_calls": verifier_calls,
        "offline_verifier": offline_verifier,
        "choice_contract_parse_seconds": choice_contract_seconds,
        "goal_contract_parse_seconds": goal_contract_seconds,
        "goal_contract_extractor": (
            getattr(goal_contract, "extractor", None)
            if goal_contract is not None
            else None
        ),
        "goal_contract_extraction_error": (
            getattr(goal_contract, "extraction_error", None)
            if goal_contract is not None
            else None
        ),
        "goal_contract_parser_stats": (
            goal_contract_parser.stats_dict()
            if goal_contract_parser is not None
            and callable(getattr(goal_contract_parser, "stats_dict", None))
            else None
        ),
        "policy_latency_seconds": policy_latency,
        "defense_latency_seconds": defense_latency,
        "benchmark_indirect_eligible": (
            condition == Condition.INDIRECT
            and trigger_exposed
            and not indirect_query_like_first_search
        ),
        "first_policy_search": first_policy_search,
        "indirect_query_like_first_search": (
            indirect_query_like_first_search
        ),
        "max_steps_includes_defense_search": False,
    }
    return EpisodeResult(
        manifest_digest=manifest_digest,
        run_id=run_id,
        cell_id=cell_id,
        base_task_id=task.base_task_id,
        condition=condition,
        method=method,
        terminal_candidate_id=terminal_id,
        trigger_exposed=trigger_exposed,
        reward=max(0.0, min(1.0, reward_total)),
        intervention_count=interventions,
        action_count=action_count,
        action_overhead=action_overhead,
        latency_seconds=latency,
        latency_overhead_seconds=defense_latency,
        log_path=str(steps_path),
        metadata=metadata,
    )


def _resolved_run_config(
    *,
    method: str,
    condition: Condition,
    checkpoint: Path,
    manifest: BenchmarkManifest,
    settings: EvaluationSettings,
    seed: int,
    num_tasks: int,
    offline_verifier: bool,
    environment_sha256: str,
    checkpoint_content_sha256: str,
    checkpoint_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "method": method,
        "condition": condition.value,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_config_sha256": (
            _file_sha256(checkpoint / "config.json")
            if (checkpoint / "config.json").is_file()
            else None
        ),
        "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(checkpoint),
        "checkpoint_content_sha256": checkpoint_content_sha256,
        "checkpoint_provenance": _jsonable(checkpoint_provenance),
        "checkpoint_provenance_sha256": checkpoint_provenance["sha256"],
        "implementation_sha256": _implementation_sha256(repo_root),
        "environment_sha256": environment_sha256,
        "git_sha": _git_sha(repo_root),
        "manifest_digest": manifest.manifest_digest,
        "benchmark_id": manifest.benchmark_id,
        "seed": seed,
        "num_tasks": num_tasks,
        "offline_verifier": offline_verifier,
        "evaluation": settings.to_dict(),
    }


def run_cell(
    *,
    manifest_path: str | Path,
    config: Mapping[str, Any],
    method: str,
    condition: str,
    checkpoint: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    num_tasks: int = -1,
) -> dict[str, Any]:
    """Run or resume one method/condition cell."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    condition_value = Condition(condition)
    settings = EvaluationSettings.from_mapping(config)
    manifest = read_manifest(manifest_path)
    protocol_validation = validate_manifest_protocol(manifest)
    if manifest.metadata.get("public_evidence_protocol") != (
        "rendered_catalog_description_features_options_price_rating_v1"
    ):
        raise ProtocolViolation(
            "manifest predates the annotation-free public-evidence comparison "
            "protocol; rebuild and audit it before evaluation"
        )
    if (
        manifest.metadata.get("page_size") != 10
        or not isinstance(manifest.metadata.get("shortlist_size"), int)
        or manifest.metadata.get("shortlist_size") > 10
    ):
        raise ProtocolViolation(
            "manifest does not use the bounded one-page public shortlist "
            "protocol"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_path}")
    checkpoint_provenance = _validate_checkpoint_provenance(
        checkpoint_path,
        manifest,
    )
    if num_tasks == 0 or num_tasks < -1:
        raise ValueError("num_tasks must be -1 or a positive integer")

    expected_environment = manifest_environment_record(manifest)
    webshop_root = Path(__file__).resolve().parents[1]
    try:
        current_environment = fingerprint_environment(webshop_root)
    except EnvironmentFingerprintError as exc:
        raise ProtocolViolation(str(exc)) from exc
    if _jsonable(current_environment) != _jsonable(expected_environment):
        raise ProtocolViolation(
            "live WebShop catalogue/index content differs from the frozen "
            "benchmark environment; rebuild the benchmark intentionally or "
            "restore the recorded data and index"
        )
    environment_sha256 = str(current_environment["sha256"])

    tasks = list(manifest.tasks)
    if num_tasks != -1:
        tasks = tasks[:num_tasks]
    if not tasks:
        raise ValueError("no benchmark tasks selected")

    offline_verifier = (
        method == "state_aware_verifier"
        and not _as_bool(os.environ.get("CI_REQUIRE_OPENAI", "1"))
    )
    if (
        method == "state_aware_verifier"
        and not offline_verifier
        and not os.environ.get("OPENAI_API_KEY")
    ):
        raise ProtocolViolation(
            "OPENAI_API_KEY is required for the state-aware verifier"
        )
    if (
        method in {"gate", "gate_ci", "gate_ci_no_ledger", "gate_ci_no_dominance"}
        and settings.gate_use_openai
        and not os.environ.get("OPENAI_API_KEY")
    ):
        raise ProtocolViolation("OPENAI_API_KEY is required by configured GATE")

    # Rating restoration must be resolved before WebShop imports and catalogue
    # loading.  The manifest records whether rating tasks were included.
    if any(
        task.preference.kind.value == "rating_max_price_tiebreak"
        for task in tasks
    ):
        ratings_setting = os.environ.get("WEBSHOP_USE_CATALOG_RATINGS")
        if ratings_setting is None:
            os.environ["WEBSHOP_USE_CATALOG_RATINGS"] = "1"
        elif not _as_bool(ratings_setting):
            raise ProtocolViolation(
                "the frozen manifest contains displayed-rating tasks, but "
                "WEBSHOP_USE_CATALOG_RATINGS is disabled"
            )

    benchmark_seed = int(
        manifest.metadata.get(
            "seed",
            config.get("benchmark", {}).get("seed", 42),
        )
    )
    random.seed(benchmark_seed)
    from transformers.trainer_utils import set_seed
    from test import Chat
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    benchmark_section = config.get("benchmark", {})
    target_brand = str(manifest.metadata.get("target_brand") or "")
    category_terms = tuple(manifest.metadata.get("category_terms") or ())
    if not target_brand or not category_terms:
        raise ProtocolViolation(
            "manifest lacks its frozen target-brand/category environment filter"
        )
    configured_brand = benchmark_section.get("target_brand")
    configured_categories = benchmark_section.get("category_terms")
    if (
        configured_brand is not None
        and normalize_brand(configured_brand) != normalize_brand(target_brand)
    ):
        raise ProtocolViolation(
            "runtime config target_brand differs from the frozen manifest"
        )
    if (
        configured_categories is not None
        and tuple(configured_categories) != category_terms
    ):
        raise ProtocolViolation(
            "runtime config category_terms differ from the frozen manifest"
        )
    env = WebAgentTextEnv(
        observation_mode="text",
        human_goals=False,
        show_attrs=False,
        public_fields=True,
        filter_goals=make_goal_filter(category_terms, target_brand),
    )
    comparison_env = None
    if method in {
        "state_aware_verifier",
        "gate_ci",
        "gate_ci_no_dominance",
    }:
        comparison_env = WebAgentTextEnv(
            observation_mode="html",
            server=env.server,
            session_prefix="ci_compare_",
        )
    # Environment/catalog randomness belongs to the frozen benchmark. Policy
    # repetition seeds are applied only after environment construction.
    set_seed(seed)

    original_goal_instructions = {
        index: str(goal.get("instruction_text") or "")
        for index, goal in enumerate(env.server.goals)
    }
    for task in tasks:
        if task.environment_index not in original_goal_instructions:
            raise ValueError(
                f"task {task.base_task_id} environment index is out of range"
            )
        actual = original_goal_instructions[task.environment_index].strip()
        if actual != task.original_instruction.strip():
            raise ProtocolViolation(
                "frozen task no longer matches WebShop goal ordering: "
                f"{task.base_task_id}"
            )

    chat = Chat(
        cpk=str(checkpoint_path),
        gpu=0,
        defense="none",
        max_input_length=settings.max_input_length,
        max_new_tokens=settings.max_new_tokens,
    )
    components = _make_method_components(
        method,
        settings,
        offline_verifier=offline_verifier,
    )
    components["comparison_env"] = comparison_env
    components["comparison_shortlist_size"] = int(
        manifest.metadata["shortlist_size"]
    )
    components["comparison_page_size"] = int(
        manifest.metadata["page_size"]
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "episodes.jsonl"
    steps_path = output / "steps.jsonl"
    resolved_path = output / "resolved_config.json"
    provenance_path = output / "provenance.json"
    summary_path = output / "summary.json"
    success_path = output / "_SUCCESS.json"

    run_id = _run_id()
    cell_id = f"{method}:{condition_value.value}:seed_{seed}"
    checkpoint_cache_directory = (
        output.parents[3]
        if len(output.parents) > 3 and output.parents[2].name == "cells"
        else output
    )
    checkpoint_content_sha256 = _checkpoint_content_sha256(
        checkpoint_path,
        cache_directory=checkpoint_cache_directory,
    )
    resolved = _resolved_run_config(
        method=method,
        condition=condition_value,
        checkpoint=checkpoint_path,
        manifest=manifest,
        settings=settings,
        seed=seed,
        num_tasks=len(tasks),
        offline_verifier=offline_verifier,
        environment_sha256=environment_sha256,
        checkpoint_content_sha256=checkpoint_content_sha256,
        checkpoint_provenance=checkpoint_provenance,
    )
    if resolved_path.exists():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous != resolved:
            raise ProtocolViolation(
                f"resume configuration mismatch in {resolved_path}"
            )
    else:
        _atomic_json(resolved_path, resolved)

    provenance = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": resolved["git_sha"],
        "implementation_sha256": resolved["implementation_sha256"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_digest": manifest.manifest_digest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_config_sha256": resolved["checkpoint_config_sha256"],
        "checkpoint_metadata_sha256": resolved[
            "checkpoint_metadata_sha256"
        ],
        "checkpoint_content_sha256": resolved[
            "checkpoint_content_sha256"
        ],
        "checkpoint_provenance": resolved["checkpoint_provenance"],
        "environment_sha256": resolved["environment_sha256"],
        "run_id": run_id,
        "cell_id": cell_id,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST",
            )
        },
        "catalog_ratings_enabled": os.environ.get(
            "WEBSHOP_USE_CATALOG_RATINGS"
        ),
        "public_evidence_protocol": manifest.metadata[
            "public_evidence_protocol"
        ],
        "openai_protocol_required": (
            (
                method
                in {
                    "gate",
                    "gate_ci",
                    "gate_ci_no_ledger",
                    "gate_ci_no_dominance",
                }
                and settings.gate_use_openai
            )
            or (
                method == "state_aware_verifier"
                and not offline_verifier
            )
        ),
        "paired_condition_protocol": protocol_validation,
    }
    if provenance_path.exists():
        _append_jsonl(output / "resume_events.jsonl", provenance)
    else:
        _atomic_json(provenance_path, provenance)

    existing = _read_episode_jsonl(episodes_path)
    completed_ids = {
        row.base_task_id
        for row in existing
        if (
            row.manifest_digest == manifest.manifest_digest
            and row.run_id == run_id
            and row.cell_id == cell_id
            and row.method == method
            and row.condition == condition_value
        )
    }
    if len(completed_ids) != len(existing):
        raise ProtocolViolation(
            f"{episodes_path} contains rows from a different cell/configuration"
        )
    expected_task_ids = {task.base_task_id for task in tasks}
    if not completed_ids.issubset(expected_task_ids):
        raise ProtocolViolation(
            f"{episodes_path} contains tasks outside the resolved task set"
        )
    if success_path.exists():
        # A resubmitted cell is incomplete until this invocation has validated
        # all existing rows and rewritten the terminal marker.
        success_path.unlink()

    for task_number, task in enumerate(tasks, start=1):
        if task.base_task_id in completed_ids:
            print(
                f"[{task_number}/{len(tasks)}] resume skip {task.base_task_id}",
                flush=True,
            )
            continue
        print(
            f"[{task_number}/{len(tasks)}] {method}/{condition_value.value} "
            f"{task.base_task_id}",
            flush=True,
        )
        result = _run_episode(
            task=task,
            condition=condition_value,
            method=method,
            env=env,
            chat=chat,
            settings=settings,
            run_id=run_id,
            cell_id=cell_id,
            attempt_id=(
                f"{cell_id}:{task.base_task_id}:{time.time_ns()}"
            ),
            manifest_digest=manifest.manifest_digest,
            steps_path=steps_path,
            offline_verifier=offline_verifier,
            components=components,
        )
        _append_jsonl(episodes_path, result.to_dict())
        existing.append(result)

    metrics = aggregate_metrics(existing, manifest)
    ci_by_cell = clustered_bootstrap_ci(
        existing,
        manifest,
        n_resamples=settings.bootstrap_samples,
        seed=settings.bootstrap_seed,
    )
    ci = {
        "|".join(key): value
        for key, value in ci_by_cell.items()
    }
    summary = {
        "run_id": run_id,
        "cell_id": cell_id,
        "method": method,
        "condition": condition_value.value,
        "manifest_digest": manifest.manifest_digest,
        "completed_tasks": len(existing),
        "expected_tasks": len(tasks),
        "metrics": metrics,
        "bootstrap_95_ci": ci,
        "by_preference": _preference_slices(
            existing,
            manifest,
            settings,
        ),
    }
    _atomic_json(summary_path, summary)
    _atomic_json(
        success_path,
        {
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "episodes": len(existing),
            "expected_episodes": len(tasks),
            "task_ids_sha256": hashlib.sha256(
                "\n".join(sorted(expected_task_ids)).encode("utf-8")
            ).hexdigest(),
            "manifest_digest": manifest.manifest_digest,
            "cell_id": cell_id,
            "episodes_sha256": _file_sha256(episodes_path),
            "resolved_config_sha256": _file_sha256(resolved_path),
        },
    )
    return summary


def _all_episode_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("cells/**/episodes.jsonl"))


def _all_cell_directories(run_dir: Path) -> list[Path]:
    """Discover attempted cells even when no episode row was written."""

    cells_root = run_dir / "cells"
    directories = {
        path
        for path in cells_root.glob("*/*/*")
        if path.is_dir()
    }
    for artifact_name in (
        "episodes.jsonl",
        "resolved_config.json",
        "_SUCCESS.json",
    ):
        directories.update(
            path.parent
            for path in cells_root.glob(f"**/{artifact_name}")
        )
    return sorted(directories)


def _preference_slices(
    rows: Sequence[EpisodeResult],
    manifest: BenchmarkManifest,
    settings: EvaluationSettings,
) -> dict[str, dict[str, Any]]:
    task_kind = {
        task.base_task_id: task.preference.kind.value
        for task in manifest.tasks
    }
    output: dict[str, dict[str, Any]] = {}
    for kind in sorted(set(task_kind.values())):
        subset = [row for row in rows if task_kind[row.base_task_id] == kind]
        if not subset:
            continue
        output[kind] = {
            "metrics": aggregate_metrics(subset, manifest),
            "bootstrap_95_ci": {
                "|".join(key): value
                for key, value in clustered_bootstrap_ci(
                    subset,
                    manifest,
                    n_resamples=settings.bootstrap_samples,
                    seed=settings.bootstrap_seed,
                ).items()
            },
        }
    return output


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RESOLVED_FIELDS = frozenset(
    {
        "method",
        "condition",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_metadata_sha256",
        "checkpoint_content_sha256",
        "checkpoint_provenance",
        "checkpoint_provenance_sha256",
        "implementation_sha256",
        "environment_sha256",
        "git_sha",
        "manifest_digest",
        "benchmark_id",
        "seed",
        "num_tasks",
        "offline_verifier",
        "evaluation",
    }
)
_REQUIRED_RESOLVED_FINGERPRINT_FIELDS = (
    "checkpoint_config_sha256",
    "checkpoint_metadata_sha256",
    "checkpoint_content_sha256",
    "checkpoint_provenance_sha256",
    "implementation_sha256",
    "environment_sha256",
)
_CELL_SPECIFIC_PROTOCOL_FIELDS = frozenset(
    {"method", "condition", "seed", "offline_verifier"}
)


def _require_sha256(
    record: Mapping[str, Any],
    field_name: str,
    *,
    source: Path,
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolViolation(
            f"{source} lacks a valid {field_name} SHA-256 fingerprint"
        )
    return value


def _validate_completed_resolved_config(
    resolved: Mapping[str, Any],
    *,
    source: Path,
    manifest: BenchmarkManifest,
) -> str:
    missing = sorted(_REQUIRED_RESOLVED_FIELDS - set(resolved))
    if missing:
        raise ProtocolViolation(
            f"{source} lacks required resolved fields: {missing}"
        )
    for field_name in _REQUIRED_RESOLVED_FINGERPRINT_FIELDS:
        _require_sha256(resolved, field_name, source=source)

    method = resolved["method"]
    if not isinstance(method, str) or method not in METHODS:
        raise ProtocolViolation(f"{source} has an invalid method: {method!r}")
    try:
        condition = Condition(resolved["condition"])
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(
            f"{source} has an invalid condition: "
            f"{resolved['condition']!r}"
        ) from exc
    seed = resolved["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ProtocolViolation(
            f"{source} has an invalid resolved seed: {seed!r}"
        )
    num_tasks = resolved["num_tasks"]
    if (
        isinstance(num_tasks, bool)
        or not isinstance(num_tasks, int)
        or num_tasks <= 0
    ):
        raise ProtocolViolation(
            f"{source} has an invalid resolved task count: {num_tasks!r}"
        )
    checkpoint = resolved["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ProtocolViolation(
            f"{source} lacks a non-empty checkpoint identity"
        )
    if resolved["benchmark_id"] != manifest.benchmark_id:
        raise ProtocolViolation(
            f"{source} identifies a different benchmark"
        )
    expected_environment_sha256 = str(
        manifest_environment_record(manifest)["sha256"]
    )
    if resolved["environment_sha256"] != expected_environment_sha256:
        raise ProtocolViolation(
            f"{source} identifies a different WebShop environment"
        )
    offline_verifier = resolved["offline_verifier"]
    if not isinstance(offline_verifier, bool):
        raise ProtocolViolation(
            f"{source} has a non-boolean offline_verifier setting"
        )
    if offline_verifier and method != "state_aware_verifier":
        raise ProtocolViolation(
            f"{source} enables offline_verifier for method {method!r}"
        )

    evaluation = resolved["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise ProtocolViolation(
            f"{source} lacks a complete evaluation protocol"
        )
    required_evaluation_fields = set(
        EvaluationSettings.__dataclass_fields__
    )
    missing_evaluation = sorted(
        required_evaluation_fields - set(evaluation)
    )
    if missing_evaluation:
        raise ProtocolViolation(
            f"{source} lacks evaluation fields: {missing_evaluation}"
        )
    try:
        EvaluationSettings.from_mapping({"evaluation": evaluation})
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(
            f"{source} has an invalid evaluation protocol: {exc}"
        ) from exc

    return f"{method}:{condition.value}:seed_{seed}"


def _shared_evaluation_protocol(
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Return every resolved field that must be identical across cells."""

    return {
        key: value
        for key, value in resolved.items()
        if key not in _CELL_SPECIFIC_PROTOCOL_FIELDS
    }


def aggregate_run(
    *,
    manifest_path: str | Path,
    config: Mapping[str, Any],
    run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate every currently completed/resumable cell under a run."""

    manifest = read_manifest(manifest_path)
    protocol_validation = validate_manifest_protocol(manifest)
    settings = EvaluationSettings.from_mapping(config)
    run_root = Path(run_dir)
    rows: list[EpisodeResult] = []
    all_episode_files = _all_episode_files(run_root)
    all_cell_directories = _all_cell_directories(run_root)
    files: list[Path] = []
    ignored_partial_episode_files: list[str] = []
    ignored_incomplete_cell_directories: list[str] = []
    observed_seed_names: set[str] = set()
    unattributed_partial_episode_files: list[str] = []
    unattributed_incomplete_cell_directories: list[str] = []
    marker_records: list[dict[str, Any]] = []
    resolved_configs: list[dict[str, Any]] = []

    cells_root = run_root / "cells"
    valid_conditions = {condition.value for condition in Condition}
    for cell_directory in all_cell_directories:
        try:
            relative_parts = cell_directory.relative_to(cells_root).parts
        except ValueError:
            relative_parts = ()
        canonical_layout = (
            len(relative_parts) == 3
            and relative_parts[0] in METHODS
            and relative_parts[1] in valid_conditions
            and re.fullmatch(r"seed_\d+", relative_parts[2]) is not None
        )
        if canonical_layout:
            observed_seed_names.add(relative_parts[2])
        else:
            unattributed_incomplete_cell_directories.append(
                str(cell_directory)
            )

        required_artifacts = (
            cell_directory / "episodes.jsonl",
            cell_directory / "resolved_config.json",
            cell_directory / "_SUCCESS.json",
        )
        if not all(path.is_file() for path in required_artifacts):
            ignored_incomplete_cell_directories.append(
                str(cell_directory)
            )

    for path in all_episode_files:
        seed_directory = path.parent.name
        if re.fullmatch(r"seed_\d+", seed_directory):
            observed_seed_names.add(seed_directory)
        marker_path = path.parent / "_SUCCESS.json"
        resolved_path = path.parent / "resolved_config.json"
        if not marker_path.is_file() or not resolved_path.is_file():
            ignored_partial_episode_files.append(str(path))
            if not re.fullmatch(r"seed_\d+", seed_directory):
                unattributed_partial_episode_files.append(str(path))
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"invalid completion metadata beside {path}: {exc}"
            ) from exc
        if not isinstance(marker, Mapping):
            raise ProtocolViolation(
                f"completion marker is not a JSON object: {marker_path}"
            )
        episodes_sha256 = _require_sha256(
            marker,
            "episodes_sha256",
            source=marker_path,
        )
        resolved_config_sha256 = _require_sha256(
            marker,
            "resolved_config_sha256",
            source=marker_path,
        )
        if episodes_sha256 != _file_sha256(path):
            raise ProtocolViolation(
                f"completed episode artifact changed after success: {path}"
            )
        if resolved_config_sha256 != _file_sha256(resolved_path):
            raise ProtocolViolation(
                "resolved configuration changed after success: "
                f"{resolved_path}"
            )
        try:
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"invalid completion metadata beside {path}: {exc}"
            ) from exc
        if not isinstance(resolved, Mapping):
            raise ProtocolViolation(
                f"resolved configuration is not a JSON object: {resolved_path}"
            )
        canonical_cell_id = _validate_completed_resolved_config(
            resolved,
            source=resolved_path,
            manifest=manifest,
        )
        expected_cell_path = (
            str(resolved["method"]),
            str(resolved["condition"]),
            f"seed_{resolved['seed']}",
        )
        actual_cell_path = (
            path.parent.parent.parent.name,
            path.parent.parent.name,
            path.parent.name,
        )
        if actual_cell_path != expected_cell_path:
            raise ProtocolViolation(
                "completed cell is stored under a non-canonical path: "
                f"expected {expected_cell_path!r}, got "
                f"{actual_cell_path!r}"
            )
        if marker.get("cell_id") != canonical_cell_id:
            raise ProtocolViolation(
                "success marker has a non-canonical cell ID: "
                f"expected {canonical_cell_id!r}, got "
                f"{marker.get('cell_id')!r}"
            )
        if (
            marker.get("manifest_digest") != manifest.manifest_digest
            or resolved.get("manifest_digest") != manifest.manifest_digest
        ):
            raise ProtocolViolation(
                f"completed cell uses a different manifest: {path.parent}"
            )
        cell_rows = _read_episode_jsonl(path)
        expected_count = int(resolved.get("num_tasks", -1))
        if expected_count <= 0 or expected_count > len(manifest.tasks):
            raise ProtocolViolation(
                f"resolved cell has an invalid task count: {resolved_path}"
            )
        expected_ids = {
            task.base_task_id for task in manifest.tasks[:expected_count]
        }
        actual_ids = {row.base_task_id for row in cell_rows}
        expected_id_digest = hashlib.sha256(
            "\n".join(sorted(expected_ids)).encode("utf-8")
        ).hexdigest()
        if (
            len(cell_rows) != expected_count
            or len(actual_ids) != len(cell_rows)
            or actual_ids != expected_ids
            or marker.get("episodes") != len(cell_rows)
            or marker.get("expected_episodes") != expected_count
            or marker.get("task_ids_sha256") != expected_id_digest
        ):
            raise ProtocolViolation(
                f"stale or inconsistent success marker for {path}"
        )
        if any(
            row.cell_id != canonical_cell_id
            or row.method != resolved.get("method")
            or row.condition.value != resolved.get("condition")
            for row in cell_rows
        ):
            raise ProtocolViolation(
                f"episode rows do not match their cell metadata: {path}"
            )
        if any(row.reward is None for row in cell_rows):
            raise ProtocolViolation(
                f"completed paper cell contains a missing reward: {path}"
            )
        marker_records.append(marker)
        resolved_configs.append(resolved)
        files.append(path)
        rows.extend(cell_rows)
    if not rows:
        raise ValueError(
            f"no successfully completed episode cells found under {run_root}"
        )
    run_ids = {row.run_id for row in rows}
    if len(run_ids) != 1:
        raise ProtocolViolation(
            f"aggregate directory mixes run IDs: {sorted(run_ids)}"
        )

    shared_protocol = _shared_evaluation_protocol(resolved_configs[0])
    for resolved in resolved_configs[1:]:
        candidate_protocol = _shared_evaluation_protocol(resolved)
        if candidate_protocol != shared_protocol:
            changed_fields = sorted(
                key
                for key in set(shared_protocol) | set(candidate_protocol)
                if shared_protocol.get(key) != candidate_protocol.get(key)
            )
            raise ProtocolViolation(
                "completed cells use different evaluation protocols; "
                f"mismatched fields: {changed_fields}"
            )
    if shared_protocol.get("evaluation") != settings.to_dict():
        raise ProtocolViolation(
            "aggregation config does not match the evaluation protocol "
            "frozen in the completed cells"
        )
    offline_verifier_by_method: dict[str, bool] = {}
    for resolved in resolved_configs:
        method = resolved["method"]
        offline_verifier = resolved["offline_verifier"]
        previous = offline_verifier_by_method.setdefault(
            method,
            offline_verifier,
        )
        if previous != offline_verifier:
            raise ProtocolViolation(
                "completed cells use inconsistent method-specific verifier "
                f"protocols for {method!r}"
            )

    checkpoint_hashes = {
        value["checkpoint_config_sha256"]
        for value in resolved_configs
    }
    checkpoint_paths = {
        value["checkpoint"]
        for value in resolved_configs
    }
    checkpoint_metadata_hashes = {
        value["checkpoint_metadata_sha256"]
        for value in resolved_configs
    }
    checkpoint_content_hashes = {
        value["checkpoint_content_sha256"]
        for value in resolved_configs
    }
    checkpoint_provenance_hashes = {
        value["checkpoint_provenance_sha256"]
        for value in resolved_configs
    }
    implementation_hashes = {
        value["implementation_sha256"]
        for value in resolved_configs
    }
    environment_hashes = {
        value["environment_sha256"]
        for value in resolved_configs
    }
    if len(implementation_hashes) != 1:
        raise ProtocolViolation(
            "completed cells were produced by different implementation states: "
            f"{sorted(implementation_hashes)}"
        )
    current_implementation_hash = _implementation_sha256(
        Path(__file__).resolve().parents[4]
    )
    if implementation_hashes != {current_implementation_hash}:
        raise ProtocolViolation(
            "completed cells were produced by a different implementation "
            "than the code performing this aggregation; rerun aggregation "
            "from the recorded implementation state"
        )
    expected_environment_hash = str(
        manifest_environment_record(manifest)["sha256"]
    )
    if environment_hashes != {expected_environment_hash}:
        raise ProtocolViolation(
            "completed cells were produced from a different WebShop "
            "catalogue/index environment than the frozen benchmark"
        )
    if settings.require_shared_checkpoint and (
        len(checkpoint_paths) != 1
        or len(checkpoint_hashes) != 1
        or len(checkpoint_metadata_hashes) != 1
        or len(checkpoint_content_hashes) != 1
        or len(checkpoint_provenance_hashes) != 1
    ):
        raise ProtocolViolation(
            "completed cells do not identify exactly one shared policy "
            "checkpoint; the paper protocol requires one compromised model "
            f"(paths={sorted(checkpoint_paths)}, "
            f"config_hashes={sorted(checkpoint_hashes)}, "
            f"metadata_hashes={sorted(checkpoint_metadata_hashes)}, "
            f"content_hashes={sorted(checkpoint_content_hashes)}, "
            f"provenance_hashes={sorted(checkpoint_provenance_hashes)})"
        )

    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.cell_id, row.base_task_id, row.manifest_digest)
        if key in seen:
            raise ProtocolViolation(
                "duplicate aggregate episode for "
                f"cell={row.cell_id} task={row.base_task_id}"
            )
        seen.add(key)

    by_cell = aggregate_by_cell(rows, manifest)
    ci_by_cell = clustered_bootstrap_ci(
        rows,
        manifest,
        n_resamples=settings.bootstrap_samples,
        seed=settings.bootstrap_seed,
    )
    cells: list[dict[str, Any]] = []
    for key in sorted(by_cell):
        metrics = by_cell[key]
        cell_rows = [
            row
            for row in rows
            if (
                row.run_id,
                row.cell_id,
                row.condition.value,
                row.method,
            )
            == key
        ]
        cells.append(
            {
                "run_id": key[0],
                "cell_id": key[1],
                "condition": key[2],
                "method": key[3],
                "metrics": metrics,
                "bootstrap_95_ci": ci_by_cell.get(key, {}),
                "by_preference": _preference_slices(
                    cell_rows,
                    manifest,
                    settings,
                ),
            }
        )

    completed_cell_ids: list[str] = []
    full_completed_cell_ids: list[str] = []
    for marker in marker_records:
        cell_id = marker.get("cell_id")
        if not isinstance(cell_id, str):
            raise ValueError("success marker lacks a valid cell_id")
        completed_cell_ids.append(cell_id)
        if marker.get("episodes") == len(manifest.tasks):
            full_completed_cell_ids.append(cell_id)

    expected_main_pairs = {
        f"{method}:{condition.value}"
        for method in MAIN_METHODS
        for condition in Condition
    }
    expected_ablation_pairs = {
        f"{method}:{condition.value}"
        for method in METHODS
        if method not in MAIN_METHODS
        for condition in Condition
    }
    expected_full_pairs = expected_main_pairs | expected_ablation_pairs

    completed_by_seed: dict[str, set[str]] = {}
    for cell_id in full_completed_cell_ids:
        parts = cell_id.split(":")
        if len(parts) != 3 or not parts[2].startswith("seed_"):
            raise ProtocolViolation(f"malformed cell ID in success marker: {cell_id}")
        completed_by_seed.setdefault(parts[2], set()).add(
            f"{parts[0]}:{parts[1]}"
        )
    for seed_name in observed_seed_names:
        completed_by_seed.setdefault(seed_name, set())
    seed_completeness = {
        seed_name: {
            "completed_method_condition_pairs": sorted(pairs),
            "missing_main_method_condition_pairs": sorted(
                expected_main_pairs - pairs
            ),
            "missing_ablation_method_condition_pairs": sorted(
                expected_ablation_pairs - pairs
            ),
            "complete_main_matrix": expected_main_pairs.issubset(pairs),
            "complete_full_matrix": expected_full_pairs.issubset(pairs),
        }
        for seed_name, pairs in sorted(completed_by_seed.items())
    }
    complete_main_matrix = (
        bool(seed_completeness)
        and not unattributed_partial_episode_files
        and not unattributed_incomplete_cell_directories
        and all(
            value["complete_main_matrix"]
            for value in seed_completeness.values()
        )
    )
    complete_full_matrix = (
        bool(seed_completeness)
        and not unattributed_partial_episode_files
        and not unattributed_incomplete_cell_directories
        and all(
            value["complete_full_matrix"]
            for value in seed_completeness.values()
        )
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "manifest_digest": manifest.manifest_digest,
        "episode_files": [str(path) for path in files],
        "ignored_partial_episode_files": ignored_partial_episode_files,
        "ignored_incomplete_cell_directories": (
            ignored_incomplete_cell_directories
        ),
        "unattributed_partial_episode_files": (
            unattributed_partial_episode_files
        ),
        "unattributed_incomplete_cell_directories": (
            unattributed_incomplete_cell_directories
        ),
        "observed_seed_names": sorted(observed_seed_names),
        "episodes": len(rows),
        "cells": cells,
        "completed_cell_ids": sorted(completed_cell_ids),
        "full_task_completed_cell_ids": sorted(full_completed_cell_ids),
        "expected_tasks_per_full_cell": len(manifest.tasks),
        "expected_main_method_condition_pairs": sorted(expected_main_pairs),
        "expected_ablation_method_condition_pairs": sorted(
            expected_ablation_pairs
        ),
        "seed_completeness": seed_completeness,
        "complete_main_matrix": complete_main_matrix,
        "complete_full_matrix": complete_full_matrix,
        "checkpoint_config_sha256": sorted(checkpoint_hashes),
        "checkpoint_paths": sorted(checkpoint_paths),
        "checkpoint_metadata_sha256": sorted(
            checkpoint_metadata_hashes
        ),
        "checkpoint_content_sha256": sorted(
            checkpoint_content_hashes
        ),
        "checkpoint_provenance_sha256": sorted(
            checkpoint_provenance_hashes
        ),
        "implementation_sha256": sorted(implementation_hashes),
        "aggregation_implementation_sha256": current_implementation_hash,
        "environment_sha256": sorted(environment_hashes),
        "shared_checkpoint_required": settings.require_shared_checkpoint,
        "aggregation_settings": {
            "bootstrap_samples": settings.bootstrap_samples,
            "bootstrap_seed": settings.bootstrap_seed,
            "cluster_unit": "webshop_environment_index",
        },
        "paired_condition_protocol": protocol_validation,
    }
    _atomic_json(output / "summary.json", result)

    fieldnames = [
        "run_id",
        "cell_id",
        "condition",
        "method",
        "preference_kind",
        "episodes",
        "pl_asr",
        "pl_asr_ci_low",
        "pl_asr_ci_high",
        "pl_asr_denominator",
        "pl_asr_unconditioned",
        "pl_asr_unconditioned_ci_low",
        "pl_asr_unconditioned_ci_high",
        "pl_asr_unconditioned_denominator",
        "preference_satisfaction",
        "preference_satisfaction_ci_low",
        "preference_satisfaction_ci_high",
        "preference_satisfaction_denominator",
        "price_regret",
        "price_regret_ci_low",
        "price_regret_ci_high",
        "price_regret_denominator",
        "rating_regret",
        "rating_regret_ci_low",
        "rating_regret_ci_high",
        "rating_regret_denominator",
        "aer",
        "aer_ci_low",
        "aer_ci_high",
        "aer_denominator",
        "preference_flip",
        "preference_flip_ci_low",
        "preference_flip_ci_high",
        "preference_flip_denominator",
        "targeted_preference_flip",
        "targeted_preference_flip_ci_low",
        "targeted_preference_flip_ci_high",
        "targeted_preference_flip_denominator",
        "intervention_rate",
        "clean_intervention_rate",
        "clean_intervention_rate_ci_low",
        "clean_intervention_rate_ci_high",
        "clean_intervention_rate_denominator",
        "mean_action_overhead",
        "mean_action_overhead_ci_low",
        "mean_action_overhead_ci_high",
        "mean_latency_overhead_seconds",
        "mean_latency_overhead_seconds_ci_low",
        "mean_latency_overhead_seconds_ci_high",
    ]

    def interval(bundle: Mapping[str, Any], metric_name: str) -> tuple[Any, Any]:
        if "ci95" in bundle:
            item = bundle.get("ci95", {}).get(metric_name, {})
            return item.get("low"), item.get("high")
        if bundle:
            first = next(iter(bundle.values()))
            item = first.get("ci95", {}).get(metric_name, {})
            return item.get("low"), item.get("high")
        return None, None

    csv_tmp = output / "summary.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            slices = {
                "all": {
                    "metrics": cell["metrics"],
                    "bootstrap_95_ci": cell["bootstrap_95_ci"],
                },
                **cell["by_preference"],
            }
            for preference_kind, slice_data in slices.items():
                metric = slice_data["metrics"]
                ci_bundle = slice_data["bootstrap_95_ci"]
                row = {
                    "run_id": cell["run_id"],
                    "cell_id": cell["cell_id"],
                    "condition": cell["condition"],
                    "method": cell["method"],
                    "preference_kind": preference_kind,
                }
                for name in (
                    "episodes",
                    "pl_asr",
                    "pl_asr_denominator",
                    "pl_asr_unconditioned",
                    "pl_asr_unconditioned_denominator",
                    "preference_satisfaction",
                    "preference_satisfaction_denominator",
                    "price_regret",
                    "price_regret_denominator",
                    "rating_regret",
                    "rating_regret_denominator",
                    "aer",
                    "aer_denominator",
                    "preference_flip",
                    "preference_flip_denominator",
                    "targeted_preference_flip",
                    "targeted_preference_flip_denominator",
                    "intervention_rate",
                    "clean_intervention_rate",
                    "clean_intervention_rate_denominator",
                    "mean_action_overhead",
                    "mean_latency_overhead_seconds",
                ):
                    row[name] = metric.get(name)
                for name in (
                    "pl_asr",
                    "pl_asr_unconditioned",
                    "preference_satisfaction",
                    "price_regret",
                    "rating_regret",
                    "aer",
                    "preference_flip",
                    "targeted_preference_flip",
                    "clean_intervention_rate",
                    "mean_action_overhead",
                    "mean_latency_overhead_seconds",
                ):
                    low, high = interval(ci_bundle, name)
                    row[f"{name}_ci_low"] = low
                    row[f"{name}_ci_high"] = high
                writer.writerow(row)
    csv_tmp.replace(output / "summary.csv")
    return result
