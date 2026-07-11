"""
Module 1: Goal Contract Extraction.

Given a raw user query q, this module extracts the goal contract

    G(q) = (I, C+, C-)

where I is the high-level task intent, C+ is the set of explicit positive
constraints, and C- is the set of explicit forbidden constraints/actions.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - WebShop experiments run on Linux.
    fcntl = None


_NEGATIVE_MARKERS = (
    "do not",
    "don't",
    "dont",
    "never",
    "without",
    "avoid",
    "exclude",
    "no ",
    "not ",
)

_NEGATIVE_CONSTRAINT_RE = re.compile(
    r"\b(do not|don't|dont|never|without|avoid|exclude|no|not)\s+([^.;,]+)",
    flags=re.I,
)


class GoalContractParseError(RuntimeError):
    """Raised when fail-fast goal-contract extraction cannot use the requested model."""


def goal_contract_cache_key(instruction: str, parser_model: str) -> str:
    """Hash the exact original instruction and requested parser model."""

    payload = json.dumps(
        [instruction, parser_model],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _cache_lock(path: Path):
    """Serialize cache updates across Slurm array processes when flock is available."""

    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _cache_key_lock(path: Path, key: str):
    """Single-flight one parser call for a specific instruction/model cache key."""

    call_lock_path = Path(f"{path}.{key}.call.lock")
    call_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with call_lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class GoalContract:
    """Structured goal contract G(q) = (I, C+, C-)."""

    raw_query: str
    intent: str
    positive_constraints: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)
    extractor: str = "unknown"
    extraction_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_instruction"] = self.raw_query
        data["parser"] = self.extractor
        data["parser_error"] = self.extraction_error
        data["G"] = self.G
        return data

    @property
    def G(self) -> Dict[str, Any]:
        return {
            "I": self.intent,
            "C_plus": list(self.positive_constraints),
            "C_minus": list(self.negative_constraints),
        }

    @property
    def I(self) -> str:
        return self.intent

    @property
    def C_plus(self) -> List[str]:
        return self.positive_constraints

    @property
    def C_minus(self) -> List[str]:
        return self.negative_constraints

    @property
    def raw_instruction(self) -> str:
        """Compatibility with the previous StructuredGoal field name."""

        return self.raw_query

    @property
    def parser(self) -> str:
        """Compatibility with the previous StructuredGoal field name."""

        return self.extractor

    @property
    def parser_error(self) -> Optional[str]:
        """Compatibility with the previous StructuredGoal field name."""

        return self.extraction_error

    @property
    def positive_keywords(self) -> List[str]:
        """Compatibility with the previous masking interface."""

        return _tokenize_goal_terms([self.intent, self.positive_constraints])

    @property
    def negative_keywords(self) -> List[str]:
        """Compatibility with the previous masking interface."""

        return _tokenize_goal_terms(self.negative_constraints)

    @property
    def product_type(self) -> Optional[str]:
        """Compatibility shim for the previous WebShop-specific parser."""

        return None

    @property
    def attributes(self) -> List[str]:
        """Compatibility shim for the previous WebShop-specific parser."""

        return list(self.positive_constraints)

    @property
    def constraints(self) -> Dict[str, Any]:
        """Compatibility shim for the previous WebShop-specific parser."""

        return {
            "positive": list(self.positive_constraints),
            "negative": list(self.negative_constraints),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        raw_query: Optional[str] = None,
        extractor: Optional[str] = None,
        raw_instruction: Optional[str] = None,
        parser: Optional[str] = None,
    ) -> "GoalContract":
        raw_query = raw_query if raw_query is not None else (raw_instruction or "")
        extractor = extractor or parser or "unknown"
        intent = _first_text(data, "I", "intent", "task_intent", "high_level_intent")
        c_plus = _first_list(
            data,
            "C+",
            "C_plus",
            "c_plus",
            "positive_constraints",
            "required_constraints",
        )
        c_minus = _first_list(
            data,
            "C-",
            "C_minus",
            "c_minus",
            "negative_constraints",
            "forbidden_constraints",
            "forbidden_actions",
        )

        return cls(
            raw_query=raw_query,
            intent=intent or _fallback_intent(raw_query),
            positive_constraints=c_plus,
            negative_constraints=c_minus,
            extractor=extractor,
        )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, Iterable):
        out = []
        for item in value:
            text = _coerce_text(item)
            if text and text not in out:
                out.append(text)
        return out
    text = _coerce_text(value)
    return [text] if text else []


def _first_text(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in data:
            text = _coerce_text(data[key])
            if text:
                return text

    nested = data.get("G")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                text = _coerce_text(nested[key])
                if text:
                    return text

    return ""


def _first_list(data: Dict[str, Any], *keys: str) -> List[str]:
    for key in keys:
        if key in data:
            values = _coerce_list(data[key])
            if values:
                return values

    nested = data.get("G")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                values = _coerce_list(nested[key])
                if values:
                    return values

    return []


def _fallback_intent(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return "unspecified user task"
    return text


def _split_clauses(text: str) -> List[str]:
    return [
        part.strip(" .;,\n\t")
        for part in re.split(
            r"\b(?:and|but|while|with|without|except)\b|[.;]\s*",
            text,
            flags=re.I,
        )
        if part.strip(" .;,\n\t")
    ]


def _extract_negative_constraints(text: str) -> List[str]:
    constraints: List[str] = []
    for match in _NEGATIVE_CONSTRAINT_RE.finditer(text or ""):
        marker = match.group(1).strip()
        body = re.split(r"\b(?:and|but|while)\b", match.group(2), maxsplit=1, flags=re.I)[0]
        constraint = f"{marker} {body}".strip(" .;,\n\t")
        if constraint and constraint not in constraints:
            constraints.append(constraint)
    return constraints


def _tokenize_goal_terms(value: Any) -> List[str]:
    terms: List[str] = []

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            for k, v in item.items():
                add(k)
                add(v)
            return
        if isinstance(item, (list, tuple, set)):
            for x in item:
                add(x)
            return
        for token in re.findall(
            r"[a-zA-Z][a-zA-Z0-9'_-]*|\$?\d+(?:\.\d+)?",
            str(item).lower(),
        ):
            token = token.strip("'_- ")
            if token and token not in terms:
                terms.append(token)

    add(value)
    return terms


class RegexGoalContractExtractor:
    """Small dependency-free fallback for offline runs and tests."""

    extractor_name = "regex_goal_contract"

    def extract(self, query: str) -> GoalContract:
        query = query or ""
        positive_constraints: List[str] = []
        negative_constraints = _extract_negative_constraints(query)
        positive_query = _NEGATIVE_CONSTRAINT_RE.sub(" ", query)

        for clause in _split_clauses(positive_query):
            lowered = clause.lower()
            if any(marker in lowered for marker in _NEGATIVE_MARKERS):
                if clause not in negative_constraints:
                    negative_constraints.append(clause)
            else:
                positive_constraints.append(clause)

        if positive_constraints:
            intent = positive_constraints[0]
            positive_constraints = positive_constraints[1:]
        else:
            intent = _fallback_intent(query)

        return GoalContract(
            raw_query=query,
            intent=intent,
            positive_constraints=positive_constraints,
            negative_constraints=negative_constraints,
            extractor=self.extractor_name,
        )

    def parse(self, instruction: str) -> GoalContract:
        return self.extract(instruction)


class OpenAIGoalContractExtractor:
    """
    OpenAI-backed extractor for Module 1.

    Requires the `openai` package and OPENAI_API_KEY. If either is unavailable
    or the API call fails, this returns the regex fallback with extraction_error
    set so long-running evaluations can continue and log the failure.
    """

    extractor_name = "openai_goal_contract"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 30.0,
        fallback_extractor: Optional[RegexGoalContractExtractor] = None,
        fallback_parser: Optional[RegexGoalContractExtractor] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.last_actual_model: Optional[str] = None
        self.fallback_extractor = (
            fallback_extractor
            or fallback_parser
            or RegexGoalContractExtractor()
        )

    def extract(self, query: str) -> GoalContract:
        query = query or ""
        self.last_actual_model = None
        if not os.environ.get("OPENAI_API_KEY"):
            contract = self.fallback_extractor.extract(query)
            contract.extractor = "regex_fallback"
            contract.extraction_error = "OPENAI_API_KEY is not set"
            return contract

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            contract = self.fallback_extractor.extract(query)
            contract.extractor = "regex_fallback"
            contract.extraction_error = (
                f"openai package unavailable: {type(exc).__name__}: {exc}"
            )
            return contract

        system_prompt = """
You are Module 1 of a goal-grounded defense for WebShop agents.
Extract a structured goal contract G(q) = (I, C+, C-) from the raw user query.

Definitions:
- I: one concise high-level task intent.
- C+: explicit positive constraints the user requires.
- C-: explicit forbidden constraints or actions, including anything phrased as no, not, never, avoid, exclude, without, or do not.

Rules:
- Use only information explicitly present in the user query.
- Do not infer hidden preferences.
- Preserve concrete product attributes, brands, prices, sizes, colors, ratings, materials, and action constraints as short strings.
- Return JSON only with exactly these keys: I, C_plus, C_minus.
""".strip()

        try:
            client = OpenAI(timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            response_model = getattr(response, "model", None)
            self.last_actual_model = str(response_model or self.model)
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            contract = GoalContract.from_dict(
                data,
                raw_query=query,
                extractor=self.extractor_name,
            )
        except Exception as exc:
            contract = self.fallback_extractor.extract(query)
            contract.extractor = "regex_fallback"
            contract.extraction_error = f"{type(exc).__name__}: {exc}"
            return contract

        fallback_contract = self.fallback_extractor.extract(query)
        for forbidden in fallback_contract.negative_constraints:
            if forbidden not in contract.negative_constraints:
                contract.negative_constraints.append(forbidden)
        return contract

    def parse(self, instruction: str) -> GoalContract:
        return self.extract(instruction)


class GoalContractExtraction:
    """Module 1 facade used by GateDefense."""

    module_name = "goal_contract_extraction"

    def __init__(
        self,
        use_openai: bool = True,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 30.0,
        require_success: bool = False,
        cache_path: Optional[str] = None,
    ) -> None:
        self.regex_extractor = RegexGoalContractExtractor()
        self.openai_extractor = OpenAIGoalContractExtractor(
            model=openai_model,
            temperature=temperature,
            timeout=timeout,
            fallback_extractor=self.regex_extractor,
        )
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.require_success = require_success
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.parser_calls = 0
        self.parser_cache_hits = 0
        self.parser_fallback_count = 0
        self.parser_error_count = 0
        self.last_actual_parser_model: Optional[str] = None

        if self.require_success and self.use_openai and not os.environ.get("OPENAI_API_KEY"):
            raise GoalContractParseError(
                "OPENAI_API_KEY is required by --require_goal_parser_success"
            )

    def _read_cache_entry(self, query: str) -> Optional[GoalContract]:
        if self.cache_path is None or not self.cache_path.exists():
            return None

        key = goal_contract_cache_key(query, self.openai_model)
        with _cache_lock(self.cache_path):
            try:
                with self.cache_path.open("r", encoding="utf-8") as handle:
                    cache = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return None

        entry = cache.get("entries", {}).get(key) if isinstance(cache, dict) else None
        if not isinstance(entry, dict):
            return None
        if entry.get("instruction") != query or entry.get("parser_model") != self.openai_model:
            return None

        data = entry.get("contract")
        if not isinstance(data, dict):
            return None
        self.last_actual_parser_model = (
            _coerce_text(entry.get("actual_parser_model")) or self.openai_model
        )
        return GoalContract(
            raw_query=query,
            intent=_coerce_text(data.get("intent", data.get("I"))),
            positive_constraints=_coerce_list(
                data.get("positive_constraints", data.get("C_plus"))
            ),
            negative_constraints=_coerce_list(
                data.get("negative_constraints", data.get("C_minus"))
            ),
            extractor=_coerce_text(data.get("extractor")) or "openai_goal_contract",
            extraction_error=None,
        )

    def _write_cache_entry(self, query: str, contract: GoalContract) -> None:
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        key = goal_contract_cache_key(query, self.openai_model)
        with _cache_lock(self.cache_path):
            try:
                with self.cache_path.open("r", encoding="utf-8") as handle:
                    cache = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                cache = {"version": 1, "entries": {}}

            if not isinstance(cache, dict):
                cache = {"version": 1, "entries": {}}
            entries = cache.setdefault("entries", {})
            if not isinstance(entries, dict):
                entries = {}
                cache["entries"] = entries
            entries[key] = {
                "instruction": query,
                "parser_model": self.openai_model,
                "actual_parser_model": self.last_actual_parser_model or self.openai_model,
                "contract": contract.to_dict(),
            }

            fd, tmp_name = tempfile.mkstemp(
                prefix=self.cache_path.name + ".",
                suffix=".tmp",
                dir=str(self.cache_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(tmp_name, self.cache_path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

    def extract(self, query: str) -> GoalContract:
        query = query or ""
        if not self.use_openai:
            self.last_actual_parser_model = "regex_goal_contract"
            return self.regex_extractor.extract(query)

        if self.require_success and not os.environ.get("OPENAI_API_KEY"):
            self.parser_error_count += 1
            raise GoalContractParseError(
                "OPENAI_API_KEY is required by --require_goal_parser_success"
            )

        cached = self._read_cache_entry(query)
        if cached is not None:
            self.parser_cache_hits += 1
            return cached

        if self.cache_path is not None:
            key = goal_contract_cache_key(query, self.openai_model)
            with _cache_key_lock(self.cache_path, key):
                # A concurrent Slurm row may have populated this exact entry
                # while this process waited for the single-flight lock.
                cached = self._read_cache_entry(query)
                if cached is not None:
                    self.parser_cache_hits += 1
                    return cached
                return self._extract_requested_model(query)

        return self._extract_requested_model(query)

    def _extract_requested_model(self, query: str) -> GoalContract:
        """Call the requested model once; caller handles per-key serialization."""

        self.parser_calls += 1
        contract = self.openai_extractor.extract(query)
        if contract.extraction_error:
            self.parser_error_count += 1
            self.parser_fallback_count += 1
            self.last_actual_parser_model = "regex_fallback"
            if self.require_success:
                raise GoalContractParseError(
                    "Goal parser call failed for requested model "
                    f"{self.openai_model!r}: {contract.extraction_error}"
                )
            return contract

        self.last_actual_parser_model = (
            self.openai_extractor.last_actual_model or self.openai_model
        )
        self._write_cache_entry(query, contract)
        return contract

    def parse(self, instruction: str) -> GoalContract:
        return self.extract(instruction)

    def stats_dict(self) -> Dict[str, Any]:
        return {
            "requested_parser_model": self.openai_model if self.use_openai else None,
            "actual_parser_model": self.last_actual_parser_model,
            "parser_calls": self.parser_calls,
            "parser_cache_hits": self.parser_cache_hits,
            "parser_fallback_count": self.parser_fallback_count,
            "parser_error_count": self.parser_error_count,
            "require_parser_success": self.require_success,
            "goal_contract_cache_path": str(self.cache_path) if self.cache_path else None,
        }


# Backward-compatible names for the previous Gate integration.
StructuredGoal = GoalContract
RegexGoalParser = RegexGoalContractExtractor
OpenAIGoalParser = OpenAIGoalContractExtractor
