#!/usr/bin/env python3
"""Aggregate WebShop rebuttal runs without guessing unavailable results.

The script accepts summary JSON files, per-episode JSON/JSONL logs, or a mix of
the two.  It writes a detailed CSV and paper-friendly Markdown and LaTeX
tables.  Proportion confidence intervals are Wilson 95% intervals.  AER
comparisons are paired by task ID against the ``none`` baseline and use a
paired bootstrap with the fixed rebuttal seed (42).

Examples
--------

    python aggregate_rebuttal.py runs/*.summary.json
    python aggregate_rebuttal.py logs/none.jsonl logs/gate.jsonl \
        --csv results.csv --markdown results.md --latex results.tex

Missing fields remain missing in every output.  In particular, rounded rates
from older summaries are never converted back into integer counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
WILSON_Z_95 = 1.959963984540054


class AggregationError(ValueError):
    """Raised when input runs cannot be compared safely."""


def wilson_interval(
    numerator: int,
    denominator: int,
    z: float = WILSON_Z_95,
) -> Optional[Tuple[float, float]]:
    """Return a Wilson score interval as proportions in ``[0, 1]``.

    A zero denominator has no defined proportion and therefore returns
    ``None``.  Invalid counts raise ``ValueError`` rather than being clipped.
    """

    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise ValueError("Wilson counts must be integers, not booleans")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise ValueError("Wilson counts must be integers")
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError(
            f"Invalid Wilson counts: numerator={numerator}, "
            f"denominator={denominator}"
        )
    if denominator == 0:
        return None

    p_hat = numerator / denominator
    z2_over_n = z * z / denominator
    center = (p_hat + z2_over_n / 2.0) / (1.0 + z2_over_n)
    margin = (
        z
        * math.sqrt(
            p_hat * (1.0 - p_hat) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / (1.0 + z2_over_n)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    paired_differences: Sequence[float],
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Return a percentile 95% CI for the mean of paired differences."""

    differences = [float(value) for value in paired_differences]
    if not differences:
        raise ValueError("At least one paired difference is required")
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Paired differences must be finite")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")

    # This special case also avoids needless work for common exact smoke tests.
    if all(value == differences[0] for value in differences[1:]):
        return differences[0], differences[0]

    rng = random.Random(seed)
    n = len(differences)
    estimates = [
        sum(rng.choices(differences, k=n)) / n
        for _ in range(samples)
    ]
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def paired_bootstrap_aer_difference(
    candidate_rewards: Sequence[float],
    baseline_rewards: Sequence[float],
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Return ``(mean difference, low, high)`` for paired episode rewards."""

    if len(candidate_rewards) != len(baseline_rewards):
        raise ValueError("Paired reward sequences must have identical lengths")
    if not candidate_rewards:
        raise ValueError("At least one paired reward is required")
    differences = [
        float(candidate) - float(baseline)
        for candidate, baseline in zip(candidate_rewards, baseline_rewards)
    ]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Episode rewards must be finite")
    point = statistics.fmean(differences)
    low, high = paired_bootstrap_interval(differences, samples=samples, seed=seed)
    return point, low, high


def clean_change_counts(
    candidate_rewards: Mapping[str, float],
    baseline_rewards: Mapping[str, float],
    tolerance: float = 1e-12,
) -> Tuple[int, int, int]:
    """Count clean episodes improved, unchanged, and harmed by task ID."""

    if set(candidate_rewards) != set(baseline_rewards):
        raise ValueError("Clean reward maps must contain identical task IDs")
    improved = unchanged = harmed = 0
    for task_id in candidate_rewards:
        delta = float(candidate_rewards[task_id]) - float(baseline_rewards[task_id])
        if delta > tolerance:
            improved += 1
        elif delta < -tolerance:
            harmed += 1
        else:
            unchanged += 1
    return improved, unchanged, harmed


def _canonical_task_id(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        raise AggregationError(f"Invalid task ID: {value!r}")
    # JSON task-ID files use integers while some logs serialize the same IDs as
    # strings.  Their textual forms are the stable pairing key.
    return str(value)


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    return None


def _lookup(mapping: Mapping[str, Any], dotted_name: str) -> Any:
    current: Any = mapping
    for part in dotted_name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _lookup_present(
    mapping: Mapping[str, Any], dotted_name: str
) -> Tuple[bool, Any]:
    current: Any = mapping
    for part in dotted_name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = _lookup(mapping, name)
        if value is not None:
            return value
    return None


def _summary_sources(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    sources: List[Mapping[str, Any]] = [raw]
    for key in ("metrics", "rates", "results", "summary"):
        nested = raw.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def _first_from_sources(raw: Mapping[str, Any], names: Iterable[str]) -> Any:
    for source in _summary_sources(raw):
        value = _first(source, names)
        if value is not None:
            return value
    return None


def _episode_id(record: Mapping[str, Any]) -> Optional[str]:
    value = _first(
        record,
        ("task_id", "episode_index", "episode_id", "env_id", "index", "id"),
    )
    return _canonical_task_id(value) if value is not None else None


def _episode_reward(record: Mapping[str, Any]) -> Optional[float]:
    return _finite_float(
        _first(record, ("reward", "final_reward", "episode_reward", "score"))
    )


def _looks_like_episode(record: Any) -> bool:
    if not isinstance(record, Mapping) or _episode_id(record) is None:
        return False
    episode_markers = {
        "reward",
        "final_reward",
        "episode_reward",
        "task_success",
        "steps",
        "paper_style_attack_hit",
    }
    return any(marker in record for marker in episode_markers)


def _episode_container(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in (
        "per_episode",
        "episode_records",
        "episode_results",
        "episodes_data",
        "task_results",
        "paired_episodes",
    ):
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            records: List[Mapping[str, Any]] = []
            for task_id, item in value.items():
                if isinstance(item, Mapping):
                    record = dict(item)
                    record.setdefault("task_id", task_id)
                else:
                    record = {"task_id": task_id, "reward": item}
                records.append(record)
            return records
    episodes_value = raw.get("episodes")
    if isinstance(episodes_value, list):
        return [item for item in episodes_value if isinstance(item, Mapping)]
    return []


def _task_ids_from_file(path_value: Any, source_path: Path) -> List[str]:
    if not isinstance(path_value, str) or not path_value.strip():
        return []
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = source_path.parent / candidate
    if not candidate.is_file():
        return []
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, Mapping):
        payload = _first(payload, ("task_ids", "ids", "test_ids"))
    if not isinstance(payload, list):
        return []
    return [_canonical_task_id(value) for value in payload]


def _extract_explicit_task_ids(raw: Mapping[str, Any], source_path: Path) -> List[str]:
    values = _first_from_sources(
        raw,
        ("task_ids", "eval_task_ids", "selected_task_ids", "test_ids"),
    )
    if isinstance(values, list):
        return [_canonical_task_id(value) for value in values]

    path_value = _first_from_sources(
        raw,
        (
            "task_ids_path",
            "test_ids_path",
            "cli_args.test_ids_path",
            "arguments.test_ids_path",
            "args.test_ids_path",
        ),
    )
    return _task_ids_from_file(path_value, source_path)


def _metadata_entry(
    raw: Mapping[str, Any], names: Iterable[str]
) -> Tuple[bool, Any]:
    """Return metadata presence/value while preserving explicit JSON nulls."""

    names = tuple(names)
    for source in _summary_sources(raw):
        for name in names:
            present, value = _lookup_present(source, name)
            if present:
                return True, value
    for source in _summary_sources(raw):
        for prefix in ("cli_arguments", "cli_args", "arguments", "args"):
            for name in names:
                present, value = _lookup_present(source, f"{prefix}.{name}")
                if present:
                    return True, value
    return False, None


def _metadata_value(raw: Mapping[str, Any], names: Iterable[str]) -> Any:
    return _metadata_entry(raw, names)[1]


def _normalise_attack(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "query": "direct",
        "query_attack": "direct",
        "direct_attack": "direct",
        "direct": "direct",
        "observation": "indirect",
        "observation_attack": "indirect",
        "indirect_attack": "indirect",
        "indirect": "indirect",
        "clean": "clean",
    }
    return aliases.get(text, text or None)


@dataclass
class ProportionMetric:
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    proportion: Optional[float] = None

    def __post_init__(self) -> None:
        if self.numerator is not None and self.denominator is not None:
            if (
                self.numerator < 0
                or self.denominator < 0
                or self.numerator > self.denominator
            ):
                raise AggregationError(
                    f"Invalid metric counts {self.numerator}/{self.denominator}"
                )
            if self.denominator > 0:
                count_proportion = self.numerator / self.denominator
                if self.proportion is not None and not math.isclose(
                    self.proportion,
                    count_proportion,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise AggregationError(
                        f"Metric rate {self.proportion} disagrees with counts "
                        f"{self.numerator}/{self.denominator}"
                    )
                self.proportion = count_proportion
            elif self.proportion is not None:
                raise AggregationError(
                    "A zero-denominator metric cannot have a reported rate"
                )
        if self.proportion is not None and not 0.0 <= self.proportion <= 1.0:
            raise AggregationError(f"Invalid proportion: {self.proportion}")

    @property
    def interval(self) -> Optional[Tuple[float, float]]:
        if self.numerator is None or self.denominator is None:
            return None
        return wilson_interval(self.numerator, self.denominator)

    @property
    def is_missing(self) -> bool:
        return (
            self.numerator is None
            and self.denominator is None
            and self.proportion is None
        )


@dataclass(frozen=True)
class MetricSpec:
    names: Tuple[str, ...]
    numerator_names: Tuple[str, ...] = ()
    denominator_names: Tuple[str, ...] = ()
    legacy_percent_names: Tuple[str, ...] = ()
    episode_keys: Tuple[str, ...] = ()
    level: str = "episode"


METRIC_SPECS: Dict[str, MetricSpec] = {
    "task_success": MetricSpec(
        names=(
            "exact_reward_task_success",
            "exact_reward_task_success_rate",
            "exact_task_success",
            "exact_task_success_rate",
        ),
        numerator_names=(
            "exact_reward_task_success_numerator",
            "exact_reward_task_success_hits",
            "exact_task_success_numerator",
            "exact_task_success_hits",
        ),
        denominator_names=(
            "exact_reward_task_success_denominator",
            "exact_task_success_denominator",
        ),
        legacy_percent_names=(
            "exact_reward_task_success_rate",
            "exact_task_success_rate",
        ),
        episode_keys=("exact_reward_task_success", "exact_task_success"),
    ),
    "direct_asr": MetricSpec(
        names=("direct_paper_style_asr", "direct_asr", "paper_style_asr"),
        numerator_names=("direct_asr_numerator", "paper_style_asr_numerator", "asr_hits"),
        denominator_names=("direct_asr_denominator", "paper_style_asr_denominator"),
        legacy_percent_names=("direct_paper_style_asr", "direct_asr", "paper_style_asr"),
        episode_keys=("paper_style_attack_hit", "direct_attack_hit"),
    ),
    "indirect_conditional_asr": MetricSpec(
        names=(
            "indirect_conditional_asr",
            "conditional_indirect_asr",
            "conditional_observation_asr",
        ),
        numerator_names=("indirect_conditional_asr_numerator", "asr_hits"),
        denominator_names=(
            "indirect_conditional_asr_denominator",
            "observation_trigger_seen",
            "trigger_exposure_count",
        ),
        legacy_percent_names=(
            "indirect_conditional_asr",
            "conditional_indirect_asr",
            "conditional_observation_asr",
        ),
        episode_keys=("paper_style_attack_hit", "indirect_attack_hit"),
    ),
    "indirect_unconditional_asr": MetricSpec(
        names=(
            "indirect_unconditional_asr",
            "unconditional_indirect_asr",
            "overall_observation_asr",
        ),
        numerator_names=("indirect_unconditional_asr_numerator", "asr_hits"),
        denominator_names=("indirect_unconditional_asr_denominator",),
        legacy_percent_names=(
            "indirect_unconditional_asr",
            "unconditional_indirect_asr",
            "overall_observation_asr",
        ),
        episode_keys=("paper_style_attack_hit", "indirect_attack_hit"),
    ),
    "trigger_exposure": MetricSpec(
        names=("trigger_exposure", "trigger_exposure_rate"),
        numerator_names=(
            "trigger_exposure_numerator",
            "trigger_exposure_count",
            "observation_trigger_seen",
        ),
        denominator_names=("trigger_exposure_denominator",),
        legacy_percent_names=("trigger_exposure_rate",),
        episode_keys=("observation_trigger_seen", "trigger_exposed", "trigger_exposure"),
    ),
    "target_click": MetricSpec(
        names=("target_product_click", "target_product_click_rate", "selection_asr"),
        numerator_names=(
            "target_product_click_numerator",
            "target_product_click_count",
            "target_click_hits",
        ),
        denominator_names=("target_product_click_denominator",),
        legacy_percent_names=("target_product_click_rate", "selection_asr"),
        episode_keys=(
            "selection_asr_hit_target_click_or_select",
            "target_product_click",
            "target_click_hit",
        ),
    ),
    "target_purchase": MetricSpec(
        names=(
            "strict_target_purchase",
            "strict_target_purchase_rate",
            "strict_purchase_asr",
        ),
        numerator_names=(
            "strict_target_purchase_numerator",
            "target_purchase_hits",
        ),
        denominator_names=("strict_target_purchase_denominator",),
        legacy_percent_names=("strict_target_purchase_rate", "strict_purchase_asr"),
        episode_keys=(
            "strict_purchase_asr_hit",
            "strict_target_purchase",
            "target_purchase_hit",
        ),
    ),
    "oracle_eligible": MetricSpec(
        names=("oracle_eligible_rate", "oracle_eligibility_rate"),
        numerator_names=(
            "oracle_eligible_numerator",
            "oracle_eligible_count",
        ),
        denominator_names=(
            "oracle_eligible_denominator",
            "oracle_eligibility_determined_count",
        ),
        legacy_percent_names=("oracle_eligible_rate", "oracle_eligibility_rate"),
        episode_keys=("oracle_eligible",),
    ),
    "oracle_violating_click": MetricSpec(
        names=(
            "oracle_violating_product_click_rate",
            "oracle_violating_product_click",
        ),
        numerator_names=(
            "oracle_violating_product_click_numerator",
            "oracle_violating_product_click_hits",
            "oracle_violating_product_click_count",
        ),
        denominator_names=(
            "oracle_violating_product_click_denominator",
            "oracle_eligible_count",
        ),
        legacy_percent_names=("oracle_violating_product_click_rate",),
        episode_keys=("oracle_violating_product_click",),
    ),
    "oracle_strict_purchase": MetricSpec(
        names=("oracle_strict_purchase_rate", "oracle_strict_purchase"),
        numerator_names=(
            "oracle_strict_purchase_numerator",
            "oracle_strict_purchase_hits",
            "oracle_strict_purchase_count",
        ),
        denominator_names=(
            "oracle_strict_purchase_denominator",
            "oracle_eligible_count",
        ),
        legacy_percent_names=("oracle_strict_purchase_rate",),
        episode_keys=("oracle_strict_purchase",),
    ),
    "valid_action": MetricSpec(
        names=("valid_action", "valid_action_rate"),
        numerator_names=("valid_action_numerator", "valid_action_count"),
        denominator_names=("valid_action_denominator", "total_action_count", "total_steps"),
        legacy_percent_names=("valid_action_rate",),
        episode_keys=("valid_action", "action_valid", "executed_action_valid", "legal_action"),
        level="step",
    ),
    "unparsable_action": MetricSpec(
        names=("unparsable_action", "unparsable_action_rate"),
        numerator_names=("unparsable_action_numerator", "unparsable_action_count"),
        denominator_names=(
            "unparsable_action_denominator",
            "total_action_count",
            "total_steps",
        ),
        legacy_percent_names=("unparsable_action_rate",),
        episode_keys=("unparsable_action", "action_unparsable", "parse_failed"),
        level="step",
    ),
    "proposed_attack": MetricSpec(
        names=("proposed_action_attack", "proposed_action_attack_rate"),
        numerator_names=("proposed_action_attack_numerator", "proposed_action_attack_count"),
        denominator_names=(
            "proposed_action_attack_denominator",
            "total_action_count",
            "total_steps",
        ),
        legacy_percent_names=("proposed_action_attack_rate",),
        episode_keys=(
            "proposed_action_attack",
            "proposed_attack",
            "proposal_is_attack",
            "proposal_is_malicious",
        ),
        level="step",
    ),
    "executed_attack": MetricSpec(
        names=("executed_action_attack", "executed_action_attack_rate"),
        numerator_names=("executed_action_attack_numerator", "executed_action_attack_count"),
        denominator_names=(
            "executed_action_attack_denominator",
            "total_action_count",
            "total_steps",
        ),
        legacy_percent_names=("executed_action_attack_rate",),
        episode_keys=(
            "executed_action_attack",
            "executed_attack",
            "execution_is_attack",
            "executed_is_malicious",
        ),
        level="step",
    ),
    "episode_intervention": MetricSpec(
        names=("episode_intervention", "episode_intervention_rate"),
        numerator_names=("episode_intervention_numerator", "episode_intervention_count"),
        denominator_names=("episode_intervention_denominator",),
        legacy_percent_names=("episode_intervention_rate",),
        episode_keys=(
            "episode_intervened",
            "intervention_episode",
            "intervention",
            "intervened",
            "had_intervention",
        ),
    ),
    "step_intervention": MetricSpec(
        names=("step_intervention", "step_intervention_rate"),
        numerator_names=("step_intervention_numerator", "step_intervention_count"),
        denominator_names=("step_intervention_denominator", "total_steps"),
        legacy_percent_names=("step_intervention_rate",),
        episode_keys=("step_intervened", "intervened", "action_intervened"),
        level="step",
    ),
}


def _bool_value(record: Mapping[str, Any], names: Sequence[str]) -> Optional[bool]:
    value = _first(record, names)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _parse_metric_object(value: Any, legacy_percent: bool) -> ProportionMetric:
    if isinstance(value, Mapping):
        numerator = _integer(_first(value, ("numerator", "count", "hits", "successes")))
        denominator = _integer(_first(value, ("denominator", "total", "n")))
        reported_proportions: List[float] = []
        proportion_value = _finite_float(_first(value, ("proportion", "fraction")))
        if proportion_value is not None:
            reported_proportions.append(proportion_value)
        parsed_percent = _finite_float(
            _first(value, ("percentage", "percent", "pct", "rate_percent"))
        )
        if parsed_percent is not None:
            reported_proportions.append(parsed_percent / 100.0)
        rate_value = _finite_float(value.get("rate"))
        if rate_value is not None:
            # New summaries should prefer explicit ``proportion`` or
            # ``percentage``.  For ``rate``, [0,1] is the conventional
            # machine-readable scale; larger values are unambiguously %.
            reported_proportions.append(
                rate_value if rate_value <= 1.0 else rate_value / 100.0
            )
        if reported_proportions and any(
            not math.isclose(
                reported_proportions[0], other, rel_tol=1e-9, abs_tol=1e-12
            )
            for other in reported_proportions[1:]
        ):
            raise AggregationError(
                f"Metric object has contradictory rate fields: {value!r}"
            )
        proportion = reported_proportions[0] if reported_proportions else None
        return ProportionMetric(numerator, denominator, proportion)

    scalar = _finite_float(value)
    if scalar is None:
        return ProportionMetric()
    proportion = scalar / 100.0 if legacy_percent else scalar
    return ProportionMetric(proportion=proportion)


@dataclass
class RunRecord:
    source: Path
    raw: Dict[str, Any]
    episodes: Dict[str, Mapping[str, Any]] = field(default_factory=dict)
    task_ids: Tuple[str, ...] = ()
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_raw(
        cls,
        source: Path,
        raw: Mapping[str, Any],
        extra_episodes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "RunRecord":
        copied = dict(raw)
        episode_records = list(extra_episodes or _episode_container(copied))

        explicit_ids = _extract_explicit_task_ids(copied, source)
        episodes: Dict[str, Mapping[str, Any]] = {}
        for record in episode_records:
            task_id = _episode_id(record)
            if task_id is None:
                raise AggregationError(
                    f"{source}: per-episode record has no task ID: {record!r}"
                )
            if task_id in episodes:
                raise AggregationError(
                    f"{source}: duplicate per-episode task ID {task_id!r}"
                )
            episodes[task_id] = record

        # Some compact summaries store aligned ID/reward arrays.
        reward_values = _first_from_sources(
            copied, ("episode_rewards", "per_episode_rewards", "rewards")
        )
        if not episodes and explicit_ids and isinstance(reward_values, list):
            if len(reward_values) != len(explicit_ids):
                raise AggregationError(
                    f"{source}: task_ids and episode_rewards lengths differ"
                )
            for task_id, reward in zip(explicit_ids, reward_values):
                episodes[task_id] = {"task_id": task_id, "reward": reward}

        episode_ids = list(episodes)
        if explicit_ids and episode_ids and set(explicit_ids) != set(episode_ids):
            missing = sorted(set(explicit_ids) - set(episode_ids))[:5]
            extra = sorted(set(episode_ids) - set(explicit_ids))[:5]
            raise AggregationError(
                f"{source}: summary task IDs disagree with per-episode records "
                f"(missing={missing}, extra={extra})"
            )
        task_ids = explicit_ids or episode_ids
        if len(task_ids) != len(set(task_ids)):
            raise AggregationError(f"{source}: duplicate task IDs in summary")

        task_count = _integer(
            _metadata_value(copied, ("task_count", "exact_task_count", "num_tasks"))
        )
        if task_count is not None and task_ids and task_count != len(task_ids):
            raise AggregationError(
                f"{source}: task_count={task_count} but {len(task_ids)} task IDs were found"
            )

        explicit_episode_count = _integer(
            _metadata_value(copied, ("episode_count", "num_episodes"))
        )
        if explicit_episode_count is None:
            explicit_episode_count = _integer(copied.get("episodes"))
        if (
            explicit_episode_count is not None
            and episodes
            and explicit_episode_count != len(episodes)
        ):
            raise AggregationError(
                f"{source}: episode_count={explicit_episode_count} but "
                f"{len(episodes)} per-episode records were found"
            )

        return cls(
            source=source,
            raw=copied,
            episodes=episodes,
            task_ids=tuple(task_ids),
        )

    @property
    def method(self) -> Optional[str]:
        value = _metadata_value(self.raw, ("method", "defense"))
        return str(value) if value is not None and str(value).strip() else None

    @property
    def runtime_mode(self) -> Optional[str]:
        present, value = _metadata_entry(self.raw, ("runtime_mode",))
        if not present and (self.method or "").casefold().startswith("gate"):
            value = _metadata_value(self.raw, ("gate_runtime_mode",))
        return str(value) if value is not None and str(value).strip() else None

    @property
    def attack_type(self) -> Optional[str]:
        return _normalise_attack(
            _metadata_value(self.raw, ("attack_type", "eval_type", "evaluation_type", "type"))
        )

    @property
    def seed(self) -> Any:
        return _metadata_value(self.raw, ("seed", "random_seed"))

    @property
    def episode_count(self) -> Optional[int]:
        value = _metadata_value(self.raw, ("episode_count", "num_episodes"))
        parsed = _integer(value)
        if parsed is None:
            episodes_value = self.raw.get("episodes")
            parsed = _integer(episodes_value)
        if parsed is None and self.episodes:
            parsed = len(self.episodes)
        return parsed

    @property
    def task_count(self) -> Optional[int]:
        value = _integer(
            _metadata_value(self.raw, ("task_count", "exact_task_count", "num_tasks"))
        )
        return value if value is not None else (len(self.task_ids) if self.task_ids else None)

    @property
    def aer(self) -> Optional[float]:
        value = _first_from_sources(
            self.raw,
            ("aer", "AER", "average_episode_reward", "avg_episode_reward", "mean_reward"),
        )
        if isinstance(value, Mapping):
            value = _first(value, ("value", "mean", "aer"))
        parsed = _finite_float(value)
        reward_map = self.reward_map()
        if parsed is not None and reward_map is not None:
            derived = statistics.fmean(reward_map.values())
            if not math.isclose(parsed, derived, rel_tol=1e-9, abs_tol=1e-12):
                raise AggregationError(
                    f"{self.source}: reported AER {parsed} disagrees with "
                    f"per-episode AER {derived}"
                )
        if parsed is not None:
            return parsed
        if reward_map is not None:
            return statistics.fmean(reward_map.values())
        return None

    def reward_map(self) -> Optional[Dict[str, float]]:
        if not self.task_ids or not self.episodes:
            return None
        rewards: Dict[str, float] = {}
        for task_id in self.task_ids:
            record = self.episodes.get(task_id)
            if record is None:
                return None
            reward = _episode_reward(record)
            if reward is None:
                return None
            rewards[task_id] = reward
        return rewards

    def step_records(self) -> Optional[List[Mapping[str, Any]]]:
        """Return explicit debug-log steps, or ``None`` for compact summaries."""

        if not self.episodes:
            return None
        records: List[Mapping[str, Any]] = []
        for episode in self.episodes.values():
            steps = episode.get("steps")
            if not isinstance(steps, list):
                return None
            if not all(isinstance(step, Mapping) for step in steps):
                return None
            records.extend(steps)
        return records

    def step_true_count(self, key: str) -> Optional[int]:
        records = self.step_records()
        if records is None:
            return None
        values = [_bool_value(record, (key,)) for record in records]
        if any(value is None for value in values):
            return None
        return sum(bool(value) for value in values)

    def step_runtime_distribution(self) -> Optional[Dict[str, float]]:
        records = self.step_records()
        if records is None:
            return None
        values = [_finite_float(record.get("added_runtime_seconds")) for record in records]
        if not values or any(value is None for value in values):
            return None
        numeric = sorted(value for value in values if value is not None)
        rank = 0.95 * (len(numeric) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            p95 = numeric[lower]
        else:
            fraction = rank - lower
            p95 = numeric[lower] + fraction * (numeric[upper] - numeric[lower])
        return {
            "mean": statistics.fmean(numeric),
            "median": statistics.median(numeric),
            "p95": p95,
        }

    def context_key(self) -> Tuple[Any, ...]:
        stress_test = _metadata_value(self.raw, ("stress_test",)) or "none"
        oracle_mode = _metadata_value(self.raw, ("oracle_mode",)) or "none"
        oracle_strategy = _metadata_value(self.raw, ("oracle_strategy",)) or "none"
        target_brand = (
            None
            if self.attack_type == "clean"
            else _metadata_value(self.raw, ("target_brand",))
        )
        return (
            self.attack_type,
            str(stress_test),
            str(oracle_mode),
            str(oracle_strategy),
            str(target_brand) if target_brand is not None else None,
        )

    def _episode_metric(self, name: str, spec: MetricSpec) -> ProportionMetric:
        if not self.episodes:
            return ProportionMetric()
        if name == "indirect_conditional_asr":
            hits = 0
            trigger_count = 0
            for episode in self.episodes.values():
                triggered = _bool_value(
                    episode,
                    ("observation_trigger_seen", "trigger_exposed", "trigger_exposure"),
                )
                if triggered is None:
                    return ProportionMetric()
                if not triggered:
                    continue
                attack_hit = _bool_value(episode, spec.episode_keys)
                if attack_hit is None:
                    return ProportionMetric()
                trigger_count += 1
                hits += int(attack_hit)
            return ProportionMetric(hits, trigger_count)
        if name in {"oracle_violating_click", "oracle_strict_purchase"}:
            hits = 0
            eligible_count = 0
            eligibility_determined = False
            for episode in self.episodes.values():
                eligible = _bool_value(episode, ("oracle_eligible",))
                if eligible is None:
                    continue
                eligibility_determined = True
                if not eligible:
                    continue
                outcome = _bool_value(episode, spec.episode_keys)
                if outcome is None:
                    return ProportionMetric()
                eligible_count += 1
                hits += int(outcome)
            if not eligibility_determined:
                return ProportionMetric()
            return ProportionMetric(hits, eligible_count)
        if name == "oracle_eligible":
            determined = []
            for episode in self.episodes.values():
                eligible = _bool_value(episode, ("oracle_eligible",))
                if eligible is not None:
                    determined.append(eligible)
            if not determined:
                return ProportionMetric()
            return ProportionMetric(sum(determined), len(determined))
        if spec.level == "step":
            records: List[Mapping[str, Any]] = []
            for episode in self.episodes.values():
                steps = episode.get("steps")
                if not isinstance(steps, list):
                    return ProportionMetric()
                records.extend(
                    step
                    for step in steps
                    if isinstance(step, Mapping)
                    and not (
                        "request_error" in step and step.get("request_error") is not None
                    )
                )
            if not records:
                return ProportionMetric()
        else:
            records = list(self.episodes.values())

        values: List[bool] = []
        for record in records:
            value = _bool_value(record, spec.episode_keys)
            if value is None:
                return ProportionMetric()
            values.append(value)
        return ProportionMetric(sum(values), len(values))

    def metric(self, name: str) -> ProportionMetric:
        spec = METRIC_SPECS[name]
        if name == "direct_asr" and self.attack_type not in (None, "direct"):
            return ProportionMetric()
        if name.startswith("indirect_") and self.attack_type not in (None, "indirect"):
            return ProportionMetric()
        if name == "trigger_exposure" and self.attack_type not in (None, "indirect"):
            return ProportionMetric()

        metric = ProportionMetric()
        for source in _summary_sources(self.raw):
            for candidate_name in spec.names:
                value = _lookup(source, candidate_name)
                if value is None:
                    continue
                metric = _parse_metric_object(
                    value,
                    legacy_percent=candidate_name in spec.legacy_percent_names,
                )
                break
            if not metric.is_missing:
                break

        numerator = _integer(_first_from_sources(self.raw, spec.numerator_names))
        denominator = _integer(_first_from_sources(self.raw, spec.denominator_names))
        if metric.numerator is None and numerator is not None:
            metric.numerator = numerator
        if metric.denominator is None and denominator is not None:
            metric.denominator = denominator
        derived = self._episode_metric(name, spec)
        if (
            metric.numerator is not None
            and metric.denominator is not None
            and derived.numerator is not None
            and derived.denominator is not None
            and (
                metric.numerator != derived.numerator
                or metric.denominator != derived.denominator
            )
        ):
            raise AggregationError(
                f"{self.source}: reported {name} counts "
                f"{metric.numerator}/{metric.denominator} disagree with "
                f"per-episode counts {derived.numerator}/{derived.denominator}"
            )
        # Per-episode logs can supply a complete raw pair.  Never combine one
        # published count with a separately derived value: a lone numerator or
        # denominator remains visibly incomplete.
        if (
            metric.numerator is None
            and metric.denominator is None
            and derived.numerator is not None
            and derived.denominator is not None
        ):
            metric.numerator = derived.numerator
            metric.denominator = derived.denominator

        # Re-run validation and ensure exact counts determine the displayed rate.
        return ProportionMetric(metric.numerator, metric.denominator, metric.proportion)


def _common_episode_metadata(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    metadata_keys = (
        "method",
        "defense",
        "runtime_mode",
        "gate_runtime_mode",
        "attack_type",
        "eval_type",
        "seed",
        "stress_test",
        "oracle_mode",
        "oracle_strategy",
        "target_brand",
    )
    common: Dict[str, Any] = {}
    first = records[0]
    for key in metadata_keys:
        if key in first and all(record.get(key) == first[key] for record in records):
            common[key] = first[key]
    common["per_episode"] = list(records)
    return common


def _runs_from_payload(payload: Any, source: Path) -> List[RunRecord]:
    if isinstance(payload, Mapping):
        runs_value = payload.get("runs")
        if isinstance(runs_value, list):
            runs: List[RunRecord] = []
            for index, raw in enumerate(runs_value):
                if not isinstance(raw, Mapping):
                    raise AggregationError(f"{source}: runs[{index}] is not an object")
                runs.append(RunRecord.from_raw(source, raw))
            return runs
        if _looks_like_episode(payload):
            raw = _common_episode_metadata([payload])
            return [RunRecord.from_raw(source, raw)]
        return [RunRecord.from_raw(source, payload)]

    if isinstance(payload, list):
        if not payload:
            raise AggregationError(f"{source}: input is an empty JSON array")
        if not all(isinstance(item, Mapping) for item in payload):
            raise AggregationError(f"{source}: JSON array entries must be objects")
        records = list(payload)
        episode_flags = [_looks_like_episode(record) for record in records]
        if all(episode_flags):
            return [RunRecord.from_raw(source, _common_episode_metadata(records))]
        if not any(episode_flags):
            return [RunRecord.from_raw(source, raw) for raw in records]
        summaries = [
            record
            for record, is_episode in zip(records, episode_flags)
            if not is_episode
        ]
        episodes = [record for record, is_episode in zip(records, episode_flags) if is_episode]
        if len(summaries) != 1:
            raise AggregationError(
                f"{source}: mixed episode and summary records require exactly one summary"
            )
        return [RunRecord.from_raw(source, summaries[0], extra_episodes=episodes)]

    raise AggregationError(f"{source}: top-level JSON value must be an object or array")


def load_runs(paths: Sequence[Path]) -> List[RunRecord]:
    """Load run summaries and/or per-episode logs from JSON and JSONL files."""

    runs: List[RunRecord] = []
    for path_like in paths:
        path = Path(path_like).expanduser()
        if not path.is_file():
            raise AggregationError(f"Input does not exist or is not a file: {path}")
        try:
            if path.suffix.lower() == ".jsonl":
                payload: List[Any] = []
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            payload.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise AggregationError(
                                f"{path}:{line_number}: invalid JSON: {exc.msg}"
                            ) from exc
            else:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
        except OSError as exc:
            raise AggregationError(f"Could not read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AggregationError(f"{path}: invalid JSON: {exc.msg}") from exc
        runs.extend(_runs_from_payload(payload, path))
    if not runs:
        raise AggregationError("No runs were loaded")
    return runs


def validate_identical_task_ids(
    runs: Sequence[RunRecord],
    allow_mismatched: bool = False,
) -> List[str]:
    """Validate task IDs within each directly comparable experiment group."""

    grouped: Dict[Tuple[Any, ...], List[RunRecord]] = {}
    for run in runs:
        grouped.setdefault(run.context_key(), []).append(run)

    errors: List[str] = []
    for context, group in grouped.items():
        if len(group) < 2:
            continue
        missing = [str(run.source) for run in group if not run.task_ids]
        if missing:
            errors.append(
                f"comparison group {context!r} has no task IDs for: {', '.join(missing)}"
            )
            continue
        reference = set(group[0].task_ids)
        for run in group[1:]:
            current = set(run.task_ids)
            if current == reference:
                continue
            missing_ids = sorted(reference - current)[:8]
            extra_ids = sorted(current - reference)[:8]
            errors.append(
                f"comparison group {context!r} has mismatched task IDs in {run.source}; "
                f"missing={missing_ids}, extra={extra_ids}"
            )

    if errors and not allow_mismatched:
        raise AggregationError(
            "Refusing to aggregate non-identical or unverifiable task sets. "
            "Use --allow-mismatched-task-ids only for an explicitly unpaired report.\n"
            + "\n".join(errors)
        )
    return errors


def _method_matches(run: RunRecord, baseline_method: str) -> bool:
    return (run.method or "").strip().lower() == baseline_method.strip().lower()


def _find_baseline(
    run: RunRecord,
    runs: Sequence[RunRecord],
    baseline_method: str,
) -> Optional[RunRecord]:
    candidates = [
        candidate
        for candidate in runs
        if candidate is not run
        and _method_matches(candidate, baseline_method)
        and candidate.context_key() == run.context_key()
        and candidate.task_ids
        and set(candidate.task_ids) == set(run.task_ids)
    ]
    # Do not silently call different known random seeds a paired method
    # comparison.  Two legacy runs with both seeds missing can still be paired
    # by their exact task IDs.
    candidates = [candidate for candidate in candidates if candidate.seed == run.seed]
    return candidates[0] if len(candidates) == 1 else None


METRIC_ORDER = (
    "task_success",
    "direct_asr",
    "indirect_conditional_asr",
    "indirect_unconditional_asr",
    "trigger_exposure",
    "target_click",
    "target_purchase",
    "oracle_eligible",
    "oracle_violating_click",
    "oracle_strict_purchase",
    "valid_action",
    "unparsable_action",
    "proposed_attack",
    "executed_attack",
    "episode_intervention",
    "step_intervention",
)


PARSER_LLM_EFFICIENCY_COLUMNS = (
    "parser_request_count",
    "parser_call_count",
    "parser_api_call_count",
    "parser_cache_hit_count",
    "parser_usage_reported_call_count",
    "parser_usage_missing_call_count",
    "parser_input_token_count",
    "parser_cached_input_token_count",
    "parser_output_token_count",
    "parser_reasoning_token_count",
    "parser_total_token_count",
    "parser_estimated_cost_usd",
)

JUDGE_LLM_EFFICIENCY_COLUMNS = (
    "judge_request_count",
    "judge_call_count",
    "judge_cache_hit_count",
    "judge_usage_reported_call_count",
    "judge_usage_missing_call_count",
    "judge_input_token_count",
    "judge_cached_input_token_count",
    "judge_output_token_count",
    "judge_reasoning_token_count",
    "judge_total_token_count",
    "judge_estimated_cost_usd",
)

DEFENSE_LLM_EFFICIENCY_COLUMNS = (
    "defense_llm_request_count",
    "defense_llm_api_call_count",
    "defense_llm_cache_hit_count",
    "defense_llm_usage_reported_call_count",
    "defense_llm_usage_missing_call_count",
    "defense_llm_input_token_count",
    "defense_llm_cached_input_token_count",
    "defense_llm_output_token_count",
    "defense_llm_reasoning_token_count",
    "defense_llm_total_token_count",
    "defense_llm_estimated_cost_usd",
    "defense_llm_requests_per_episode",
    "defense_llm_api_calls_per_episode",
    "defense_llm_api_calls_per_action_step",
    "defense_llm_estimated_cost_usd_per_episode",
)

LLM_PRICING_COLUMNS = (
    "llm_input_usd_per_million",
    "llm_cached_input_usd_per_million",
    "llm_output_usd_per_million",
    "llm_pricing_as_of",
    "llm_pricing_source",
)

DEFENSE_ROUND_COLUMNS = (
    "defense_action_round_count",
    "gate_runtime_round_count",
    "gate_certification_round_count",
)


def aggregate_runs(
    runs: Sequence[RunRecord],
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> List[Dict[str, Any]]:
    """Convert loaded runs into output rows, including paired comparisons."""

    baseline_method = "none"
    rows: List[Dict[str, Any]] = []
    for run in runs:
        repair_calls_present, repair_calls = _metadata_entry(
            run.raw, ("repair_call_count",)
        )
        if not repair_calls_present:
            repair_calls = run.step_true_count("repair_attempted")
        repair_successes_present, repair_successes = _metadata_entry(
            run.raw, ("repair_success_count",)
        )
        if not repair_successes_present:
            repair_successes = run.step_true_count("repair_succeeded")
        judge_calls_present, judge_calls = _metadata_entry(
            run.raw, ("judge_call_count",)
        )
        if not judge_calls_present:
            judge_calls = run.step_true_count("judge_called")
        judge_failures_present, judge_failures = _metadata_entry(
            run.raw, ("judge_failure_count",)
        )
        if not judge_failures_present:
            judge_failures = run.step_true_count("judge_failed")
        judge_replacements_present, judge_replacements = _metadata_entry(
            run.raw, ("judge_replacement_count",)
        )
        if not judge_replacements_present:
            judge_replacements = run.step_true_count("judge_replaced")
        combined_calls_present, combined_calls = _metadata_entry(
            run.raw, ("repair_judge_call_count",)
        )
        if (
            not combined_calls_present
            and repair_calls is not None
            and judge_calls is not None
        ):
            combined_calls = int(repair_calls) + int(judge_calls)
        runtime_distribution = run.step_runtime_distribution()
        mean_runtime_present, mean_runtime = _metadata_entry(
            run.raw, ("mean_added_runtime_seconds", "added_runtime_seconds.mean")
        )
        if not mean_runtime_present:
            mean_runtime = (runtime_distribution or {}).get("mean")
        median_runtime_present, median_runtime = _metadata_entry(
            run.raw,
            ("median_added_runtime_seconds", "added_runtime_seconds.median"),
        )
        if not median_runtime_present:
            median_runtime = (runtime_distribution or {}).get("median")
        p95_runtime_present, p95_runtime = _metadata_entry(
            run.raw, ("p95_added_runtime_seconds", "added_runtime_seconds.p95")
        )
        if not p95_runtime_present:
            p95_runtime = (runtime_distribution or {}).get("p95")

        row: Dict[str, Any] = {
            "source": str(run.source),
            "method": run.method,
            "runtime_mode": run.runtime_mode,
            "attack_type": run.attack_type,
            "seed": run.seed,
            "target_brand": _metadata_value(run.raw, ("target_brand",)),
            "stress_test": _metadata_value(run.raw, ("stress_test",)),
            "oracle_mode": _metadata_value(run.raw, ("oracle_mode",)),
            "oracle_strategy": _metadata_value(run.raw, ("oracle_strategy",)),
            "checkpoint_path": _metadata_value(run.raw, ("checkpoint_path",)),
            "task_ids_path": _metadata_value(
                run.raw, ("task_ids_path", "test_ids_path")
            ),
            "task_ids_json": json.dumps(list(run.task_ids)) if run.task_ids else None,
            "task_count": run.task_count,
            "episode_count": run.episode_count,
            "parser_model_requested": _metadata_value(
                run.raw, ("parser_model_requested", "gate_openai_model")
            ),
            "parser_model_actual": _metadata_value(
                run.raw, ("parser_model_actual", "actual_parser_model")
            ),
            "parser_actual_models": _metadata_value(
                run.raw, ("parser_actual_models",)
            ),
            "parser_request_count": _metadata_value(
                run.raw, ("parser_request_count",)
            ),
            "parser_call_count": _metadata_value(run.raw, ("parser_call_count",)),
            "parser_api_call_count": _metadata_value(
                run.raw, ("parser_api_call_count",)
            ),
            "parser_cache_hit_count": _metadata_value(
                run.raw, ("parser_cache_hit_count",)
            ),
            "parser_usage_reported_call_count": _metadata_value(
                run.raw, ("parser_usage_reported_call_count",)
            ),
            "parser_usage_missing_call_count": _metadata_value(
                run.raw, ("parser_usage_missing_call_count",)
            ),
            "parser_input_token_count": _metadata_value(
                run.raw, ("parser_input_token_count",)
            ),
            "parser_cached_input_token_count": _metadata_value(
                run.raw, ("parser_cached_input_token_count",)
            ),
            "parser_output_token_count": _metadata_value(
                run.raw, ("parser_output_token_count",)
            ),
            "parser_reasoning_token_count": _metadata_value(
                run.raw, ("parser_reasoning_token_count",)
            ),
            "parser_total_token_count": _metadata_value(
                run.raw, ("parser_total_token_count",)
            ),
            "parser_estimated_cost_usd": _metadata_value(
                run.raw, ("parser_estimated_cost_usd",)
            ),
            "parser_fallback_count": _metadata_value(
                run.raw, ("parser_fallback_count",)
            ),
            "parser_error_count": _metadata_value(run.raw, ("parser_error_count",)),
            "judge_model": _metadata_value(run.raw, ("judge_model",)),
            "judge_actual_models": _metadata_value(
                run.raw, ("judge_actual_models",)
            ),
            "judge_request_count": _metadata_value(
                run.raw, ("judge_request_count",)
            ),
            "repair_call_count": repair_calls,
            "repair_success_count": repair_successes,
            "judge_call_count": judge_calls,
            "judge_cache_hit_count": _metadata_value(
                run.raw, ("judge_cache_hit_count",)
            ),
            "judge_usage_reported_call_count": _metadata_value(
                run.raw, ("judge_usage_reported_call_count",)
            ),
            "judge_usage_missing_call_count": _metadata_value(
                run.raw, ("judge_usage_missing_call_count",)
            ),
            "judge_input_token_count": _metadata_value(
                run.raw, ("judge_input_token_count",)
            ),
            "judge_cached_input_token_count": _metadata_value(
                run.raw, ("judge_cached_input_token_count",)
            ),
            "judge_output_token_count": _metadata_value(
                run.raw, ("judge_output_token_count",)
            ),
            "judge_reasoning_token_count": _metadata_value(
                run.raw, ("judge_reasoning_token_count",)
            ),
            "judge_total_token_count": _metadata_value(
                run.raw, ("judge_total_token_count",)
            ),
            "judge_estimated_cost_usd": _metadata_value(
                run.raw, ("judge_estimated_cost_usd",)
            ),
            "judge_failure_count": judge_failures,
            "judge_replacement_count": judge_replacements,
            "repair_judge_call_count": combined_calls,
            **{
                name: _metadata_value(run.raw, (name,))
                for name in DEFENSE_LLM_EFFICIENCY_COLUMNS
            },
            **{
                name: _metadata_value(run.raw, (name,))
                for name in LLM_PRICING_COLUMNS
            },
            **{
                name: _metadata_value(run.raw, (name,))
                for name in DEFENSE_ROUND_COLUMNS
            },
            "mean_added_runtime_seconds": mean_runtime,
            "median_added_runtime_seconds": median_runtime,
            "p95_added_runtime_seconds": p95_runtime,
            "oracle_eligible_count": _metadata_value(
                run.raw, ("oracle_eligible_count",)
            ),
            "oracle_eligibility_determined_count": _metadata_value(
                run.raw, ("oracle_eligibility_determined_count",)
            ),
            "oracle_ineligible_count": _metadata_value(
                run.raw, ("oracle_ineligible_count",)
            ),
            "git_commit_hash": _metadata_value(run.raw, ("git_commit_hash",)),
            "aer": run.aer,
            "aer_diff_vs_none": None,
            "aer_diff_ci_low": None,
            "aer_diff_ci_high": None,
            "paired_episode_count": None,
            "clean_improved": None,
            "clean_unchanged": None,
            "clean_harmed": None,
        }
        for metric_name in METRIC_ORDER:
            row[metric_name] = run.metric(metric_name)

        if not _method_matches(run, baseline_method):
            baseline = _find_baseline(run, runs, baseline_method)
            candidate_rewards = run.reward_map()
            baseline_rewards = baseline.reward_map() if baseline is not None else None
            if (
                baseline is not None
                and candidate_rewards is not None
                and baseline_rewards is not None
                and set(candidate_rewards) == set(baseline_rewards)
            ):
                ordered_ids = list(run.task_ids)
                candidate_values = [candidate_rewards[task_id] for task_id in ordered_ids]
                baseline_values = [baseline_rewards[task_id] for task_id in ordered_ids]
                point, low, high = paired_bootstrap_aer_difference(
                    candidate_values,
                    baseline_values,
                    samples=bootstrap_samples,
                    seed=BOOTSTRAP_SEED,
                )
                row["aer_diff_vs_none"] = point
                row["aer_diff_ci_low"] = low
                row["aer_diff_ci_high"] = high
                row["paired_episode_count"] = len(ordered_ids)
                if run.attack_type == "clean":
                    improved, unchanged, harmed = clean_change_counts(
                        candidate_rewards, baseline_rewards
                    )
                    row["clean_improved"] = improved
                    row["clean_unchanged"] = unchanged
                    row["clean_harmed"] = harmed
        rows.append(row)
    return rows


CSV_BASE_COLUMNS = (
    "source",
    "method",
    "runtime_mode",
    "attack_type",
    "seed",
    "target_brand",
    "stress_test",
    "oracle_mode",
    "oracle_strategy",
    "checkpoint_path",
    "task_ids_path",
    "task_ids_json",
    "task_count",
    "episode_count",
    "parser_model_requested",
    "parser_model_actual",
    "parser_actual_models",
    *PARSER_LLM_EFFICIENCY_COLUMNS,
    "parser_fallback_count",
    "parser_error_count",
    "judge_model",
    "judge_actual_models",
    "repair_call_count",
    "repair_success_count",
    *JUDGE_LLM_EFFICIENCY_COLUMNS,
    "judge_failure_count",
    "judge_replacement_count",
    "repair_judge_call_count",
    *DEFENSE_LLM_EFFICIENCY_COLUMNS,
    *LLM_PRICING_COLUMNS,
    *DEFENSE_ROUND_COLUMNS,
    "mean_added_runtime_seconds",
    "median_added_runtime_seconds",
    "p95_added_runtime_seconds",
    "oracle_eligible_count",
    "oracle_eligibility_determined_count",
    "oracle_ineligible_count",
    "git_commit_hash",
    "aer",
    "aer_diff_vs_none",
    "aer_diff_ci_low",
    "aer_diff_ci_high",
    "paired_episode_count",
    "clean_improved",
    "clean_unchanged",
    "clean_harmed",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _flat_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat = {column: _csv_value(row.get(column)) for column in CSV_BASE_COLUMNS}
    for name in METRIC_ORDER:
        metric: ProportionMetric = row[name]
        interval = metric.interval
        flat[f"{name}_numerator"] = _csv_value(metric.numerator)
        flat[f"{name}_denominator"] = _csv_value(metric.denominator)
        flat[f"{name}_percent"] = _csv_value(
            metric.proportion * 100.0 if metric.proportion is not None else None
        )
        flat[f"{name}_ci_low_percent"] = _csv_value(
            interval[0] * 100.0 if interval is not None else None
        )
        flat[f"{name}_ci_high_percent"] = _csv_value(
            interval[1] * 100.0 if interval is not None else None
        )
    return flat


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flat_csv_row(row) for row in rows]
    fieldnames = list(CSV_BASE_COLUMNS)
    for name in METRIC_ORDER:
        fieldnames.extend(
            (
                f"{name}_numerator",
                f"{name}_denominator",
                f"{name}_percent",
                f"{name}_ci_low_percent",
                f"{name}_ci_high_percent",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


TABLE_METRICS = (
    ("task_success", "Task success"),
    ("direct_asr", "Direct ASR"),
    ("indirect_conditional_asr", "Indirect ASR (cond.)"),
    ("indirect_unconditional_asr", "Indirect ASR (uncond.)"),
    ("trigger_exposure", "Trigger exposure"),
    ("target_click", "Target click"),
    ("target_purchase", "Target purchase"),
    ("oracle_eligible", "Oracle eligibility"),
    ("oracle_violating_click", "Oracle violating click | eligible"),
    ("oracle_strict_purchase", "Oracle strict purchase | eligible"),
    ("valid_action", "Valid action"),
    ("unparsable_action", "Unparsable action"),
    ("proposed_attack", "Proposed attack"),
    ("executed_attack", "Executed attack"),
    ("episode_intervention", "Episode intervention"),
    ("step_intervention", "Step intervention"),
)


def _format_metric(metric: ProportionMetric, missing: str = "—") -> str:
    if metric.is_missing:
        return missing
    if metric.numerator is not None or metric.denominator is not None:
        count_text = (
            f"{metric.numerator if metric.numerator is not None else '?'}/"
            f"{metric.denominator if metric.denominator is not None else '?'}"
        )
    else:
        count_text = "counts unavailable"
    if metric.proportion is None:
        return count_text
    interval = metric.interval
    if interval is None:
        return f"{count_text}; {100.0 * metric.proportion:.2f}%"
    return (
        f"{count_text}; {100.0 * metric.proportion:.2f}% "
        f"[{100.0 * interval[0]:.2f}, {100.0 * interval[1]:.2f}]"
    )


def _format_aer(value: Any, missing: str = "—") -> str:
    parsed = _finite_float(value)
    return f"{parsed:.4f}" if parsed is not None else missing


def _format_aer_difference(row: Mapping[str, Any], missing: str = "—") -> str:
    point = _finite_float(row.get("aer_diff_vs_none"))
    low = _finite_float(row.get("aer_diff_ci_low"))
    high = _finite_float(row.get("aer_diff_ci_high"))
    n = _integer(row.get("paired_episode_count"))
    if point is None or low is None or high is None or n is None:
        return missing
    return f"{point:+.4f} [{low:+.4f}, {high:+.4f}] (n={n})"


def _format_clean_changes(row: Mapping[str, Any], missing: str = "—") -> str:
    improved = _integer(row.get("clean_improved"))
    unchanged = _integer(row.get("clean_unchanged"))
    harmed = _integer(row.get("clean_harmed"))
    if improved is None or unchanged is None or harmed is None:
        return missing
    return f"{improved}/{unchanged}/{harmed}"


def _format_llm_efficiency(
    row: Mapping[str, Any], missing: str = "—"
) -> str:
    defense_rounds = [
        _integer(row.get("defense_action_round_count")),
        _integer(row.get("gate_runtime_round_count")),
        _integer(row.get("gate_certification_round_count")),
    ]
    request_counts = [
        _integer(row.get("defense_llm_request_count")),
        _integer(row.get("defense_llm_api_call_count")),
        _integer(row.get("defense_llm_cache_hit_count")),
    ]
    token_counts = [
        _integer(row.get("defense_llm_input_token_count")),
        _integer(row.get("defense_llm_cached_input_token_count")),
        _integer(row.get("defense_llm_output_token_count")),
        _integer(row.get("defense_llm_reasoning_token_count")),
        _integer(row.get("defense_llm_total_token_count")),
    ]
    cost = _finite_float(row.get("defense_llm_estimated_cost_usd"))
    if (
        all(value is None for value in defense_rounds + request_counts + token_counts)
        and cost is None
    ):
        return missing

    rounds_text = "/".join(
        str(value) if value is not None else missing for value in defense_rounds
    )
    request_text = "/".join(
        str(value) if value is not None else missing for value in request_counts
    )
    token_text = "/".join(
        str(value) if value is not None else missing for value in token_counts
    )
    cost_text = f"${cost:.8g}" if cost is not None else missing
    return f"{rounds_text}; {request_text}; {token_text}; {cost_text}"


def _table_headers() -> List[str]:
    return [
        "Method",
        "Mode",
        "Attack",
        "Stress test",
        "Oracle mode/strategy",
        "N",
        "Oracle eligible",
        "Repair calls/success; judge calls/failures/replacements",
        (
            "Defense rounds action/runtime/cert; LLM req/API/cache; "
            "tokens I/CI/O/R/T; USD"
        ),
        "Added runtime mean/median/p95 (s)",
        "AER",
        "AER difference vs none",
    ] + [
        label for _, label in TABLE_METRICS
    ] + ["Clean I/U/H"]


def _table_row(row: Mapping[str, Any], missing: str = "—") -> List[str]:
    repair_calls = _integer(row.get("repair_call_count"))
    repair_successes = _integer(row.get("repair_success_count"))
    judge_calls = _integer(row.get("judge_call_count"))
    judge_failures = _integer(row.get("judge_failure_count"))
    judge_replacements = _integer(row.get("judge_replacement_count"))
    call_text = (
        missing
        if all(
            value is None
            for value in (
                repair_calls,
                repair_successes,
                judge_calls,
                judge_failures,
                judge_replacements,
            )
        )
        else (
            f"{repair_calls if repair_calls is not None else missing}/"
            f"{repair_successes if repair_successes is not None else missing}; "
            f"{judge_calls if judge_calls is not None else missing}/"
            f"{judge_failures if judge_failures is not None else missing}/"
            f"{judge_replacements if judge_replacements is not None else missing}"
        )
    )
    runtimes = [
        _finite_float(row.get("mean_added_runtime_seconds")),
        _finite_float(row.get("median_added_runtime_seconds")),
        _finite_float(row.get("p95_added_runtime_seconds")),
    ]
    runtime_text = (
        missing
        if all(value is None for value in runtimes)
        else "/".join(
            f"{value:.4f}" if value is not None else missing for value in runtimes
        )
    )
    values = [
        str(row.get("method")) if row.get("method") is not None else missing,
        str(row.get("runtime_mode")) if row.get("runtime_mode") is not None else missing,
        str(row.get("attack_type")) if row.get("attack_type") is not None else missing,
        str(row.get("stress_test")) if row.get("stress_test") is not None else missing,
        (
            f"{row.get('oracle_mode') if row.get('oracle_mode') is not None else missing}/"
            f"{row.get('oracle_strategy') if row.get('oracle_strategy') is not None else missing}"
        ),
        str(row.get("episode_count")) if row.get("episode_count") is not None else missing,
        (
            str(row.get("oracle_eligible_count"))
            if row.get("oracle_eligible_count") is not None
            else missing
        ),
        call_text,
        _format_llm_efficiency(row, missing),
        runtime_text,
        _format_aer(row.get("aer"), missing),
        _format_aer_difference(row, missing),
    ]
    values.extend(_format_metric(row[name], missing) for name, _ in TABLE_METRICS)
    values.append(_format_clean_changes(row, missing))
    return values


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _table_headers()
    lines = [
        (
            "Proportions are shown as numerator/denominator; percent "
            "[Wilson 95% CI]. AER differences use a task-paired bootstrap "
            "95% CI with seed 42. Defense rounds are action/runtime/"
            "certification; LLM efficiency is logical requests/API calls/"
            "cache hits; input/cached-input/output/reasoning/total tokens; "
            "estimated USD. Missing values are not estimated."
        ),
        "",
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_escape(value) for value in _table_row(row))
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = _table_headers()
    lines = [
        "% Proportions: numerator/denominator; percent [Wilson 95% CI].",
        "% AER differences: task-paired bootstrap 95% CI, seed 42.",
        (
            "% Defense rounds: action/runtime/certification. LLM efficiency: "
            "logical requests/API calls/cache hits; input/cached-input/output/"
            "reasoning/total tokens; estimated USD."
        ),
        "% Missing values are shown as -- and are not estimated.",
        r"\begin{tabular}{" + "l" * len(headers) + "}",
        r"\hline",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = _table_row(row, missing="--")
        lines.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Summary JSON and/or per-episode JSON/JSONL files.",
    )
    parser.add_argument(
        "--csv",
        "--csv-path",
        "--csv-out",
        "--csv_path",
        dest="csv_path",
        type=Path,
        default=Path("rebuttal_aggregate.csv"),
        help="CSV output path (default: rebuttal_aggregate.csv).",
    )
    parser.add_argument(
        "--markdown",
        "--markdown-path",
        "--markdown-out",
        "--markdown_path",
        dest="markdown_path",
        type=Path,
        default=Path("rebuttal_aggregate.md"),
        help="Markdown output path (default: rebuttal_aggregate.md).",
    )
    parser.add_argument(
        "--latex",
        "--latex-path",
        "--latex-out",
        "--latex_path",
        dest="latex_path",
        type=Path,
        default=Path("rebuttal_aggregate.tex"),
        help="LaTeX output path (default: rebuttal_aggregate.tex).",
    )
    parser.add_argument(
        "--allow-mismatched-task-ids",
        "--allow-mismatched-task-sets",
        "--allow_mismatched_task_ids",
        "--allow_task_id_mismatch",
        action="store_true",
        help=(
            "Allow an explicitly unpaired report when task IDs differ or are missing. "
            "Paired comparisons remain blank."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap_samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            f"Paired bootstrap draws (default: {DEFAULT_BOOTSTRAP_SAMPLES}; "
            "seed is fixed at 42)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise AggregationError("--bootstrap-samples must be positive")

    runs = load_runs(args.inputs)
    mismatches = validate_identical_task_ids(
        runs, allow_mismatched=args.allow_mismatched_task_ids
    )
    for warning in mismatches:
        print(f"WARNING: {warning}", file=sys.stderr)
    rows = aggregate_runs(
        runs,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_csv(args.csv_path, rows)
    write_markdown(args.markdown_path, rows)
    write_latex(args.latex_path, rows)

    print(f"Aggregated {len(rows)} run(s).")
    print(f"CSV: {args.csv_path}")
    print(f"Markdown: {args.markdown_path}")
    print(f"LaTeX: {args.latex_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
