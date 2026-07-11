"""Dependency-free metric helpers shared by rebuttal evaluation/tests."""

from __future__ import annotations

from typing import Any, Dict, Optional


def proportion_summary(numerator: int, denominator: int) -> Dict[str, Any]:
    """Return raw counts and rates without inventing a zero-denominator value."""

    rate: Optional[float] = None if denominator == 0 else numerator / denominator
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "percent": None if rate is None else 100.0 * rate,
    }


def attack_metric_summaries(
    attack_type: str,
    attack_hits: int,
    trigger_count: int,
    episodes: int,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute direct and both conditional/unconditional indirect ASR metrics."""

    direct = None
    indirect_conditional = None
    indirect_unconditional = None
    paper_style = None
    if attack_type == "query_attack":
        direct = proportion_summary(attack_hits, episodes)
        paper_style = direct
    elif attack_type == "observation_attack":
        indirect_conditional = proportion_summary(attack_hits, trigger_count)
        indirect_unconditional = proportion_summary(attack_hits, episodes)
        paper_style = indirect_conditional
    return {
        "paper_style_asr": paper_style,
        "direct_paper_style_asr": direct,
        "indirect_conditional_asr": indirect_conditional,
        "indirect_unconditional_asr": indirect_unconditional,
    }


__all__ = ["attack_metric_summaries", "proportion_summary"]
