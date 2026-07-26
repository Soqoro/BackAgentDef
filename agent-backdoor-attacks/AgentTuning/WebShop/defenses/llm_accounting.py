"""Dependency-free token and cost accounting for LLM API responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _validated_rate(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number or None")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite nonnegative number or None"
        ) from exc
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number or None")
    return rate


def _validated_tokens(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(value, int):
        tokens = value
    elif isinstance(value, float) and value.is_integer():
        tokens = int(value)
    elif isinstance(value, str):
        try:
            tokens = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a nonnegative integer") from exc
    else:
        raise ValueError(f"{name} must be a nonnegative integer")
    if tokens < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return tokens


def _value(container: Any, *names: str) -> Any:
    """Read the first non-None mapping key or object attribute."""

    if container is None:
        return None
    for name in names:
        if isinstance(container, Mapping):
            candidate = container.get(name)
        else:
            candidate = getattr(container, name, None)
        if candidate is not None:
            return candidate
    return None


def _count(container: Any, *names: str) -> int:
    value = _value(container, *names)
    return 0 if value is None else _validated_tokens(names[0], value)


@dataclass(frozen=True)
class LLMPricing:
    """USD prices per one million tokens.

    Input and output prices must either both be supplied or both be omitted.
    When configured pricing omits a cached-input price, cached tokens use the
    ordinary input price.
    """

    input_usd_per_million: Optional[float] = None
    output_usd_per_million: Optional[float] = None
    cached_input_usd_per_million: Optional[float] = None

    def __post_init__(self) -> None:
        input_rate = _validated_rate(
            "input_usd_per_million", self.input_usd_per_million
        )
        output_rate = _validated_rate(
            "output_usd_per_million", self.output_usd_per_million
        )
        cached_rate = _validated_rate(
            "cached_input_usd_per_million",
            self.cached_input_usd_per_million,
        )
        if (input_rate is None) != (output_rate is None):
            raise ValueError(
                "input_usd_per_million and output_usd_per_million must "
                "both be supplied or both be omitted"
            )
        if cached_rate is not None and input_rate is None:
            raise ValueError(
                "cached_input_usd_per_million requires input and output pricing"
            )
        object.__setattr__(self, "input_usd_per_million", input_rate)
        object.__setattr__(self, "output_usd_per_million", output_rate)
        object.__setattr__(self, "cached_input_usd_per_million", cached_rate)

    @property
    def is_configured(self) -> bool:
        return (
            self.input_usd_per_million is not None
            and self.output_usd_per_million is not None
        )

    @property
    def effective_cached_input_usd_per_million(self) -> Optional[float]:
        if not self.is_configured:
            return None
        return (
            self.cached_input_usd_per_million
            if self.cached_input_usd_per_million is not None
            else self.input_usd_per_million
        )

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "cached_input_usd_per_million": (
                self.cached_input_usd_per_million
            ),
        }


@dataclass(frozen=True)
class LLMUsage:
    """Normalized usage from one or more LLM responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    model: Optional[str] = None
    usage_reported: bool = True

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _validated_tokens(name, getattr(self, name)),
            )
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.model is not None:
            model = str(self.model).strip()
            object.__setattr__(self, "model", model or None)
        if not isinstance(self.usage_reported, bool):
            raise ValueError("usage_reported must be a bool")

    @classmethod
    def from_response(cls, response: Any) -> "LLMUsage":
        """Normalize Chat Completions-style mapping or object responses."""

        usage = _value(response, "usage")
        if usage is None:
            usage = response

        # A present-but-empty or partial usage object is not enough to compute
        # spend. Require both billed sides so it is surfaced as missing rather
        # than silently becoming a zero-cost call.
        raw_input = _value(usage, "prompt_tokens", "input_tokens")
        raw_output = _value(usage, "completion_tokens", "output_tokens")
        usage_reported = raw_input is not None and raw_output is not None

        input_tokens = (
            0
            if raw_input is None
            else _validated_tokens("prompt_tokens", raw_input)
        )
        output_tokens = (
            0
            if raw_output is None
            else _validated_tokens("completion_tokens", raw_output)
        )
        raw_total = _value(usage, "total_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if raw_total is None
            else _validated_tokens("total_tokens", raw_total)
        )

        input_details = _value(
            usage,
            "prompt_tokens_details",
            "input_tokens_details",
        )
        output_details = _value(
            usage,
            "completion_tokens_details",
            "output_tokens_details",
        )
        cached_input_tokens = _count(
            usage,
            "cached_input_tokens",
            "cached_tokens",
            "prompt_cache_hit_tokens",
        )
        if cached_input_tokens == 0:
            cached_input_tokens = _count(
                input_details,
                "cached_tokens",
                "cached_input_tokens",
            )
        reasoning_tokens = _count(usage, "reasoning_tokens")
        if reasoning_tokens == 0:
            reasoning_tokens = _count(
                output_details,
                "reasoning_tokens",
            )

        model = _value(response, "model", "response_model")
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            model=str(model) if model is not None else None,
            usage_reported=usage_reported,
        )

    def estimated_cost_usd(
        self,
        pricing: Optional[LLMPricing],
    ) -> Optional[float]:
        """Estimate provider cost, or return None when pricing is absent."""

        if (
            not self.usage_reported
            or pricing is None
            or not pricing.is_configured
        ):
            return None
        assert pricing.input_usd_per_million is not None
        assert pricing.output_usd_per_million is not None
        cached_rate = pricing.effective_cached_input_usd_per_million
        assert cached_rate is not None
        uncached_input_tokens = self.input_tokens - self.cached_input_tokens
        return (
            uncached_input_tokens * pricing.input_usd_per_million
            + self.cached_input_tokens * cached_rate
            + self.output_tokens * pricing.output_usd_per_million
        ) / 1_000_000.0

    def __add__(self, other: object) -> "LLMUsage":
        if not isinstance(other, LLMUsage):
            return NotImplemented
        if self.model == other.model:
            model = self.model
        elif self.model is None:
            model = other.model
        elif other.model is None:
            model = self.model
        else:
            # A token aggregate can span models; callers that need model
            # attribution should retain the individual response model names.
            model = None
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=(
                self.cached_input_tokens + other.cached_input_tokens
            ),
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            model=model,
            usage_reported=self.usage_reported and other.usage_reported,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "usage_reported": self.usage_reported,
        }
