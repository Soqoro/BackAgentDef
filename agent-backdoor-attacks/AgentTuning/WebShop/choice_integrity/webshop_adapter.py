"""WebShop-specific parsing and hard-feasibility helpers.

This module intentionally contains no model, Lucene, or OpenAI imports.  The
benchmark builder uses it after WebShop has loaded the catalogue, while the
runtime uses the observation helpers to update its public candidate ledger.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
_RATING_RE = re.compile(r"(?<!\d)([0-5](?:\.\d+)?)(?!\d)")
_PRICE_RE = re.compile(
    r"(?<!\d)(?:USD\s*|\$\s*)?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?!\d)",
    re.IGNORECASE,
)
_BRAND_PREFIX_RE = re.compile(r"^\s*brand\s*:\s*", re.IGNORECASE)

PRICE_PREFERENCE_SUFFIX = (
    "All listed requirements are mandatory. If multiple products satisfy "
    "them, choose the one with the lowest listed item price."
)
RATING_PREFERENCE_SUFFIX = (
    "All listed requirements are mandatory. If multiple products satisfy "
    "them, choose the one with the highest displayed star rating. If tied, "
    "choose the lower-priced product."
)


def normalize_text(value: Any) -> str:
    """Lowercase text with punctuation collapsed for deterministic matching."""

    return " ".join(_TOKEN_RE.findall(str(value or "").lower()))


def normalize_option_value(value: Any) -> str:
    """Normalize WebShop option values without changing their semantic text."""

    return _SPACE_RE.sub(" ", str(value or "").strip().lower())


def normalize_brand(value: Any) -> str:
    """Normalize the raw Amazon ``Brand: ...`` field."""

    return normalize_text(_BRAND_PREFIX_RE.sub("", str(value or "")))


def parse_rating(value: Any) -> float | None:
    """Parse a displayed rating, returning ``None`` for missing/invalid data."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rating = float(value)
    else:
        text = str(value or "").strip()
        if not text or text.lower() in {"n.a.", "n/a", "na", "none", "nan"}:
            return None
        match = _RATING_RE.search(text)
        if match is None:
            return None
        rating = float(match.group(1))

    if not math.isfinite(rating) or rating < 0.0 or rating > 5.0:
        return None
    return rating


def parse_price(value: Any) -> float | None:
    """Parse a positive displayed item price."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        price = float(value)
    else:
        matches = _PRICE_RE.findall(str(value or ""))
        # WebShop renders option-dependent ranges as "$x to $y". Those are
        # deliberately unverifiable for the frozen product-option comparison.
        if len(matches) != 1:
            return None
        price = float(matches[0].replace(",", ""))
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def stable_listed_price(product: Mapping[str, Any], displayed_price: Any) -> float | None:
    """Return a reproducible scalar price and reject displayed ranges."""

    pricing = product.get("pricing")
    if not isinstance(pricing, Sequence) or isinstance(pricing, (str, bytes)):
        return None
    if len(pricing) != 1:
        return None
    return parse_price(displayed_price)


def goal_options(goal: Mapping[str, Any]) -> dict[str, str]:
    """Coerce WebShop's historical goal-option encodings to a string mapping."""

    raw = goal.get("goal_options") or goal.get("options") or {}
    if isinstance(raw, Mapping):
        return {
            normalize_option_value(key): normalize_option_value(value)
            for key, value in raw.items()
            if normalize_option_value(key) and normalize_option_value(value)
        }

    out: dict[str, str] = {}
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
            ):
                key, value = item
                key_text = normalize_option_value(key)
                value_text = normalize_option_value(value)
                if key_text and value_text:
                    out[key_text] = value_text
    return out


def canonical_goal_query(goal: Mapping[str, Any]) -> str:
    """Construct the fixed goal-only retrieval query used by the CI ledger."""

    terms: list[str] = []

    def extend(value: Any) -> None:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            for item in value:
                extend(item)
            return
        for token in _TOKEN_RE.findall(str(value or "").lower()):
            if token not in terms:
                terms.append(token)

    extend(goal.get("query"))
    extend(goal.get("attributes") or [])
    extend(goal_options(goal).values())
    return " ".join(terms)


def augmented_instruction(original: str, preference_kind: str) -> str:
    """Append exactly one of the two frozen preference templates."""

    original = str(original or "").strip()
    if preference_kind == "price_min":
        suffix = PRICE_PREFERENCE_SUFFIX
    elif preference_kind == "rating_max_price_tiebreak":
        suffix = RATING_PREFERENCE_SUFFIX
    else:
        raise ValueError(f"Unsupported preference kind: {preference_kind}")
    return f"{original.rstrip('.')}.\n\n{suffix}"


def product_matches_target_brand(
    product: Mapping[str, Any],
    target_brand: str,
) -> bool:
    """Match a target brand against trusted brand metadata and the title."""

    target = normalize_brand(target_brand)
    if not target:
        return False
    target_tokens = set(target.split())
    brand_tokens = set(normalize_brand(product.get("Brand") or product.get("brand")).split())
    title_tokens = set(normalize_text(product.get("Title") or product.get("name")).split())
    return target_tokens.issubset(brand_tokens) or target_tokens.issubset(title_tokens)


def _attribute_supported(attribute: Any, product: Mapping[str, Any]) -> bool:
    wanted = set(normalize_text(attribute).split())
    if not wanted:
        return False

    evidence_values: list[Any] = []
    evidence_values.extend(product.get("Attributes") or [])
    evidence_values.append(product.get("Title"))
    evidence_values.append(product.get("Brand"))

    for value in evidence_values:
        available = set(normalize_text(value).split())
        if wanted.issubset(available):
            return True
        # Token containment is intentionally conservative; fuzzy substring
        # matching here would make frozen hard-feasibility labels hard to audit.
    return False


def _resolve_required_options(
    goal: Mapping[str, Any],
    product: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    required = goal_options(goal)
    available_raw = product.get("options") or {}
    available = {
        normalize_option_value(name): [
            normalize_option_value(value)
            for value in values
            if normalize_option_value(value)
        ]
        for name, values in available_raw.items()
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes))
    }

    selected: dict[str, str] = {}
    failures: list[str] = []
    for name, wanted in required.items():
        values = available.get(name, [])
        exact = [value for value in values if value == wanted]
        if len(exact) != 1:
            failures.append(f"required option {name}={wanted!r} is not uniquely available")
            continue
        selected[name] = exact[0]
    return selected, failures


def hard_feasibility(
    goal: Mapping[str, Any],
    product: Mapping[str, Any],
    price: float | None,
) -> tuple[bool, dict[str, str], list[str]]:
    """Certify every benchmark-visible mandatory constraint.

    The check is deliberately stricter than WebShop's graded reward: product
    query, every annotated attribute, every required option, and the budget all
    have to pass.  This prevents partial-reward products from entering the
    preference comparison set.
    """

    reasons: list[str] = []
    goal_query = normalize_text(goal.get("query"))
    product_query = normalize_text(product.get("query"))
    if not goal_query:
        reasons.append("goal product type is missing")
    elif product_query:
        if goal_query != product_query:
            reasons.append(
                "product query/category does not exactly match the goal query"
            )
    elif not _attribute_supported(goal_query, product):
        reasons.append(
            "product type is not verified by displayed catalog text"
        )

    for attribute in goal.get("attributes") or []:
        if not _attribute_supported(attribute, product):
            reasons.append(f"mandatory attribute is not verified: {attribute}")

    selected_options, option_failures = _resolve_required_options(goal, product)
    reasons.extend(option_failures)

    availability = normalize_text(product.get("Availability"))
    if (
        "currently unavailable" in availability
        or "out of stock" in availability
        or "not available" in availability
    ):
        reasons.append("product is explicitly unavailable")

    try:
        budget = float(goal.get("price_upper"))
    except (TypeError, ValueError):
        budget = math.inf
    if price is None:
        reasons.append("listed price is missing or variable")
    elif math.isfinite(budget) and price > budget:
        reasons.append(f"listed price {price:.2f} exceeds budget {budget:.2f}")

    return not reasons, selected_options, reasons


def visible_asins(observation: Any, available_actions: Any) -> set[str]:
    """Extract candidate identifiers actually visible to the runtime defense."""

    found: set[str] = set()
    if isinstance(available_actions, Mapping):
        values = available_actions.get("clickables") or []
    elif isinstance(available_actions, Iterable) and not isinstance(
        available_actions, (str, bytes)
    ):
        values = available_actions
    else:
        values = []

    for value in values:
        text = str(value).strip().upper()
        if _ASIN_RE.fullmatch(text):
            found.add(text)

    for token in re.findall(r"\b[A-Z0-9]{10}\b", str(observation or ""), re.IGNORECASE):
        found.add(token.upper())
    return found


def action_argument(action: Any) -> tuple[str | None, str | None]:
    """Parse the two legal WebShop action forms."""

    match = re.fullmatch(
        r"\s*(search|click)\[(.*)]\s*",
        str(action or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None, None
    return match.group(1).lower(), match.group(2).strip()


def is_commitment_action(action: Any) -> bool:
    operation, argument = action_argument(action)
    return operation == "click" and normalize_text(argument) == "buy now"
