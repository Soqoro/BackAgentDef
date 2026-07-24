"""Public-interface collection for the fixed WebShop comparison shortlist.

The defense uses a session isolated from the policy rollout and interacts with
it only through reset/step, available actions, and rendered HTML. Frozen
benchmark labels are used later only to detect environment drift; they are not
copied into the runtime ledger.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from .schema import Candidate
from .webshop_adapter import (
    hard_feasibility,
    normalize_brand,
    normalize_text,
    parse_price,
    parse_rating,
)


_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


class PublicLedgerError(RuntimeError):
    """Raised when the comparison session cannot produce auditable evidence."""


@dataclass
class _HtmlNode:
    tag: str
    attrs: dict[str, str]
    content: list[Any] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def text(self) -> str:
        parts = [
            item.text() if isinstance(item, _HtmlNode) else str(item)
            for item in self.content
        ]
        return re.sub(r"\s+", " ", " ".join(parts)).strip()


class _PublicHTMLParser(HTMLParser):
    _VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("document", {})
        self._stack = [self.root]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        node = _HtmlNode(
            tag.lower(),
            {
                str(key).lower(): str(value or "")
                for key, value in attrs
            },
        )
        self._stack[-1].content.append(node)
        if node.tag not in self._VOID:
            self._stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        wanted = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == wanted:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].content.append(data)


def _document(html: Any) -> _HtmlNode:
    parser = _PublicHTMLParser()
    parser.feed(str(html or ""))
    parser.close()
    return parser.root


def _nodes(
    node: _HtmlNode,
    *,
    class_name: str | None = None,
    tag: str | None = None,
) -> list[_HtmlNode]:
    matches: list[_HtmlNode] = []
    for item in node.content:
        if not isinstance(item, _HtmlNode):
            continue
        if (
            (class_name is None or class_name in item.classes)
            and (tag is None or tag == item.tag)
        ):
            matches.append(item)
        matches.extend(_nodes(item, class_name=class_name, tag=tag))
    return matches


def _first(
    node: _HtmlNode,
    class_name: str,
) -> _HtmlNode | None:
    matches = _nodes(node, class_name=class_name)
    return matches[0] if matches else None


def _text(node: _HtmlNode | None) -> str:
    return node.text() if node is not None else ""


def _strip_label(value: str, label: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(label)}\s*:\s*",
        "",
        str(value or ""),
        flags=re.IGNORECASE,
    ).strip()


def parse_search_results(html: Any) -> list[dict[str, Any]]:
    """Parse only values visibly rendered on a WebShop results page."""

    parsed: list[dict[str, Any]] = []
    for card in _nodes(_document(html), class_name="list-group-item"):
        asin = _text(_first(card, "product-asin")).upper()
        if not _ASIN_RE.fullmatch(asin):
            continue
        parsed.append(
            {
                "asin": asin,
                "title": _text(_first(card, "product-title")),
                "brand": _strip_label(
                    _text(_first(card, "product-brand")),
                    "Brand",
                ),
                "price": parse_price(_text(_first(card, "product-price"))),
                "rating": parse_rating(
                    _text(_first(card, "product-rating"))
                ),
                "availability": _strip_label(
                    _text(_first(card, "product-availability")),
                    "Availability",
                ),
            }
        )
    return parsed


def parse_item_page(html: Any) -> dict[str, Any]:
    """Parse identity, comparison fields, and option values from an item page."""

    document = _document(html)
    options: dict[str, list[str]] = {}
    for node in _nodes(document, tag="input"):
        if node.attrs.get("type", "").lower() != "radio":
            continue
        name = str(node.attrs.get("name") or "").strip()
        value = str(node.attrs.get("value") or "").strip()
        if name and value and value not in options.setdefault(name, []):
            options[name].append(value)
    return {
        "asin": _strip_label(
            _text(_first(document, "product-asin")),
            "ASIN",
        ).upper(),
        "title": _text(_first(document, "product-title")),
        "brand": _strip_label(
            _text(_first(document, "product-brand")),
            "Brand",
        ),
        "price": parse_price(_text(_first(document, "product-price"))),
        "rating": parse_rating(_text(_first(document, "product-rating"))),
        "availability": _strip_label(
            _text(_first(document, "product-availability")),
            "Availability",
        ),
        "options": options,
    }


def parse_feature_page(html: Any) -> list[str]:
    return [
        text
        for text in (
            _text(node)
            for node in _nodes(_document(html), class_name="product-info")
        )
        if text
    ]


def parse_description_page(html: Any) -> str:
    return " ".join(parse_feature_page(html)).strip()


def _text_parts(observation: Any) -> list[str]:
    return [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(
            r"\s*\[SEP]\s*",
            str(observation or ""),
            flags=re.IGNORECASE,
        )
        if re.sub(r"\s+", " ", part).strip()
    ]


def _labeled_text(part: str, label: str) -> str | None:
    match = re.match(
        rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
        str(part or ""),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def parse_text_search_results(
    observation: Any,
    *,
    allowed_asins: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse visible cards from a policy's ``text``-mode observation.

    ``allowed_asins`` should be the current public clickable set.  Supplying it
    prevents ASIN-shaped text in the instruction or product prose from being
    interpreted as a result identity.
    """

    allowed = (
        {
            str(value).strip().upper()
            for value in allowed_asins
            if _ASIN_RE.fullmatch(str(value).strip())
        }
        if allowed_asins is not None
        else None
    )
    parts = _text_parts(observation)
    page: int | None = None
    for part in parts:
        match = re.match(r"^\s*Page\s+(\d+)\b", part, flags=re.IGNORECASE)
        if match is not None:
            page = int(match.group(1))
            break

    positions = [
        index
        for index, part in enumerate(parts)
        if _ASIN_RE.fullmatch(part)
        and (allowed is None or part.upper() in allowed)
    ]
    parsed: list[dict[str, Any]] = []
    for rank, index in enumerate(positions, start=1):
        asin = parts[index].upper()
        stop = positions[rank] if rank < len(positions) else len(parts)
        card = parts[index + 1 : stop]
        if not card:
            continue

        title = card[0]
        if (
            _ASIN_RE.fullmatch(title)
            or any(
                _labeled_text(title, label) is not None
                for label in ("Brand", "Price", "Rating", "Availability")
            )
        ):
            title = ""
        brand = ""
        availability = ""
        price = None
        rating = None
        for part in card[1:] if title else card:
            brand_value = _labeled_text(part, "Brand")
            if brand_value is not None:
                brand = brand_value
                continue
            availability_value = _labeled_text(part, "Availability")
            if availability_value is not None:
                availability = availability_value
                continue
            if _labeled_text(part, "Rating") is not None:
                rating = parse_rating(part)
                continue
            if re.match(
                r"^\s*(?:Price\s*:\s*)?(?:US(?:D)?\s*)?\$",
                part,
                flags=re.IGNORECASE,
            ):
                price = parse_price(part)

        parsed.append(
            {
                "asin": asin,
                "title": title,
                "brand": brand,
                "price": price,
                "rating": rating,
                "availability": availability,
                "result_rank": rank,
                "page": page,
            }
        )
    return parsed


def parse_text_item_page(observation: Any) -> dict[str, Any]:
    """Parse the rendered identity and scalars from a text-mode item page."""

    parts = _text_parts(observation)
    asin = ""
    identity_index: int | None = None
    for index, part in enumerate(parts):
        value = _labeled_text(part, "ASIN")
        if value is not None and _ASIN_RE.fullmatch(value):
            asin = value.upper()
            identity_index = index
            break

    title = ""
    if identity_index is not None and identity_index + 1 < len(parts):
        possible_title = parts[identity_index + 1]
        if not any(
            _labeled_text(possible_title, label) is not None
            for label in ("Brand", "Price", "Rating", "Availability")
        ):
            title = possible_title

    brand = ""
    availability = ""
    price = None
    rating = None
    scan = parts[(identity_index + 1) if identity_index is not None else 0 :]
    for part in scan:
        brand_value = _labeled_text(part, "Brand")
        if brand_value is not None:
            brand = brand_value
            continue
        availability_value = _labeled_text(part, "Availability")
        if availability_value is not None:
            availability = availability_value
            continue
        if _labeled_text(part, "Price") is not None:
            price = parse_price(part)
            continue
        if _labeled_text(part, "Rating") is not None:
            rating = parse_rating(part)

    return {
        "asin": asin,
        "title": title,
        "brand": brand,
        "price": price,
        "rating": rating,
        "availability": availability,
    }


def parse_text_subpage(observation: Any) -> list[str]:
    """Return public Description/Features content after the last back control."""

    parts = _text_parts(observation)
    previous = [
        index
        for index, part in enumerate(parts)
        if normalize_text(part) == normalize_text("< Prev")
    ]
    if not previous:
        return []
    return parts[previous[-1] + 1 :]


def _available_clickables(env: Any) -> set[str]:
    available = env.get_available_actions()
    if not isinstance(available, Mapping):
        return set()
    return {
        normalize_text(item)
        for item in (available.get("clickables") or [])
        if normalize_text(item)
    }


def _perform(
    env: Any,
    action: str,
    trace: list[dict[str, Any]],
) -> str:
    started = time.perf_counter()
    observation, reward, done, _info = env.step(action)
    elapsed = time.perf_counter() - started
    html = str(observation or "")
    if not html.strip():
        raise PublicLedgerError(
            f"comparison action returned an empty HTML observation: {action}"
        )
    observation_digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    trace.append(
        {
            "action": action,
            "latency_seconds": elapsed,
            "reward": float(reward or 0.0),
            "done": bool(done),
            "observation_sha256": observation_digest,
            # Keep the established trace field while making explicit that the
            # HTML is the public observation returned by ``step``.
            "html_sha256": observation_digest,
        }
    )
    if done:
        raise PublicLedgerError(
            f"comparison action unexpectedly terminated the shop: {action}"
        )
    return html


def _merge_visible(
    search_record: Mapping[str, Any],
    item_record: Mapping[str, Any],
) -> dict[str, Any]:
    asin = str(item_record.get("asin") or "").upper()
    expected = str(search_record.get("asin") or "").upper()
    if asin != expected:
        raise PublicLedgerError(
            f"opened item identity mismatch: expected {expected}, observed {asin}"
        )
    merged: dict[str, Any] = {}
    for field in ("title", "brand", "availability"):
        item_value = item_record.get(field)
        search_value = search_record.get(field)
        if item_value and search_value and normalize_text(item_value) != normalize_text(
            search_value
        ):
            raise PublicLedgerError(
                f"search/item {field} mismatch for {asin}: "
                f"{search_value!r} != {item_value!r}"
            )
        merged[field] = item_value or search_value or ""
    for field in ("price", "rating"):
        item_value = item_record.get(field)
        search_value = search_record.get(field)
        if (
            item_value is not None
            and search_value is not None
            and float(item_value) != float(search_value)
        ):
            raise PublicLedgerError(
                f"search/item {field} mismatch for {asin}: "
                f"{search_value!r} != {item_value!r}"
            )
        merged[field] = (
            item_value if item_value is not None else search_value
        )
    merged["asin"] = asin
    merged["options"] = dict(item_record.get("options") or {})
    return merged


def collect_fixed_shortlist(
    env: Any,
    *,
    hard_constraints: Mapping[str, Any],
    canonical_query: str,
    shortlist_size: int,
    page_size: int,
) -> tuple[tuple[Candidate, ...], list[dict[str, Any]]]:
    """Collect and verify a fixed shortlist through public WebShop actions."""

    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    if page_size != 10:
        raise ValueError("WebShop public collector currently requires page_size=10")

    trace: list[dict[str, Any]] = []
    results_html = _perform(env, f"search[{canonical_query}]", trace)
    search_records = parse_search_results(results_html)[:shortlist_size]
    if len(search_records) != shortlist_size:
        raise PublicLedgerError(
            "comparison search exposed "
            f"{len(search_records)} valid products; expected {shortlist_size}"
        )

    candidates: list[Candidate] = []
    for rank, search_record in enumerate(search_records, start=1):
        asin = str(search_record["asin"])
        if normalize_text(asin) not in _available_clickables(env):
            raise PublicLedgerError(
                f"shortlist product {asin} is not publicly clickable"
            )

        item_html = _perform(env, f"click[{asin}]", trace)
        item_record = parse_item_page(item_html)
        visible = _merge_visible(search_record, item_record)

        if normalize_text("Attributes") in _available_clickables(env):
            raise PublicLedgerError(
                "comparison session exposes the annotation-backed Attributes "
                "page; construct it with show_attrs=False"
            )

        description = ""
        features: list[str] = []
        for label in ("Description", "Features"):
            if normalize_text(label) not in _available_clickables(env):
                raise PublicLedgerError(
                    f"comparison item page does not expose public {label}"
                )
            subpage_html = _perform(env, f"click[{label}]", trace)
            if label == "Description":
                description = parse_description_page(subpage_html)
            else:
                features = parse_feature_page(subpage_html)
            if normalize_text("< Prev") not in _available_clickables(env):
                raise PublicLedgerError(
                    f"{label} page has no public back action"
                )
            _perform(env, "click[< Prev]", trace)

        product = {
            "Title": visible["title"],
            "Brand": visible["brand"],
            "Availability": visible["availability"],
            # These are ordinary rendered catalog descriptions, not WebShop's
            # evaluator-authored ``Attributes``/``query`` annotations.
            "Attributes": [description, *features],
            "Description": description,
            "BulletPoints": features,
            "options": visible["options"],
        }
        price = visible["price"]
        rating = visible["rating"]
        feasible, selected_options, failures = hard_feasibility(
            goal=hard_constraints,
            product=product,
            price=price,
        )
        evidence = {
            "retrieval": {
                "query": canonical_query,
                "rank": rank,
                "page": 1,
                "source": "public_defense_comparison_session",
            },
            "title": {
                "value": visible["title"],
                "source": "displayed_search_and_product_page",
            },
            "brand": {
                "value": normalize_brand(visible["brand"]),
                "source": "displayed_search_and_product_page",
            },
            "price": {
                "value": price,
                "source": "displayed_search_and_product_page",
            },
            "rating": {
                "value": rating,
                "source": "displayed_search_and_product_page",
            },
            "attributes": {
                "value": [description, *features],
                "source": "displayed_description_and_features_pages",
            },
            "product_type": {
                "value": hard_constraints.get("query"),
                "source": "verified_from_displayed_catalog_text",
            },
            "options": {
                "value": visible["options"],
                "source": "displayed_product_page",
            },
            "availability": {
                "value": visible["availability"],
                "source": "displayed_search_and_product_page",
            },
        }
        candidates.append(
            Candidate(
                asin=asin,
                options=selected_options,
                feasible=feasible,
                price=price,
                rating=rating,
                brand=normalize_brand(visible["brand"]) or None,
                title=str(visible["title"] or "") or None,
                evidence=evidence,
                shortlist_rank=rank,
                page=1,
                metadata={"hard_feasibility_failures": failures},
            )
        )

        if normalize_text("< Prev") not in _available_clickables(env):
            raise PublicLedgerError(f"item page for {asin} has no results back action")
        _perform(env, "click[< Prev]", trace)

    return tuple(candidates), trace
