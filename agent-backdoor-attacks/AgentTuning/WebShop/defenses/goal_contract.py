"""
Module 1: Goal Contract Extraction.

Given a raw user query q, this module extracts the goal contract

    G(q) = (I, C+, C-, P)

where I is the high-level task intent, C+ is the set of explicit positive
constraints, C- is the set of explicit forbidden constraints/actions, and P is
an optional provenance-bound comparative preference supplied by an authorized
deterministic parser.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .llm_accounting import LLMPricing, LLMUsage

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


GOAL_CONTRACT_SCHEMA_VERSION = 2


def goal_contract_cache_key(
    instruction: str,
    parser_model: str,
    *,
    schema_version: int = GOAL_CONTRACT_SCHEMA_VERSION,
) -> str:
    """Hash the parser schema, exact instruction, and requested model."""

    payload = json.dumps(
        [schema_version, instruction, parser_model],
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
    """Structured goal contract G(q) = (I, C+, C-, P)."""

    raw_query: str
    intent: str
    positive_constraints: List[str] = field(default_factory=list)
    negative_constraints: List[str] = field(default_factory=list)
    product_type: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    options: Dict[str, str] = field(default_factory=dict)
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    extractor: str = "unknown"
    extraction_error: Optional[str] = None
    comparative_preference: Optional[Dict[str, Any]] = None
    preference_provenance: Optional[Dict[str, Any]] = None

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
            "product_type": self.product_type,
            "attributes": list(self.attributes),
            "options": dict(self.options),
            "max_price": self.max_price,
            "min_rating": self.min_rating,
            "P": self.comparative_preference,
            "P_provenance": self.preference_provenance,
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
    def constraints(self) -> Dict[str, Any]:
        """Compatibility shim for the previous WebShop-specific parser."""

        return {
            "positive": list(self.positive_constraints),
            "negative": list(self.negative_constraints),
            "options": dict(self.options),
            "max_price": self.max_price,
            "min_rating": self.min_rating,
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
        preference = data.get("P", data.get("comparative_preference"))
        preference_provenance = data.get(
            "P_provenance",
            data.get("preference_provenance"),
        )
        if not isinstance(preference, dict):
            preference = None
        if not isinstance(preference_provenance, dict):
            preference_provenance = None
        product_type = _coerce_text(data.get("product_type")) or None
        attributes = _coerce_list(data.get("attributes"))
        options = _coerce_options(
            data.get("options", data.get("required_options"))
        )
        constraints = data.get("constraints")
        if not isinstance(constraints, Mapping):
            constraints = {}
        max_price = _coerce_optional_number(
            data.get(
                "max_price",
                constraints.get("max_price", constraints.get("price_upper")),
            )
        )
        min_rating = _coerce_optional_number(
            data.get("min_rating", constraints.get("min_rating"))
        )

        return cls(
            raw_query=raw_query,
            intent=intent or _fallback_intent(raw_query),
            positive_constraints=c_plus,
            negative_constraints=c_minus,
            product_type=product_type,
            attributes=attributes,
            options=options,
            max_price=max_price,
            min_rating=min_rating,
            comparative_preference=preference,
            preference_provenance=preference_provenance,
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


def _coerce_options(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    options: Dict[str, str] = {}
    for key, item in value.items():
        name = _coerce_text(key)
        selected = _coerce_text(item)
        if name and selected:
            options[name] = selected
    return options


def _coerce_optional_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0 or not math.isfinite(number):
        return None
    return number


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

        product_type = None
        for pattern in (
            r"\b(sneakers?|shoes?|boots?|sandals?|slippers?)\b",
            r"\b(shirts?|pants?|shorts?|jackets?|coats?|dresses?)\b",
            r"\b(bags?|backpacks?|wallets?)\b",
            r"\b(watches?|headphones?|speakers?|chargers?)\b",
        ):
            match = re.search(pattern, query, flags=re.I)
            if match:
                product_type = match.group(1).lower()
                break
        price_match = re.search(
            r"(?:price\s+lower\s+than|under|below|less\s+than|at\s+most)"
            r"\s*\$?\s*(\d+(?:\.\d+)?)",
            query,
            flags=re.I,
        )
        options = {
            match.group(1).strip(): match.group(2).strip()
            for match in re.finditer(
                r"\b([A-Za-z][A-Za-z0-9 _/-]{0,30})\s*:\s*([^,.;]+)",
                query,
            )
        }

        return GoalContract(
            raw_query=query,
            intent=intent,
            positive_constraints=positive_constraints,
            negative_constraints=negative_constraints,
            product_type=product_type,
            attributes=list(positive_constraints),
            options=options,
            max_price=(
                float(price_match.group(1)) if price_match is not None else None
            ),
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
        self.last_usage = LLMUsage(usage_reported=False)
        self.last_api_call_count = 0
        self.last_accounting_error: Optional[str] = None
        self.fallback_extractor = (
            fallback_extractor
            or fallback_parser
            or RegexGoalContractExtractor()
        )

    def extract(self, query: str) -> GoalContract:
        query = query or ""
        self.last_actual_model = None
        self.last_usage = LLMUsage(usage_reported=False)
        self.last_api_call_count = 0
        self.last_accounting_error = None
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
- product_type: the catalog-style singular head noun explicitly requested.
- attributes: explicit descriptive requirements, as short phrases in their
  original order. Do not include the product type, option values, price, or a
  comparative preference.
- options: explicit ``name: selected value`` product options.
- max_price: the numeric upper price bound, or null.
- min_rating: a mandatory numeric rating floor, or null.

Rules:
- Use only information explicitly present in the user query.
- Do not infer hidden preferences.
- Preserve concrete product attributes, brands, prices, sizes, colors, ratings, materials, and action constraints as short strings.
- Comparative phrases such as "choose the cheapest" or "choose the highest
  rated" are handled elsewhere and must not enter hard constraints.
- Return JSON only with exactly these keys: I, C_plus, C_minus, product_type,
  attributes, options, max_price, min_rating.
""".strip()

        try:
            client = OpenAI(timeout=self.timeout)
            self.last_api_call_count = 1
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                service_tier="default",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            if isinstance(response, Mapping):
                response_model = response.get("model")
            else:
                response_model = getattr(response, "model", None)
            self.last_actual_model = str(response_model or self.model)
            try:
                self.last_usage = LLMUsage.from_response(response)
            except ValueError as exc:
                # Provider accounting metadata must never change the parser's
                # contract-extraction behavior.
                self.last_accounting_error = f"{type(exc).__name__}: {exc}"
                self.last_usage = LLMUsage(
                    model=self.last_actual_model,
                    usage_reported=False,
                )
            choices = (
                response.get("choices")
                if isinstance(response, Mapping)
                else response.choices
            )
            choice = choices[0]
            message = (
                choice.get("message")
                if isinstance(choice, Mapping)
                else choice.message
            )
            content = (
                message.get("content")
                if isinstance(message, Mapping)
                else message.content
            ) or "{}"
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
        pricing: Optional[LLMPricing] = None,
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
        if pricing is not None and not isinstance(pricing, LLMPricing):
            raise TypeError("pricing must be an LLMPricing instance or None")
        self.pricing = pricing
        self.parser_requests = 0
        self.parser_calls = 0
        self.parser_api_calls = 0
        self.parser_cache_hits = 0
        self.parser_fallback_count = 0
        self.parser_error_count = 0
        self.parser_usage_reported_call_count = 0
        self.parser_usage_missing_call_count = 0
        self._parser_usage = LLMUsage()
        self._actual_parser_models: List[str] = []
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

        if (
            not isinstance(cache, dict)
            or cache.get("version") != GOAL_CONTRACT_SCHEMA_VERSION
        ):
            return None

        entry = cache.get("entries", {}).get(key)
        if not isinstance(entry, dict):
            return None
        if (
            entry.get("schema_version") != GOAL_CONTRACT_SCHEMA_VERSION
            or entry.get("instruction") != query
            or entry.get("parser_model") != self.openai_model
        ):
            return None

        data = entry.get("contract")
        if not isinstance(data, dict):
            return None
        self.last_actual_parser_model = (
            _coerce_text(entry.get("actual_parser_model")) or self.openai_model
        )
        return GoalContract.from_dict(
            data,
            raw_query=query,
            extractor=(
                _coerce_text(data.get("extractor", data.get("parser")))
                or "openai_goal_contract"
            ),
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
                cache = {
                    "version": GOAL_CONTRACT_SCHEMA_VERSION,
                    "entries": {},
                }

            if (
                not isinstance(cache, dict)
                or cache.get("version") != GOAL_CONTRACT_SCHEMA_VERSION
            ):
                cache = {
                    "version": GOAL_CONTRACT_SCHEMA_VERSION,
                    "entries": {},
                }
            entries = cache.setdefault("entries", {})
            if not isinstance(entries, dict):
                entries = {}
                cache["entries"] = entries
            entries[key] = {
                "schema_version": GOAL_CONTRACT_SCHEMA_VERSION,
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
        self.parser_requests += 1
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
        # Reset these explicitly so test doubles or custom extractors cannot
        # accidentally replay accounting metadata from a prior invocation.
        self.openai_extractor.last_api_call_count = 0
        self.openai_extractor.last_usage = LLMUsage(usage_reported=False)
        self.openai_extractor.last_accounting_error = None
        try:
            contract = self.openai_extractor.extract(query)
        finally:
            self._record_last_provider_call()
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

    def _record_last_provider_call(self) -> None:
        """Add accounting from a real provider call, never from a cache hit."""

        api_calls = self.openai_extractor.last_api_call_count
        if (
            isinstance(api_calls, bool)
            or not isinstance(api_calls, int)
            or api_calls < 0
        ):
            api_calls = 0
        self.parser_api_calls += api_calls
        if api_calls == 0:
            return

        usage = self.openai_extractor.last_usage
        if not isinstance(usage, LLMUsage):
            usage = LLMUsage(usage_reported=False)
        if usage.usage_reported:
            self.parser_usage_reported_call_count += api_calls
            self._parser_usage = self._parser_usage + usage
        else:
            self.parser_usage_missing_call_count += api_calls

        actual_model = usage.model or self.openai_extractor.last_actual_model
        if actual_model:
            actual_model = str(actual_model)
            if actual_model not in self._actual_parser_models:
                self._actual_parser_models.append(actual_model)

    def _estimated_cost_usd(self) -> Optional[float]:
        if self.pricing is None or not self.pricing.is_configured:
            return None
        if self.parser_usage_missing_call_count:
            return None
        return self._parser_usage.estimated_cost_usd(self.pricing)

    def stats_dict(self) -> Dict[str, Any]:
        usage = self._parser_usage
        known_cost = usage.estimated_cost_usd(self.pricing)
        return {
            "requested_parser_model": self.openai_model if self.use_openai else None,
            "actual_parser_model": self.last_actual_parser_model,
            "actual_parser_models": list(self._actual_parser_models),
            "parser_requests": self.parser_requests,
            "parser_request_count": self.parser_requests,
            "parser_calls": self.parser_calls,
            "parser_api_calls": self.parser_api_calls,
            "parser_uncached_api_call_count": self.parser_api_calls,
            "parser_cache_hits": self.parser_cache_hits,
            "parser_fallback_count": self.parser_fallback_count,
            "parser_error_count": self.parser_error_count,
            "parser_usage_reported_call_count": (
                self.parser_usage_reported_call_count
            ),
            "parser_usage_missing_call_count": (
                self.parser_usage_missing_call_count
            ),
            "parser_input_tokens": usage.input_tokens,
            "parser_output_tokens": usage.output_tokens,
            "parser_total_tokens": usage.total_tokens,
            "parser_cached_input_tokens": usage.cached_input_tokens,
            "parser_reasoning_tokens": usage.reasoning_tokens,
            "parser_estimated_cost_usd": self._estimated_cost_usd(),
            "parser_known_estimated_cost_usd": known_cost,
            "parser_pricing": (
                self.pricing.to_dict() if self.pricing is not None else None
            ),
            "require_parser_success": self.require_success,
            "goal_contract_cache_path": str(self.cache_path) if self.cache_path else None,
        }


# Backward-compatible names for the previous Gate integration.
StructuredGoal = GoalContract
RegexGoalParser = RegexGoalContractExtractor
OpenAIGoalParser = OpenAIGoalContractExtractor
