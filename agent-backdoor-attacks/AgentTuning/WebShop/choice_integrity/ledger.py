"""Public, evidence-backed candidate ledger for choice-integrity decisions.

The ledger accepts :class:`schema.Candidate` objects only.  It intentionally
has no API that accepts a ``ChoiceTask`` and never touches task-level winner or
attacker-target labels.  Candidate metadata is also discarded on ingestion;
only the public product fields and sanitized evidence are retained.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from .schema import Candidate, FrozenDict, Preference, PreferenceKind, SchemaError


class LedgerError(ValueError):
    """Base error for invalid or contradictory ledger operations."""


class LedgerIncompleteError(LedgerError):
    """Raised when a choice cannot be certified from the available evidence."""

    def __init__(self, message: str, missing: Optional[Dict[str, Tuple[str, ...]]] = None):
        super().__init__(message)
        self.missing = missing or {}


_LABEL_KEY_PARTS = (
    "winner",
    "attacker",
    "attack_target",
    "target",
    "target_label",
    "ground_truth",
)

_RETAINED_EVIDENCE_FIELDS = {
    "asin",
    "options",
    "feasible",
    "hard_feasible",
    "price",
    "rating",
    "brand",
    "title",
    "attributes",
    "product_type",
    "availability",
    "features",
    "description",
    "constraints",
    "retrieval",
    "observations",
    "page",
    "shortlist_rank",
    "source",
    "location",
    "url",
    "text",
    "value",
}

_PUBLIC_SCALAR_UPDATE_FIELDS = {
    "price",
    "rating",
    "brand",
    "title",
}

_PUBLIC_OBSERVATION_FIELDS = _RETAINED_EVIDENCE_FIELDS - {
    # Identity and comparison-set membership are fixed when the ledger is
    # seeded.  A policy-page observation may corroborate them in its evidence
    # payload, but it cannot claim to have supplied or changed them.
    "asin",
    "options",
    "feasible",
    "hard_feasible",
    "page",
    "shortlist_rank",
}


def _is_hidden_label_key(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(part in normalized for part in _LABEL_KEY_PARTS)


def _sanitize_json(value: Any, *, evidence_root: bool = False) -> Any:
    """Copy JSON-like evidence while dropping evaluator-label-shaped fields."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_hidden_label_key(key_text):
                continue
            if evidence_root and key_text not in _RETAINED_EVIDENCE_FIELDS:
                # Unknown top-level evidence cannot contribute to completeness.
                # Keeping it would make it too easy to smuggle evaluator labels
                # through a novel field name.
                continue
            result[key_text] = _sanitize_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    return str(value)


def _evidence_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, FrozenDict):
        value = dict(value)
    if not isinstance(value, Mapping):
        return {}
    return _sanitize_json(value, evidence_root=True)


def _required_comparison_fields(preference: Preference) -> Tuple[str, ...]:
    if not isinstance(preference, Preference):
        raise TypeError("preference must be a schema.Preference")
    if preference.kind == PreferenceKind.PRICE_MIN:
        return ("price",)
    if preference.kind == PreferenceKind.RATING_MAX_PRICE_TIEBREAK:
        return ("rating", "price")
    raise LedgerError("unsupported preference kind: {!r}".format(preference.kind))


def _public_scalar_value(field_name: str, value: Any) -> Any:
    """Validate a scalar copied from a rendered public product page."""

    if field_name not in _PUBLIC_SCALAR_UPDATE_FIELDS:
        raise LedgerError(
            "public observations cannot update candidate field {!r}".format(
                field_name
            )
        )
    if value is None:
        return None
    if field_name in {"price", "rating"}:
        if type(value) not in (int, float):
            raise LedgerError(
                "public {} observation must be numeric".format(field_name)
            )
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise LedgerError(
                "public {} observation is out of range".format(field_name)
            )
        if field_name == "rating" and number > 5:
            raise LedgerError("public rating observation is out of range")
        return number
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(
            "public {} observation must be a non-empty string".format(
                field_name
            )
        )
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class EvidenceObservation:
    """One public observation that supplied candidate fields."""

    source: str
    fields: Tuple[str, ...]
    evidence: FrozenDict = field(default_factory=FrozenDict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "fields": list(self.fields),
            "evidence": _sanitize_json(dict(self.evidence)),
        }


@dataclass
class LedgerEntry:
    """Merged public state for one ASIN-plus-options identity."""

    candidate_id: str
    asin: str
    options: Tuple[Tuple[str, str], ...]
    feasible: Optional[bool] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    title: Optional[str] = None
    shortlist_rank: Optional[int] = None
    page: Optional[int] = None
    observations: List[EvidenceObservation] = field(default_factory=list)
    sources: Set[str] = field(default_factory=set)
    ambiguous_fields: Set[str] = field(default_factory=set)

    @classmethod
    def from_candidate(
        cls,
        candidate: Candidate,
        source: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> "LedgerEntry":
        if not isinstance(candidate, Candidate):
            raise TypeError("CandidateLedger accepts schema.Candidate objects only")
        entry = cls(
            candidate_id=candidate.candidate_id,
            asin=candidate.asin,
            options=candidate.options,
        )
        entry.merge(candidate, source=source, evidence=evidence)
        return entry

    def merge(
        self,
        candidate: Candidate,
        source: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(candidate, Candidate):
            raise TypeError("ledger entries can merge schema.Candidate objects only")
        if candidate.candidate_id != self.candidate_id:
            raise LedgerError("cannot merge two different product-option identities")
        source = _validate_source(source)

        observed_fields = ["asin", "options", "feasible"]
        values = {
            "feasible": candidate.feasible,
            "price": candidate.price,
            "rating": candidate.rating,
            "brand": candidate.brand,
            "title": candidate.title,
            "shortlist_rank": candidate.shortlist_rank,
            "page": candidate.page,
        }
        for name, incoming in values.items():
            if incoming is not None:
                observed_fields.append(name)
            current = getattr(self, name)
            if current is None:
                setattr(self, name, incoming)
            elif incoming is not None and current != incoming:
                self.ambiguous_fields.add(name)

        public_evidence = _evidence_dict(
            candidate.evidence if evidence is None else evidence
        )
        self.observations.append(
            EvidenceObservation(
                source=source,
                fields=tuple(sorted(set(observed_fields))),
                evidence=FrozenDict(public_evidence),
            )
        )
        self.sources.add(source)

    def observe_public(
        self,
        *,
        source: str,
        fields: Optional[Mapping[str, Any]] = None,
        observed_fields: Iterable[str] = (),
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append a partial public observation without changing identity.

        Only rendered scalar comparison fields may fill an empty value or
        corroborate an existing one.  Conflicts remain fail-closed.  Feasible,
        ASIN, options, page, and shortlist rank are deliberately immutable here
        so a policy trajectory cannot expand or relabel the frozen comparison
        set.
        """

        source = _validate_source(source)
        if fields is None:
            fields = {}
        if not isinstance(fields, Mapping):
            raise TypeError("public observation fields must be a mapping")

        validated: Dict[str, Any] = {}
        supplied: Set[str] = set()
        for name, raw_value in fields.items():
            field_name = str(name)
            incoming = _public_scalar_value(field_name, raw_value)
            if incoming is None:
                continue
            supplied.add(field_name)
            validated[field_name] = incoming

        for raw_name in observed_fields:
            field_name = str(raw_name)
            if field_name not in _PUBLIC_OBSERVATION_FIELDS:
                raise LedgerError(
                    "field {!r} cannot be supplied by a runtime public "
                    "observation".format(field_name)
                )
            supplied.add(field_name)

        if not supplied:
            raise LedgerError("public observation did not supply any fields")
        observation = EvidenceObservation(
            source=source,
            fields=tuple(sorted(supplied)),
            evidence=FrozenDict(_evidence_dict(evidence)),
        )
        for field_name, incoming in validated.items():
            current = getattr(self, field_name)
            if current is None:
                setattr(self, field_name, incoming)
            elif current != incoming:
                self.ambiguous_fields.add(field_name)
        self.observations.append(observation)
        self.sources.add(source)

    def has_field_evidence(self, field_name: str) -> bool:
        return any(field_name in observation.fields for observation in self.observations)

    def missing_fields(
        self,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> Tuple[str, ...]:
        required = ["feasible"]
        if self.feasible is True:
            required.extend(_required_comparison_fields(preference))

        missing: List[str] = []
        for name in required:
            if name in self.ambiguous_fields:
                missing.append("{}:conflict".format(name))
                continue
            if getattr(self, name) is None:
                missing.append(name)
                continue
            if require_evidence and not self.has_field_evidence(name):
                missing.append("{}:no_evidence".format(name))
        return tuple(missing)

    def is_complete(
        self,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> bool:
        return not self.missing_fields(
            preference,
            require_evidence=require_evidence,
        )

    def to_candidate(self) -> Candidate:
        if self.feasible is None:
            raise LedgerIncompleteError(
                "candidate {} lacks a feasibility decision".format(self.candidate_id)
            )
        evidence = {
            "source": sorted(self.sources),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }
        return Candidate(
            asin=self.asin,
            options=self.options,
            feasible=self.feasible,
            price=self.price,
            rating=self.rating,
            brand=self.brand,
            title=self.title,
            evidence=evidence,
            shortlist_rank=self.shortlist_rank,
            page=self.page,
            # Candidate metadata is intentionally never retained.
            metadata={},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "asin": self.asin,
            "options": dict(self.options),
            "feasible": self.feasible,
            "price": self.price,
            "rating": self.rating,
            "brand": self.brand,
            "title": self.title,
            "shortlist_rank": self.shortlist_rank,
            "page": self.page,
            "sources": sorted(self.sources),
            "ambiguous_fields": sorted(self.ambiguous_fields),
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }


def _validate_source(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise LedgerError("candidate evidence source must be a non-empty string")
    normalized = source.strip().lower().replace("-", "_").replace(" ", "_")
    if any(part in normalized for part in _LABEL_KEY_PARTS):
        raise LedgerError("evaluator labels cannot be candidate evidence sources")
    return source.strip()


CandidateReference = Union[str, Candidate, LedgerEntry]


class CandidateLedger:
    """Deduplicated public candidate evidence for one episode.

    ``comparison_complete`` means the fixed defense-controlled shortlist has
    been fully observed.  Until that flag is set, non-dominance certification
    fails closed because an unseen candidate might be preferred.
    """

    def __init__(self, *, comparison_complete: bool = False) -> None:
        self._entries: Dict[str, LedgerEntry] = {}
        self.comparison_complete = bool(comparison_complete)

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[Candidate],
        *,
        source: str,
        comparison_complete: bool = True,
    ) -> "CandidateLedger":
        ledger = cls(comparison_complete=False)
        for candidate in candidates:
            ledger.upsert(candidate, source=source)
        ledger.comparison_complete = bool(comparison_complete)
        return ledger

    def mark_comparison_complete(self) -> None:
        self.comparison_complete = True

    def upsert(
        self,
        candidate: Candidate,
        *,
        source: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> LedgerEntry:
        if not isinstance(candidate, Candidate):
            raise TypeError("CandidateLedger accepts schema.Candidate objects only")
        source = _validate_source(source)
        existing = self._entries.get(candidate.candidate_id)
        if existing is None:
            existing = LedgerEntry.from_candidate(
                candidate,
                source=source,
                evidence=evidence,
            )
            self._entries[candidate.candidate_id] = existing
        else:
            existing.merge(candidate, source=source, evidence=evidence)
        return existing

    def _candidate_id(self, reference: CandidateReference) -> str:
        if isinstance(reference, str):
            return reference
        if isinstance(reference, Candidate):
            return reference.candidate_id
        if isinstance(reference, LedgerEntry):
            return reference.candidate_id
        raise TypeError("candidate reference must be an ID, Candidate, or LedgerEntry")

    def get(self, reference: CandidateReference) -> Optional[LedgerEntry]:
        return self._entries.get(self._candidate_id(reference))

    def require(self, reference: CandidateReference) -> LedgerEntry:
        candidate_id = self._candidate_id(reference)
        entry = self._entries.get(candidate_id)
        if entry is None:
            raise LedgerIncompleteError(
                "candidate {} is absent from the public ledger".format(candidate_id),
                missing={candidate_id: ("candidate",)},
            )
        return entry

    def observe_existing(
        self,
        reference: CandidateReference,
        *,
        source: str,
        fields: Optional[Mapping[str, Any]] = None,
        observed_fields: Iterable[str] = (),
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> LedgerEntry:
        """Update only an identity already in the frozen comparison ledger."""

        entry = self.require(reference)
        entry.observe_public(
            source=source,
            fields=fields,
            observed_fields=observed_fields,
            evidence=evidence,
        )
        return entry

    def entries_for_asin(self, asin: str) -> Tuple[LedgerEntry, ...]:
        """Return frozen product-option identities for a public ASIN."""

        if not isinstance(asin, str) or not asin.strip():
            raise LedgerError("ASIN lookup must be a non-empty string")
        wanted = asin.strip().upper()
        return tuple(
            entry
            for entry in self.entries()
            if entry.asin.strip().upper() == wanted
        )

    def entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def public_candidates(self) -> Tuple[Candidate, ...]:
        return tuple(entry.to_candidate() for entry in self.entries())

    def completeness(
        self,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> Dict[str, Tuple[str, ...]]:
        return {
            entry.candidate_id: missing
            for entry in self.entries()
            for missing in (
                entry.missing_fields(
                    preference,
                    require_evidence=require_evidence,
                ),
            )
            if missing
        }

    def require_complete(
        self,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> None:
        if not self.comparison_complete:
            raise LedgerIncompleteError(
                "the defense-controlled comparison shortlist is incomplete"
            )
        missing = self.completeness(
            preference,
            require_evidence=require_evidence,
        )
        # Only verified feasible candidates participate in preference
        # comparison.  Infeasible candidates may legitimately omit price/rating.
        relevant_missing = {
            candidate_id: fields
            for candidate_id, fields in missing.items()
            if self._entries[candidate_id].feasible is True
            or any(field.startswith("feasible") for field in fields)
        }
        if relevant_missing:
            raise LedgerIncompleteError(
                "candidate comparison evidence is incomplete",
                missing=relevant_missing,
            )

    def feasible_entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(entry for entry in self.entries() if entry.feasible is True)

    def winners(
        self,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> Tuple[LedgerEntry, ...]:
        self.require_complete(preference, require_evidence=require_evidence)
        feasible = self.feasible_entries()
        if not feasible:
            raise LedgerIncompleteError("the public ledger has no hard-feasible candidate")
        candidates = tuple(entry.to_candidate() for entry in feasible)
        try:
            winner_ids = set(preference.winner_ids(candidates))
        except SchemaError as exc:
            raise LedgerIncompleteError(str(exc)) from exc
        return tuple(
            entry for entry in feasible if entry.candidate_id in winner_ids
        )

    def dominators(
        self,
        selected: CandidateReference,
        preference: Preference,
        *,
        require_evidence: bool = True,
    ) -> Tuple[LedgerEntry, ...]:
        self.require_complete(preference, require_evidence=require_evidence)
        selected_entry = self.require(selected)
        selected_candidate = selected_entry.to_candidate()
        if not selected_candidate.feasible:
            return ()

        dominators: List[LedgerEntry] = []
        for other in self.feasible_entries():
            if other.candidate_id == selected_entry.candidate_id:
                continue
            try:
                if preference.dominates(other.to_candidate(), selected_candidate):
                    dominators.append(other)
            except SchemaError as exc:
                raise LedgerIncompleteError(str(exc)) from exc
        return tuple(sorted(dominators, key=lambda entry: entry.candidate_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_complete": self.comparison_complete,
            "candidate_count": len(self._entries),
            "entries": [entry.to_dict() for entry in self.entries()],
        }
