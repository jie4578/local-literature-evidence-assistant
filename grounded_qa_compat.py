"""Conservative compatibility normalization for grounded QA responses.

This module only handles two observed provider shape variants.  It never
rewrites evidence IDs, citations, pages, sources, sections, status values, or
claim text.  The returned metadata contains structural fingerprints only;
response values are intentionally absent from diagnostic metadata.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


NORMALIZATION_APPLIED = "APPLIED"
NORMALIZATION_NOT_NEEDED = "NOT_NEEDED"
NORMALIZATION_FAILED = "FAILED"

NORMALIZATION_CONFLICT = "NORMALIZATION_CONFLICT"
INVALID_CLAIM_ALIAS_TYPE = "INVALID_CLAIM_ALIAS_TYPE"
INVALID_LIMITATIONS_COMPAT_TYPE = "INVALID_LIMITATIONS_COMPAT_TYPE"

CLAIM_ALIAS_RULE = "claim_to_text"
LIMITATIONS_STRING_RULE = "limitations_string_to_list"


def _type_name(value: Any) -> str:
    return type(value).__name__


def safe_structural_shape(payload: Any) -> dict[str, Any]:
    """Return a value-free structural fingerprint for diagnostics."""
    if not isinstance(payload, Mapping):
        return {
            "top_level_type": _type_name(payload),
            "top_level_keys": [],
            "claim_shapes": [],
        }

    raw_claims = payload.get("claims")
    claim_shapes: list[dict[str, Any]] = []
    if isinstance(raw_claims, list):
        for index, claim in enumerate(raw_claims):
            if not isinstance(claim, Mapping):
                claim_shapes.append(
                    {
                        "claim_index": index,
                        "claim_type": _type_name(claim),
                        "claim_keys": [],
                    }
                )
                continue
            evidence_ids = claim.get("evidence_ids")
            claim_shapes.append(
                {
                    "claim_index": index,
                    "claim_type": "object",
                    "claim_keys": sorted(str(key) for key in claim.keys()),
                    "text_type": _type_name(claim.get("text")) if "text" in claim else None,
                    "text_present": "text" in claim,
                    "evidence_ids_type": _type_name(evidence_ids) if "evidence_ids" in claim else None,
                    "evidence_ids_count": len(evidence_ids) if isinstance(evidence_ids, list) else None,
                }
            )

    limitations = payload.get("limitations")
    return {
        "top_level_type": "object",
        "top_level_keys": sorted(str(key) for key in payload.keys()),
        "status_type": _type_name(payload.get("status")) if "status" in payload else None,
        "claims_type": _type_name(raw_claims) if "claims" in payload else None,
        "claims_count": len(raw_claims) if isinstance(raw_claims, list) else None,
        "limitations_type": _type_name(limitations) if "limitations" in payload else None,
        "limitations_count": len(limitations) if isinstance(limitations, list) else None,
        "claim_shapes": claim_shapes,
    }


@dataclass(frozen=True)
class CompatibilityNormalizationResult:
    """Normalized payload plus safe, value-free provenance metadata."""

    status: str
    normalized_payload: Mapping[str, Any] | None
    applied_rules: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    pre_normalization_shape: dict[str, Any] = field(default_factory=dict)
    post_normalization_shape: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != NORMALIZATION_FAILED

    def metadata(self) -> dict[str, Any]:
        """Return safe metadata without payload, claim, or limitation values."""
        return {
            "status": self.status,
            "applied_rules": list(self.applied_rules),
            "warnings": list(self.warnings),
            "conflicts": list(self.conflicts),
            "pre_normalization_shape": deepcopy(self.pre_normalization_shape),
            "post_normalization_shape": deepcopy(self.post_normalization_shape),
        }


def _failed(
    payload: Any,
    *,
    warnings: tuple[str, ...],
    conflicts: tuple[str, ...] = (),
) -> CompatibilityNormalizationResult:
    shape = safe_structural_shape(payload)
    return CompatibilityNormalizationResult(
        status=NORMALIZATION_FAILED,
        normalized_payload=None,
        warnings=warnings,
        conflicts=conflicts,
        pre_normalization_shape=shape,
        post_normalization_shape=shape,
    )


def normalize_grounded_payload(payload: Any) -> CompatibilityNormalizationResult:
    """Apply only the two approved compatibility mappings.

    The input is copied before any mapping.  A conflict or invalid compatible
    type fails the whole response; no partially normalized response is
    returned to the strict validator.
    """
    pre_shape = safe_structural_shape(payload)
    if not isinstance(payload, Mapping):
        return _failed(payload, warnings=("TOP_LEVEL_SCHEMA_ERROR",))

    normalized = deepcopy(dict(payload))
    applied_rules: list[str] = []
    errors: list[str] = []
    conflicts: list[str] = []

    if "limitations" in normalized:
        limitations = normalized["limitations"]
        if isinstance(limitations, str):
            if not limitations:
                errors.append(INVALID_LIMITATIONS_COMPAT_TYPE)
            else:
                normalized["limitations"] = [limitations]
                applied_rules.append(LIMITATIONS_STRING_RULE)
        elif isinstance(limitations, list):
            if any(not isinstance(item, str) for item in limitations):
                errors.append(INVALID_LIMITATIONS_COMPAT_TYPE)
        else:
            errors.append(INVALID_LIMITATIONS_COMPAT_TYPE)

    raw_claims = normalized.get("claims")
    if isinstance(raw_claims, list):
        normalized_claims: list[Any] = []
        for claim in raw_claims:
            if not isinstance(claim, Mapping) or "claim" not in claim:
                normalized_claims.append(claim)
                continue
            if "text" in claim:
                conflicts.append(NORMALIZATION_CONFLICT)
                normalized_claims.append(claim)
                continue
            alias_value = claim["claim"]
            if not isinstance(alias_value, str):
                errors.append(INVALID_CLAIM_ALIAS_TYPE)
                normalized_claims.append(claim)
                continue
            normalized_claim = dict(claim)
            del normalized_claim["claim"]
            normalized_claim["text"] = alias_value
            normalized_claims.append(normalized_claim)
            applied_rules.append(CLAIM_ALIAS_RULE)
        normalized["claims"] = normalized_claims

    if errors or conflicts:
        return _failed(
            payload,
            warnings=tuple(dict.fromkeys(errors)),
            conflicts=tuple(dict.fromkeys(conflicts)),
        )

    post_shape = safe_structural_shape(normalized)
    status = NORMALIZATION_APPLIED if applied_rules else NORMALIZATION_NOT_NEEDED
    return CompatibilityNormalizationResult(
        status=status,
        normalized_payload=normalized,
        applied_rules=tuple(dict.fromkeys(applied_rules)),
        pre_normalization_shape=pre_shape,
        post_normalization_shape=post_shape,
    )
