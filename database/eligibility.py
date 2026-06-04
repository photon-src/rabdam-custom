"""Eligibility checks for Bnet reference database workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral
from typing import SupportsFloat, SupportsIndex, cast


DEFAULT_MIN_TEMPERATURE_K = 80.0
DEFAULT_MAX_TEMPERATURE_K = 120.0
DEFAULT_MAX_RESOLUTION_ANGSTROM = 3.5
DEFAULT_MAX_RFREE = 0.4
DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT = 20

_Floatable = str | bytes | bytearray | SupportsFloat | SupportsIndex


class BnetEligibilityReason(str, Enum):
    """Machine-readable reasons a structure is not eligible."""

    ELIGIBLE = "eligible"
    MISSING_RESOLUTION = "missing_resolution"
    INVALID_RESOLUTION = "invalid_resolution"
    RESOLUTION_WORSE_THAN_LIMIT = "resolution_worse_than_limit"

    MISSING_RFREE = "missing_rfree"
    INVALID_RFREE = "invalid_rfree"
    RFREE_TOO_HIGH = "rfree_too_high"

    MISSING_TEMPERATURE = "cannot_verify_temperature"
    INVALID_TEMPERATURE = "invalid_temperature"
    TEMPERATURE_OUTSIDE_CRYO_RANGE = "temperature_outside_cryo_range"

    INVALID_ASP_GLU_CARBOXYL_OXYGEN_COUNT = (
        "invalid_asp_glu_carboxyl_oxygen_count"
    )
    TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS = "too_few_asp_glu_carboxyl_oxygens"
    INVALID_OCCUPANCY_FLAG = "invalid_occupancy_flag"
    ASP_GLU_OCCUPANCY_LESS_THAN_ONE = "asp_glu_occupancy_less_than_one"

    INVALID_B_FACTOR_MODEL_FLAG = "invalid_b_factor_model_flag"
    NOT_PER_ATOM_B_FACTOR_MODEL = "not_per_atom_b_factor_model"

    MISSING_BNET = "missing_bnet"
    INVALID_BNET = "invalid_bnet"


@dataclass(frozen=True, slots=True)
class BnetEligibilityIssue:
    """One eligibility issue."""

    reason: BnetEligibilityReason
    message: str
    value: object | None = None


@dataclass(frozen=True, slots=True)
class BnetEligibilityResult:
    """Result of checking Bnet reference database eligibility."""

    is_eligible: bool
    issues: tuple[BnetEligibilityIssue, ...] = field(default_factory=tuple)

    @property
    def primary_reason(self) -> BnetEligibilityReason:
        """Return the first failure reason, or ELIGIBLE."""

        if not self.issues:
            return BnetEligibilityReason.ELIGIBLE
        return self.issues[0].reason

    @property
    def primary_message(self) -> str:
        """Return the first failure message, or a success message."""

        if not self.issues:
            return "Structure is eligible for Bnet reference database inclusion."
        return self.issues[0].message


@dataclass(frozen=True, slots=True)
class BnetEligibilityContext:
    """Inputs needed to assess Bnet reference database eligibility.

    This object deliberately avoids any PDB-REDO-specific fields. Database
    builders can construct it from PDB-REDO metadata, mmCIF metadata, or RABDAM
    workflow outputs.
    """

    resolution_angstrom: float | None
    r_free: float | None
    temperature_k: float | Sequence[float] | None
    asp_glu_carboxyl_oxygen_count: int
    has_asp_glu_residue_with_total_occupancy_below_one: bool
    uses_per_atom_b_factors: bool
    bnet: float | None = None


def check_bnet_reference_eligibility(
    context: BnetEligibilityContext,
    *,
    require_bnet: bool = True,
    min_temperature_k: float = DEFAULT_MIN_TEMPERATURE_K,
    max_temperature_k: float = DEFAULT_MAX_TEMPERATURE_K,
    max_resolution_angstrom: float = DEFAULT_MAX_RESOLUTION_ANGSTROM,
    max_r_free: float = DEFAULT_MAX_RFREE,
    min_asp_glu_carboxyl_oxygen_count: int = (
        DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT
    ),
) -> BnetEligibilityResult:
    """Check whether a structure is eligible for Bnet reference database inclusion."""

    issues: list[BnetEligibilityIssue] = []

    resolution = context.resolution_angstrom
    if resolution is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_RESOLUTION,
                "Resolution is missing.",
            )
        )
    elif not _is_finite_number(resolution) or resolution <= 0.0:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_RESOLUTION,
                "Resolution must be a finite positive number.",
                resolution,
            )
        )
    elif resolution > max_resolution_angstrom:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.RESOLUTION_WORSE_THAN_LIMIT,
                (
                    f"Resolution is {resolution:.3g} Å, which is worse than the "
                    f"{max_resolution_angstrom:.3g} Å Bnet reference database limit."
                ),
                resolution,
            )
        )

    r_free = context.r_free
    if r_free is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_RFREE,
                "Rfree is missing.",
            )
        )
    elif not _is_finite_number(r_free) or r_free < 0.0:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_RFREE,
                "Rfree must be a finite non-negative number.",
                r_free,
            )
        )
    elif r_free >= max_r_free:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.RFREE_TOO_HIGH,
                (
                    f"Rfree is {r_free:.3g}, which does not satisfy the "
                    f"strict Rfree < {max_r_free:.3g} Bnet reference "
                    "database limit."
                ),
                r_free,
            )
        )

    temperature_values = _temperature_values(context.temperature_k)
    if temperature_values is None:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_TEMPERATURE,
                "Collection temperature must contain only finite positive numbers.",
                context.temperature_k,
            )
        )
    elif not temperature_values:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.MISSING_TEMPERATURE,
                "Collection temperature could not be verified.",
            )
        )
    elif any(
        temperature < min_temperature_k or temperature > max_temperature_k
        for temperature in temperature_values
    ):
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.TEMPERATURE_OUTSIDE_CRYO_RANGE,
                (
                    "At least one collection temperature is outside the "
                    f"{min_temperature_k:.3g}–{max_temperature_k:.3g} K "
                    "Bnet reference database range: "
                    f"{_format_temperature_values(temperature_values)} K."
                ),
                temperature_values,
            )
        )

    asp_glu_carboxyl_oxygen_count = context.asp_glu_carboxyl_oxygen_count
    if (
        isinstance(asp_glu_carboxyl_oxygen_count, bool)
        or not isinstance(asp_glu_carboxyl_oxygen_count, Integral)
        or asp_glu_carboxyl_oxygen_count < 0
    ):
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_ASP_GLU_CARBOXYL_OXYGEN_COUNT,
                (
                    "Asp/Glu side-chain carboxyl oxygen count must be a "
                    "non-negative integer."
                ),
                asp_glu_carboxyl_oxygen_count,
            )
        )
    elif asp_glu_carboxyl_oxygen_count < min_asp_glu_carboxyl_oxygen_count:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS,
                (
                    "Too few Asp/Glu side-chain carboxyl oxygen atoms for "
                    "Bnet reference database inclusion: "
                    f"{asp_glu_carboxyl_oxygen_count} found, "
                    f"{min_asp_glu_carboxyl_oxygen_count} required."
                ),
                asp_glu_carboxyl_oxygen_count,
            )
        )

    has_low_occupancy = context.has_asp_glu_residue_with_total_occupancy_below_one
    if not isinstance(has_low_occupancy, bool):
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_OCCUPANCY_FLAG,
                (
                    "Asp/Glu side-chain occupancy flag must be a boolean "
                    f"value, got {has_low_occupancy!r}."
                ),
                has_low_occupancy,
            )
        )
    elif has_low_occupancy:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.ASP_GLU_OCCUPANCY_LESS_THAN_ONE,
                (
                    "At least one Asp/Glu residue has total side-chain occupancy "
                    "below one across listed conformers."
                ),
            )
        )

    uses_per_atom_b_factors = context.uses_per_atom_b_factors
    if not isinstance(uses_per_atom_b_factors, bool):
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.INVALID_B_FACTOR_MODEL_FLAG,
                (
                    "Per-atom B-factor model flag must be a boolean value, "
                    f"got {uses_per_atom_b_factors!r}."
                ),
                uses_per_atom_b_factors,
            )
        )
    elif not uses_per_atom_b_factors:
        issues.append(
            BnetEligibilityIssue(
                BnetEligibilityReason.NOT_PER_ATOM_B_FACTOR_MODEL,
                "Structure does not appear to use a per-atom B-factor model.",
            )
        )

    if require_bnet:
        bnet = context.bnet
        if bnet is None:
            issues.append(
                BnetEligibilityIssue(
                    BnetEligibilityReason.MISSING_BNET,
                    "Raw Bnet value is missing.",
                )
            )
        elif not _is_finite_number(bnet) or bnet < 0.0:
            issues.append(
                BnetEligibilityIssue(
                    BnetEligibilityReason.INVALID_BNET,
                    "Raw Bnet must be a finite non-negative number.",
                    bnet,
                )
            )

    return BnetEligibilityResult(
        is_eligible=not issues,
        issues=tuple(issues),
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False

    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        return False

    try:
        number = float(cast(_Floatable, value))
    except (TypeError, ValueError):
        return False

    return math.isfinite(number)


def _temperature_values(value: object) -> tuple[float, ...] | None:
    if value is None:
        return ()

    if isinstance(value, bool):
        return None

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        values: list[float] = []
        for item in value:
            if isinstance(item, bool) or not _is_finite_number(item):
                return None
            number = float(cast(_Floatable, item))
            if number <= 0.0:
                return None
            values.append(number)
        return tuple(values)

    if not _is_finite_number(value):
        return None

    number = float(cast(_Floatable, value))
    if number <= 0.0:
        return None
    return (number,)


def _format_temperature_values(values: tuple[float, ...]) -> str:
    return ", ".join(f"{value:.3g}" for value in values)


__all__ = [
    "BnetEligibilityContext",
    "BnetEligibilityIssue",
    "BnetEligibilityReason",
    "BnetEligibilityResult",
    "DEFAULT_MAX_RESOLUTION_ANGSTROM",
    "DEFAULT_MAX_RFREE",
    "DEFAULT_MAX_TEMPERATURE_K",
    "DEFAULT_MIN_ASP_GLU_CARBOXYL_OXYGEN_COUNT",
    "DEFAULT_MIN_TEMPERATURE_K",
    "check_bnet_reference_eligibility",
]
