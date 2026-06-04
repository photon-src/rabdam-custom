"""Process one PDB-REDO candidate for Bnet reference-database construction.

This module is the one-entry processing layer. It combines:

1. PDB-REDO metadata extraction.
2. Structural checks from the final mmCIF.
3. Cheap prefilter eligibility.
4. RABDAM BDamage calculation.
5. Raw protein Bnet calculation.
6. Final eligibility.
7. Accepted/rejected row construction.

It deliberately processes one candidate at a time. Batch orchestration,
multiprocessing, resume logic, and output happens in build.py/outputs.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

from bnet.calculate import (
    ProteinBnetCalculationError,
    ProteinBnetResult,
    calculate_protein_bnet,
)
from input.reader import StructureReadError, read_structure
from input.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    BDamageWorkflowResult,
    calculate_bdamage_for_structure_data,
)
from structure.models import StructurePreparationOptions

from .checks import (
    PdbRedoStructureCheckError,
    PdbRedoStructureChecks,
    read_pdb_redo_structure_checks,
)
from .discover import PdbRedoCandidate
from .eligibility import (
    BnetEligibilityContext,
    BnetEligibilityResult,
    check_bnet_reference_eligibility,
)
from .metadata import (
    PdbRedoMetadata,
    PdbRedoMetadataError,
    TemperatureCacheEntry,
    read_pdb_redo_metadata,
)


class PdbRedoProcessStage(str, Enum):
    """Processing stage at which a candidate was accepted or rejected."""

    METADATA = "metadata"
    STRUCTURE_CHECKS = "structure_checks"
    DOMAIN_FILTER = "domain_filter"
    PREFILTER_ELIGIBILITY = "prefilter_eligibility"
    RABDAM = "rabdam"
    FINAL_ELIGIBILITY = "final_eligibility"
    UNEXPECTED_ERROR = "unexpected_error"
    ACCEPTED = "accepted"


class PdbRedoRejectReason(str, Enum):
    """Machine-readable process-level rejection reasons."""

    METADATA_ERROR = "metadata_error"
    STRUCTURE_CHECK_ERROR = "structure_check_error"
    NOT_XRAY = "not_xray"
    MULTIPLE_MODELS = "multiple_models"
    NO_PROTEIN = "no_protein"
    HAS_NUCLEIC_ACID = "has_nucleic_acid"
    RABDAM_ERROR = "rabdam_error"
    UNEXPECTED_WORKER_ERROR = "unexpected_worker_error"


PER_ATOM_B_FACTOR_REFINEMENT_TYPES = frozenset({"ISOT", "ANISOT"})
NON_PER_ATOM_B_FACTOR_REFINEMENT_TYPES = frozenset({"OVER"})
STRUCTURAL_B_FACTOR_MODEL_CHECK_SOURCE = "structure_backbone_b_factor_check"


@dataclass(frozen=True, slots=True)
class AcceptedBnetReferenceRow:
    """Accepted row for the Bnet reference database."""

    pdb_id: str
    resolution_angstrom: float
    bnet: float

    r_work: float | None
    r_free: float | None
    temperature_k: float | None
    wilson_b: float | None
    b_factor_restraint_weight: float | None

    bnet_site_count: int
    asp_glu_carboxyl_oxygen_count: int
    asp_glu_residue_count: int

    median_bdamage: float
    left_area: float
    right_area: float

    atom_count: int
    non_hydrogen_atom_count: int
    protein_atom_count: int
    selected_atom_count: int
    bdamage_window_size: int

    has_protein: bool
    has_nucleic_acid: bool
    is_xray: bool
    has_nonflat_protein_b_factors: bool
    has_asp_glu_residue_with_total_occupancy_below_one: bool

    experimental_methods: tuple[str, ...]
    uses_per_atom_b_factors: bool = True
    b_factor_model_source: str | None = None
    b_factor_refinement_type: str | None = None
    b_factor_refinement_type_source: str | None = None
    metadata_warnings: tuple[str, ...] = field(default_factory=tuple)
    structure_check_warnings: tuple[str, ...] = field(default_factory=tuple)

    resolution_source: str | None = None
    r_work_source: str | None = None
    r_free_source: str | None = None
    temperature_source: str | None = None
    temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    growth_temperature_k: float | None = None
    growth_temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    growth_temperature_source: str | None = None
    growth_temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    temperature_cache_status: str | None = None
    temperature_cache_message: str | None = None
    wilson_b_source: str | None = None
    b_factor_restraint_weight_source: str | None = None

    final_cif_path: Path | None = None
    data_json_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RejectedBnetReferenceRow:
    """Rejected row for the Bnet reference-database build log."""

    pdb_id: str
    stage: PdbRedoProcessStage
    reason: str
    message: str

    final_cif_path: Path
    data_json_path: Path | None = None

    exception_type: str | None = None
    traceback_text: str | None = None

    resolution_angstrom: float | None = None
    r_free: float | None = None
    temperature_k: float | None = None
    asp_glu_carboxyl_oxygen_count: int | None = None
    bnet: float | None = None

    metadata_warnings: tuple[str, ...] = field(default_factory=tuple)
    structure_check_warnings: tuple[str, ...] = field(default_factory=tuple)
    temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    temperature_source: str | None = None
    growth_temperature_k: float | None = None
    growth_temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    growth_temperature_source: str | None = None
    growth_temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    temperature_cache_status: str | None = None
    temperature_cache_message: str | None = None
    r_work: float | None = None
    wilson_b: float | None = None
    b_factor_restraint_weight: float | None = None
    b_factor_refinement_type: str | None = None
    r_work_source: str | None = None
    r_free_source: str | None = None
    resolution_source: str | None = None
    wilson_b_source: str | None = None
    b_factor_restraint_weight_source: str | None = None
    b_factor_refinement_type_source: str | None = None
    uses_per_atom_b_factors: bool | None = None
    b_factor_model_source: str | None = None


@dataclass(frozen=True, slots=True)
class PdbRedoProcessResult:
    """Accepted-or-rejected result for one PDB-REDO candidate."""

    pdb_id: str
    accepted: AcceptedBnetReferenceRow | None = None
    rejected: RejectedBnetReferenceRow | None = None

    @property
    def is_accepted(self) -> bool:
        """Return whether the candidate was accepted."""

        return self.accepted is not None

    @property
    def is_rejected(self) -> bool:
        """Return whether the candidate was rejected."""

        return self.rejected is not None

    def __post_init__(self) -> None:
        if (self.accepted is None) == (self.rejected is None):
            raise ValueError(
                "PdbRedoProcessResult must contain exactly one of "
                "accepted or rejected."
            )


@dataclass(frozen=True, slots=True)
class _BFactorModelDecision:
    uses_per_atom_b_factors: bool
    source: str


def process_pdb_redo_candidate(
    candidate: PdbRedoCandidate,
    *,
    workflow_options: BDamageWorkflowOptions | None = None,
    preparation_options: StructurePreparationOptions | None = None,
    temperature_cache: Mapping[str, TemperatureCacheEntry] | None = None,
    fetch_rcsb_temperature: bool = False,
    attempt_bnet_for_reference_ineligible: bool = False,
    require_xray: bool = True,
    require_single_model: bool = True,
    require_protein: bool = True,
    reject_nucleic_acid: bool = False,
    include_traceback: bool = False,
) -> PdbRedoProcessResult:
    """Process one PDB-REDO candidate into an accepted or rejected result."""

    try:
        metadata = read_pdb_redo_metadata(
            candidate,
            temperature_cache=temperature_cache,
            fetch_rcsb_temperature=fetch_rcsb_temperature,
        )
    except PdbRedoMetadataError as error:
        return _rejected_result_from_exception(
            candidate,
            stage=PdbRedoProcessStage.METADATA,
            reason=PdbRedoRejectReason.METADATA_ERROR.value,
            message=str(error),
            error=error,
            include_traceback=include_traceback,
        )

    try:
        checks = read_pdb_redo_structure_checks(candidate)
    except PdbRedoStructureCheckError as error:
        return _rejected_result_from_exception(
            candidate,
            stage=PdbRedoProcessStage.STRUCTURE_CHECKS,
            reason=PdbRedoRejectReason.STRUCTURE_CHECK_ERROR.value,
            message=str(error),
            error=error,
            metadata=metadata,
            include_traceback=include_traceback,
        )

    domain_rejection = _domain_filter_rejection(
        candidate=candidate,
        metadata=metadata,
        checks=checks,
        require_xray=require_xray and not attempt_bnet_for_reference_ineligible,
        require_single_model=require_single_model,
        require_protein=require_protein,
        reject_nucleic_acid=(
            reject_nucleic_acid and not attempt_bnet_for_reference_ineligible
        ),
    )
    if domain_rejection is not None:
        return PdbRedoProcessResult(
            pdb_id=candidate.pdb_id,
            rejected=domain_rejection,
        )

    prefilter = _check_prefilter_eligibility(metadata=metadata, checks=checks)
    if not prefilter.is_eligible and not attempt_bnet_for_reference_ineligible:
        return PdbRedoProcessResult(
            pdb_id=candidate.pdb_id,
            rejected=_rejected_from_eligibility(
                candidate=candidate,
                metadata=metadata,
                checks=checks,
                eligibility=prefilter,
                stage=PdbRedoProcessStage.PREFILTER_ELIGIBILITY,
                bnet=None,
            ),
        )

    try:
        workflow_result, bnet_result = _calculate_rabdam_bnet(
            candidate,
            workflow_options=workflow_options,
            preparation_options=preparation_options,
        )
    except (
        StructureReadError,
        BDamageWorkflowError,
        ProteinBnetCalculationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        return _rejected_result_from_exception(
            candidate,
            stage=PdbRedoProcessStage.RABDAM,
            reason=PdbRedoRejectReason.RABDAM_ERROR.value,
            message=str(error),
            error=error,
            metadata=metadata,
            checks=checks,
            include_traceback=include_traceback,
        )

    reference_domain_rejection = _domain_filter_rejection(
        candidate=candidate,
        metadata=metadata,
        checks=checks,
        require_xray=require_xray,
        require_single_model=require_single_model,
        require_protein=require_protein,
        reject_nucleic_acid=reject_nucleic_acid,
        stage=PdbRedoProcessStage.FINAL_ELIGIBILITY,
        bnet=bnet_result.bnet,
    )
    if reference_domain_rejection is not None:
        return PdbRedoProcessResult(
            pdb_id=candidate.pdb_id,
            rejected=reference_domain_rejection,
        )

    final_eligibility = _check_final_eligibility(
        metadata=metadata,
        checks=checks,
        bnet_result=bnet_result,
    )
    if not final_eligibility.is_eligible:
        return PdbRedoProcessResult(
            pdb_id=candidate.pdb_id,
            rejected=_rejected_from_eligibility(
                candidate=candidate,
                metadata=metadata,
                checks=checks,
                eligibility=final_eligibility,
                stage=PdbRedoProcessStage.FINAL_ELIGIBILITY,
                bnet=bnet_result.bnet,
            ),
        )

    return PdbRedoProcessResult(
        pdb_id=candidate.pdb_id,
        accepted=_accepted_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            workflow_result=workflow_result,
            bnet_result=bnet_result,
        ),
    )


def _calculate_rabdam_bnet(
    candidate: PdbRedoCandidate,
    *,
    workflow_options: BDamageWorkflowOptions | None,
    preparation_options: StructurePreparationOptions | None,
) -> tuple[BDamageWorkflowResult, ProteinBnetResult]:
    resolved_input = ResolvedStructureInput(
        original_input=str(candidate.final_cif_path),
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=candidate.final_cif_path,
        structure_id=candidate.pdb_id,
    )

    structure_data = read_structure(resolved_input)

    workflow_result = calculate_bdamage_for_structure_data(
        structure_data,
        workflow_options=workflow_options,
        preparation_options=preparation_options,
    )

    bnet_result = calculate_protein_bnet(
        prepared_structure=workflow_result.prepared_structure,
        bdamage_score_result=workflow_result.bdamage_score_result,
    )

    return workflow_result, bnet_result


def _check_prefilter_eligibility(
    *,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
) -> BnetEligibilityResult:
    b_factor_model = _b_factor_model_decision(metadata=metadata, checks=checks)
    context = BnetEligibilityContext(
        resolution_angstrom=metadata.resolution_angstrom,
        r_free=metadata.r_free,
        temperature_k=metadata.temperature_values_k or metadata.temperature_k,
        asp_glu_carboxyl_oxygen_count=checks.asp_glu_carboxyl_oxygen_count,
        has_asp_glu_residue_with_total_occupancy_below_one=(
            checks.has_asp_glu_residue_with_total_occupancy_below_one
        ),
        uses_per_atom_b_factors=b_factor_model.uses_per_atom_b_factors,
        bnet=None,
    )

    return check_bnet_reference_eligibility(
        context,
        require_bnet=False,
    )


def _check_final_eligibility(
    *,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
    bnet_result: ProteinBnetResult,
) -> BnetEligibilityResult:
    b_factor_model = _b_factor_model_decision(metadata=metadata, checks=checks)
    context = BnetEligibilityContext(
        resolution_angstrom=metadata.resolution_angstrom,
        r_free=metadata.r_free,
        temperature_k=metadata.temperature_values_k or metadata.temperature_k,
        asp_glu_carboxyl_oxygen_count=bnet_result.site_count,
        has_asp_glu_residue_with_total_occupancy_below_one=(
            checks.has_asp_glu_residue_with_total_occupancy_below_one
        ),
        uses_per_atom_b_factors=b_factor_model.uses_per_atom_b_factors,
        bnet=bnet_result.bnet,
    )

    return check_bnet_reference_eligibility(
        context,
        require_bnet=True,
    )


def _domain_filter_rejection(
    *,
    candidate: PdbRedoCandidate,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
    require_xray: bool,
    require_single_model: bool,
    require_protein: bool,
    reject_nucleic_acid: bool,
    stage: PdbRedoProcessStage = PdbRedoProcessStage.DOMAIN_FILTER,
    bnet: float | None = None,
) -> RejectedBnetReferenceRow | None:
    if require_single_model and checks.model_count != 1:
        return _rejected_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            stage=stage,
            reason=PdbRedoRejectReason.MULTIPLE_MODELS.value,
            message=(
                "Entry contains multiple models; PDB-REDO database processing "
                f"expects one model, found {checks.model_count}."
            ),
            bnet=bnet,
        )

    if require_xray and not checks.is_xray:
        return _rejected_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            stage=stage,
            reason=PdbRedoRejectReason.NOT_XRAY.value,
            message=(
                "Entry is not marked as X-ray crystallography. "
                f"Experimental methods: {checks.experimental_methods!r}."
            ),
            bnet=bnet,
        )

    if require_protein and not checks.has_protein:
        return _rejected_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            stage=stage,
            reason=PdbRedoRejectReason.NO_PROTEIN.value,
            message="Entry does not contain a protein polymer.",
            bnet=bnet,
        )

    if reject_nucleic_acid and checks.has_nucleic_acid:
        return _rejected_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            stage=stage,
            reason=PdbRedoRejectReason.HAS_NUCLEIC_ACID.value,
            message="Entry contains a nucleic-acid polymer.",
            bnet=bnet,
        )

    return None


def _accepted_row(
    *,
    candidate: PdbRedoCandidate,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
    workflow_result: BDamageWorkflowResult,
    bnet_result: ProteinBnetResult,
) -> AcceptedBnetReferenceRow:
    if metadata.resolution_angstrom is None:
        raise ValueError("Accepted row cannot be created without resolution.")

    b_factor_model = _b_factor_model_decision(metadata=metadata, checks=checks)

    return AcceptedBnetReferenceRow(
        pdb_id=candidate.pdb_id,
        resolution_angstrom=metadata.resolution_angstrom,
        bnet=bnet_result.bnet,
        r_work=metadata.r_work,
        r_free=metadata.r_free,
        temperature_k=metadata.temperature_k,
        wilson_b=metadata.wilson_b,
        b_factor_restraint_weight=metadata.b_factor_restraint_weight,
        bnet_site_count=bnet_result.site_count,
        asp_glu_carboxyl_oxygen_count=checks.asp_glu_carboxyl_oxygen_count,
        asp_glu_residue_count=checks.asp_glu_residue_count,
        median_bdamage=bnet_result.median_bdamage,
        left_area=bnet_result.left_area,
        right_area=bnet_result.right_area,
        atom_count=checks.atom_count,
        non_hydrogen_atom_count=checks.non_hydrogen_atom_count,
        protein_atom_count=checks.protein_atom_count,
        selected_atom_count=(
            workflow_result.prepared_structure.report.selected_atom_count
        ),
        bdamage_window_size=workflow_result.window_size,
        has_protein=checks.has_protein,
        has_nucleic_acid=checks.has_nucleic_acid,
        is_xray=checks.is_xray,
        has_nonflat_protein_b_factors=checks.has_nonflat_protein_b_factors,
        has_asp_glu_residue_with_total_occupancy_below_one=(
            checks.has_asp_glu_residue_with_total_occupancy_below_one
        ),
        experimental_methods=checks.experimental_methods,
        uses_per_atom_b_factors=b_factor_model.uses_per_atom_b_factors,
        b_factor_model_source=b_factor_model.source,
        b_factor_refinement_type=metadata.b_factor_refinement_type,
        b_factor_refinement_type_source=(
            metadata.b_factor_refinement_type_source
        ),
        metadata_warnings=metadata.warnings,
        structure_check_warnings=checks.warnings,
        resolution_source=metadata.resolution_source,
        r_work_source=metadata.r_work_source,
        r_free_source=metadata.r_free_source,
        temperature_source=metadata.temperature_source,
        temperature_values_k=metadata.temperature_values_k,
        temperature_sources=metadata.temperature_sources,
        growth_temperature_k=metadata.growth_temperature_k,
        growth_temperature_values_k=metadata.growth_temperature_values_k,
        growth_temperature_source=metadata.growth_temperature_source,
        growth_temperature_sources=metadata.growth_temperature_sources,
        temperature_cache_status=metadata.temperature_cache_status,
        temperature_cache_message=metadata.temperature_cache_message,
        wilson_b_source=metadata.wilson_b_source,
        b_factor_restraint_weight_source=metadata.b_factor_restraint_weight_source,
        final_cif_path=candidate.final_cif_path,
        data_json_path=candidate.data_json_path,
    )


def _b_factor_model_decision(
    *,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
) -> _BFactorModelDecision:
    refinement_type = metadata.b_factor_refinement_type
    if refinement_type is not None:
        normalized_refinement_type = refinement_type.strip().upper()
        source = metadata.b_factor_refinement_type_source or "data_json:BREFTYPE"

        if normalized_refinement_type in PER_ATOM_B_FACTOR_REFINEMENT_TYPES:
            return _BFactorModelDecision(
                uses_per_atom_b_factors=True,
                source=source,
            )

        if normalized_refinement_type in NON_PER_ATOM_B_FACTOR_REFINEMENT_TYPES:
            return _BFactorModelDecision(
                uses_per_atom_b_factors=False,
                source=source,
            )

    return _BFactorModelDecision(
        uses_per_atom_b_factors=checks.has_nonflat_protein_b_factors,
        source=STRUCTURAL_B_FACTOR_MODEL_CHECK_SOURCE,
    )


def _rejected_from_eligibility(
    *,
    candidate: PdbRedoCandidate,
    metadata: PdbRedoMetadata,
    checks: PdbRedoStructureChecks,
    eligibility: BnetEligibilityResult,
    stage: PdbRedoProcessStage,
    bnet: float | None,
) -> RejectedBnetReferenceRow:
    return _rejected_row(
        candidate=candidate,
        metadata=metadata,
        checks=checks,
        stage=stage,
        reason=eligibility.primary_reason.value,
        message=eligibility.primary_message,
        bnet=bnet,
    )


def _rejected_result_from_exception(
    candidate: PdbRedoCandidate,
    *,
    stage: PdbRedoProcessStage,
    reason: str,
    message: str,
    error: BaseException,
    metadata: PdbRedoMetadata | None = None,
    checks: PdbRedoStructureChecks | None = None,
    include_traceback: bool,
) -> PdbRedoProcessResult:
    return PdbRedoProcessResult(
        pdb_id=candidate.pdb_id,
        rejected=_rejected_row(
            candidate=candidate,
            metadata=metadata,
            checks=checks,
            stage=stage,
            reason=reason,
            message=message,
            exception_type=type(error).__name__,
            traceback_text=traceback.format_exc() if include_traceback else None,
        ),
    )


def _rejected_row(
    *,
    candidate: PdbRedoCandidate,
    metadata: PdbRedoMetadata | None,
    checks: PdbRedoStructureChecks | None,
    stage: PdbRedoProcessStage,
    reason: str,
    message: str,
    exception_type: str | None = None,
    traceback_text: str | None = None,
    bnet: float | None = None,
) -> RejectedBnetReferenceRow:
    b_factor_model = (
        _b_factor_model_decision(metadata=metadata, checks=checks)
        if metadata is not None and checks is not None
        else None
    )

    return RejectedBnetReferenceRow(
        pdb_id=candidate.pdb_id,
        stage=stage,
        reason=reason,
        message=message,
        final_cif_path=candidate.final_cif_path,
        data_json_path=candidate.data_json_path,
        exception_type=exception_type,
        traceback_text=traceback_text,
        resolution_angstrom=(
            metadata.resolution_angstrom if metadata is not None else None
        ),
        r_work=metadata.r_work if metadata is not None else None,
        r_free=metadata.r_free if metadata is not None else None,
        temperature_k=metadata.temperature_k if metadata is not None else None,
        temperature_values_k=(
            metadata.temperature_values_k if metadata is not None else ()
        ),
        temperature_sources=(
            metadata.temperature_sources if metadata is not None else ()
        ),
        temperature_source=(
            metadata.temperature_source if metadata is not None else None
        ),
        growth_temperature_k=(
            metadata.growth_temperature_k if metadata is not None else None
        ),
        growth_temperature_values_k=(
            metadata.growth_temperature_values_k if metadata is not None else ()
        ),
        growth_temperature_source=(
            metadata.growth_temperature_source if metadata is not None else None
        ),
        growth_temperature_sources=(
            metadata.growth_temperature_sources if metadata is not None else ()
        ),
        temperature_cache_status=(
            metadata.temperature_cache_status if metadata is not None else None
        ),
        temperature_cache_message=(
            metadata.temperature_cache_message if metadata is not None else None
        ),
        wilson_b=metadata.wilson_b if metadata is not None else None,
        b_factor_restraint_weight=(
            metadata.b_factor_restraint_weight if metadata is not None else None
        ),
        b_factor_refinement_type=(
            metadata.b_factor_refinement_type if metadata is not None else None
        ),
        resolution_source=(
            metadata.resolution_source if metadata is not None else None
        ),
        r_work_source=metadata.r_work_source if metadata is not None else None,
        r_free_source=metadata.r_free_source if metadata is not None else None,
        wilson_b_source=metadata.wilson_b_source if metadata is not None else None,
        b_factor_restraint_weight_source=(
            metadata.b_factor_restraint_weight_source
            if metadata is not None
            else None
        ),
        b_factor_refinement_type_source=(
            metadata.b_factor_refinement_type_source
            if metadata is not None
            else None
        ),
        uses_per_atom_b_factors=(
            b_factor_model.uses_per_atom_b_factors
            if b_factor_model is not None
            else None
        ),
        b_factor_model_source=(
            b_factor_model.source if b_factor_model is not None else None
        ),
        asp_glu_carboxyl_oxygen_count=(
            checks.asp_glu_carboxyl_oxygen_count if checks is not None else None
        ),
        bnet=bnet,
        metadata_warnings=metadata.warnings if metadata is not None else (),
        structure_check_warnings=checks.warnings if checks is not None else (),
    )


__all__ = [
    "AcceptedBnetReferenceRow",
    "PdbRedoProcessResult",
    "PdbRedoProcessStage",
    "PdbRedoRejectReason",
    "RejectedBnetReferenceRow",
    "NON_PER_ATOM_B_FACTOR_REFINEMENT_TYPES",
    "PER_ATOM_B_FACTOR_REFINEMENT_TYPES",
    "STRUCTURAL_B_FACTOR_MODEL_CHECK_SOURCE",
    "process_pdb_redo_candidate",
]
