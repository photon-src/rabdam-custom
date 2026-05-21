"""
End-to-end BDamage workflow orchestration.

This module wires together the prepared-structure, crystal-environment,
packing-density, and BDamage scoring stages.  It intentionally keeps the
workflow thin: each scientific calculation remains in its dedicated module.

Default options match the RABDAM 2 BDamage defaults used for compatibility
comparisons:

    packing density threshold = 7.0 Angstroms
    window size fraction      = 0.02 of selected atoms
    minimum window size       = 11 atoms
    translation range         = 1, giving a 3x3x3 crystal block

The sliding-window atom count is derived from the selected-atom count in the
same way as RABDAM 2: round(atom_count * window_fraction), make the value odd,
and enforce a minimum of 11 atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bdamage.score import BDamageScoreResult, calculate_bdamage_scores_for_structure
from crystal.symmetry import SymmetryExpandedStructure, expand_prepared_structure_by_symmetry
from crystal.translate import TranslatedCrystalBlock, translate_expanded_unit_cell
from crystal.trim import TrimmedCrystalBlock, trim_translated_block_for_bdamage
from input.reader import StructureData
from packing.density import PackingDensityResult, calculate_bdamage_packing_density
from structure.models import PreparedStructure, StructurePreparationOptions
from structure.prepare import prepare_structure


class BDamageWorkflowError(ValueError):
    """Raised when RABDAM cannot run the end-to-end BDamage workflow."""


@dataclass(frozen=True)
class BDamageWorkflowOptions:
    """
    Options controlling the end-to-end BDamage workflow.

    packing_density_threshold:
        Distance cutoff in Angstroms used for trimming and neighbour counting.

    window_size_fraction:
        Fraction of selected atoms used to derive the sliding-window size.

    minimum_window_size:
        Minimum sliding-window atom count after fraction/odd adjustment.

    translation_range:
        Number of whole unit cells to translate in each direction.  The default
        of 1 generates offsets -1, 0, and +1 along a, b, and c.
    """

    packing_density_threshold: float = 7.0
    window_size_fraction: float = 0.02
    minimum_window_size: int = 11
    translation_range: int = 1


@dataclass(frozen=True)
class BDamageWorkflowResult:
    """
    Complete result of an end-to-end BDamage calculation.

    prepared_structure:
        Cleaned and selected asymmetric-unit atoms used for BDamage.

    symmetry_expanded_structure:
        Full unit-cell atom cloud after applying crystallographic symmetry.

    translated_block:
        Symmetry-expanded unit cell copied into neighbouring unit cells.

    trimmed_block:
        Local neighbour cloud retained for packing-density calculation.

    packing_density_result:
        Per-selected-atom packing-density neighbour counts.

    bdamage_score_result:
        Per-selected-atom BDamage scores.

    options:
        Workflow options used for this calculation.

    window_size:
        Final integer sliding-window atom count used for BDamage scoring.
    """

    prepared_structure: PreparedStructure
    symmetry_expanded_structure: SymmetryExpandedStructure
    translated_block: TranslatedCrystalBlock
    trimmed_block: TrimmedCrystalBlock
    packing_density_result: PackingDensityResult
    bdamage_score_result: BDamageScoreResult
    options: BDamageWorkflowOptions
    window_size: int


def calculate_bdamage_for_structure_data(
    structure_data: StructureData,
    *,
    workflow_options: BDamageWorkflowOptions | None = None,
    preparation_options: StructurePreparationOptions | None = None,
) -> BDamageWorkflowResult:
    """
    Prepare raw structure data and run the complete BDamage workflow.
    """

    prepared_structure = prepare_structure(
        structure_data,
        options=preparation_options,
    )

    return calculate_bdamage_for_prepared_structure(
        prepared_structure=prepared_structure,
        options=workflow_options,
    )


def calculate_bdamage_for_prepared_structure(
    *,
    prepared_structure: PreparedStructure,
    options: BDamageWorkflowOptions | None = None,
) -> BDamageWorkflowResult:
    """
    Run symmetry, translation, trimming, packing density, and BDamage scoring.
    """

    if options is None:
        options = BDamageWorkflowOptions()

    validate_workflow_options(options)

    selected_atom_count = len(prepared_structure.selected_atoms)
    if selected_atom_count == 0:
        raise BDamageWorkflowError(
            "Cannot calculate BDamage for a prepared structure with no selected atoms."
        )

    window_size = bdamage_window_size_from_fraction(
        atom_count=selected_atom_count,
        window_size_fraction=options.window_size_fraction,
        minimum_window_size=options.minimum_window_size,
    )

    if window_size > selected_atom_count:
        raise BDamageWorkflowError(
            "Calculated BDamage window size is larger than the number of selected atoms: "
            f"window_size={window_size!r}, selected_atom_count={selected_atom_count!r}."
        )

    symmetry_expanded_structure = expand_prepared_structure_by_symmetry(prepared_structure)
    translated_block = translate_expanded_unit_cell(
        symmetry_expanded_structure,
        translation_range=options.translation_range,
    )
    trimmed_block = trim_translated_block_for_bdamage(
        translated_block=translated_block,
        prepared_structure=prepared_structure,
        padding=options.packing_density_threshold,
    )
    packing_density_result = calculate_bdamage_packing_density(
        prepared_structure=prepared_structure,
        trimmed_block=trimmed_block,
        packing_density_threshold=options.packing_density_threshold,
    )
    bdamage_score_result = calculate_bdamage_scores_for_structure(
        prepared_structure=prepared_structure,
        packing_density_result=packing_density_result,
        window_size=window_size,
    )

    return BDamageWorkflowResult(
        prepared_structure=prepared_structure,
        symmetry_expanded_structure=symmetry_expanded_structure,
        translated_block=translated_block,
        trimmed_block=trimmed_block,
        packing_density_result=packing_density_result,
        bdamage_score_result=bdamage_score_result,
        options=options,
        window_size=window_size,
    )


def bdamage_window_size_from_fraction(
    *,
    atom_count: int,
    window_size_fraction: float = 0.02,
    minimum_window_size: int = 11,
) -> int:
    """
    Return the RABDAM-compatible integer BDamage sliding-window size.

    The window size is derived from the number of selected atoms by rounding
    atom_count * window_size_fraction, making the result odd, and enforcing a
    minimum window size.
    """

    if type(atom_count) is not int or atom_count <= 0:
        raise BDamageWorkflowError(
            f"atom_count must be a positive integer, got {atom_count!r}."
        )

    if (
        not math.isfinite(window_size_fraction)
        or not 0 < window_size_fraction < 1
    ):
        raise BDamageWorkflowError(
            "window_size_fraction must be a finite float in the range 0 to 1, "
            f"got {window_size_fraction!r}."
        )

    if type(minimum_window_size) is not int or minimum_window_size <= 0:
        raise BDamageWorkflowError(
            "minimum_window_size must be a positive integer, "
            f"got {minimum_window_size!r}."
        )

    window_size = int(round(atom_count * window_size_fraction, 0))
    if window_size % 2 == 0:
        window_size += 1

    if window_size < minimum_window_size:
        window_size = minimum_window_size

    return window_size


def validate_workflow_options(options: BDamageWorkflowOptions) -> None:
    """Validate end-to-end BDamage workflow options."""

    if (
        not math.isfinite(options.packing_density_threshold)
        or options.packing_density_threshold <= 0
    ):
        raise BDamageWorkflowError(
            "packing_density_threshold must be a finite positive number, "
            f"got {options.packing_density_threshold!r}."
        )

    if (
        not math.isfinite(options.window_size_fraction)
        or not 0 < options.window_size_fraction < 1
    ):
        raise BDamageWorkflowError(
            "window_size_fraction must be a finite float in the range 0 to 1, "
            f"got {options.window_size_fraction!r}."
        )

    if type(options.minimum_window_size) is not int or options.minimum_window_size <= 0:
        raise BDamageWorkflowError(
            "minimum_window_size must be a positive integer, "
            f"got {options.minimum_window_size!r}."
        )

    if type(options.translation_range) is not int or options.translation_range < 0:
        raise BDamageWorkflowError(
            "translation_range must be a non-negative integer, "
            f"got {options.translation_range!r}."
        )