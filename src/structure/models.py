"""
Shared structure-preparation models for RABDAM.

This module contains dataclasses and enums used across the structure package.
Keeping them here prevents circular imports between the preparation steps.
"""

from dataclasses import dataclass

from input.reader import AtomRecord, StructureMetadata


class StructurePreparationError(ValueError):
    """Raised when RABDAM cannot prepare structure data for calculation."""


@dataclass(frozen=True)
class StructurePreparationOptions:
    """
    Options controlling structure preparation.
    """

    remove_hydrogens: bool = True
    require_valid_occupancy: bool = True
    require_positive_b_factor: bool = True
    resolve_altlocs: bool = True

    include_hetatm_in_selection: bool = False
    include_protein_like_hetatm_in_selection: bool = False
    include_nucleic_acid_in_selection: bool = False
    require_protein_selection: bool = True

    remove_atom_serials: frozenset[int] = frozenset()
    add_atom_serials: frozenset[int] = frozenset()

    remove_component_names: frozenset[str] = frozenset()
    add_component_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PreparedAtom:
    """
    An atom record with structure-preparation annotations.
    """

    record: AtomRecord
    is_hydrogen: bool
    is_protein: bool
    is_nucleic_acid: bool
    is_solvent: bool
    is_hetatm: bool


@dataclass(frozen=True)
class StructurePreparationReport:
    """
    Counts and warnings produced during structure preparation.
    """

    input_atom_count: int
    cleaned_atom_count: int
    selected_atom_count: int
    removed_hydrogen_count: int
    removed_invalid_occupancy_count: int
    removed_invalid_b_factor_count: int
    removed_altloc_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedStructure:
    """
    Prepared structure data for later RABDAM stages.

    cleaned_atoms:
        All valid cleaned atoms retained from the asymmetric unit.

    selected_atoms:
        The subset of cleaned atoms selected for BDamage calculation.

    metadata:
        Structure-level metadata from the reader.

    report:
        Counts and warnings from the preparation stage.
    """

    cleaned_atoms: tuple[PreparedAtom, ...]
    selected_atoms: tuple[PreparedAtom, ...]
    metadata: StructureMetadata
    report: StructurePreparationReport
