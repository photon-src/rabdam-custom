"""
Atom classification for RABDAM 3 structure preparation.
"""

from collections.abc import Iterable

import gemmi

from input.reader import AtomRecord
from structure.filters import is_hydrogen
from structure.models import PreparedAtom


PROTEIN_COMPONENT_OVERRIDES = frozenset(
    {
        "ASH",
        "CYM",
        "CYX",
        "GLH",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
    }
)


NUCLEIC_ACID_COMPONENT_OVERRIDES = frozenset(
    {
        "5MC",
        "YYG",
    }
)


SOLVENTS = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
    }
)


def classify_atoms(atoms: Iterable[AtomRecord]) -> tuple[PreparedAtom, ...]:
    """
    Classify atom records for RABDAM preparation.
    """

    return tuple(classify_atom(atom) for atom in atoms)


def classify_atom(atom: AtomRecord) -> PreparedAtom:
    """
    Classify one atom record for RABDAM preparation.
    """

    component_name = atom.residue_name.strip().upper()
    record_type = atom.record_type.strip().upper()

    return PreparedAtom(
        record=atom,
        is_hydrogen=is_hydrogen(atom),
        is_protein=is_protein_component(component_name),
        is_nucleic_acid=is_nucleic_acid_component(component_name),
        is_solvent=is_solvent_component(component_name),
        is_hetatm=record_type == "HETATM",
    )


def is_protein_component(component_name: str) -> bool:
    """Return True if a component should be treated as protein-like."""

    if component_name in PROTEIN_COMPONENT_OVERRIDES:
        return True

    return gemmi.find_tabulated_residue(component_name).is_amino_acid()


def is_nucleic_acid_component(component_name: str) -> bool:
    """Return True if a component should be treated as nucleic-acid-like."""

    if component_name in NUCLEIC_ACID_COMPONENT_OVERRIDES:
        return True

    return gemmi.find_tabulated_residue(component_name).is_nucleic_acid()


def is_solvent_component(component_name: str) -> bool:
    """Return True if a component should be treated as solvent."""

    if component_name in SOLVENTS:
        return True

    return gemmi.find_tabulated_residue(component_name).is_water()
