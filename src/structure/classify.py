"""
Atom classification for RABDAM 3 structure preparation.
"""

from collections.abc import Iterable
from functools import lru_cache

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
    is_protein, is_nucleic_acid, is_solvent = classify_component(component_name)

    return PreparedAtom(
        record=atom,
        is_hydrogen=is_hydrogen(atom),
        is_protein=is_protein,
        is_nucleic_acid=is_nucleic_acid,
        is_solvent=is_solvent,
        is_hetatm=record_type == "HETATM",
    )


def classify_component(component_name: str) -> tuple[bool, bool, bool]:
    """
    Return protein, nucleic-acid, and solvent flags for a component name.
    """

    return _classify_normalized_component(component_name.strip().upper())


@lru_cache(maxsize=None)
def _classify_normalized_component(component_name: str) -> tuple[bool, bool, bool]:
    """
    Return classification flags for a normalized component name.
    """

    residue = gemmi.find_tabulated_residue(component_name)

    return (
        component_name in PROTEIN_COMPONENT_OVERRIDES or residue.is_amino_acid(),
        component_name in NUCLEIC_ACID_COMPONENT_OVERRIDES
        or residue.is_nucleic_acid(),
        component_name in SOLVENTS or residue.is_water(),
    )


def is_protein_component(component_name: str) -> bool:
    """Return True if a component should be treated as protein-like."""

    return classify_component(component_name)[0]


def is_nucleic_acid_component(component_name: str) -> bool:
    """Return True if a component should be treated as nucleic-acid-like."""

    return classify_component(component_name)[1]


def is_solvent_component(component_name: str) -> bool:
    """Return True if a component should be treated as solvent."""

    return classify_component(component_name)[2]
