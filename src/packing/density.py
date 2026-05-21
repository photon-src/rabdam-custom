"""
Packing-density calculation for BDamage.

The trimmed crystal block contains the local neighbour cloud around the selected
asymmetric-unit atoms. For each selected atom, packing density is the number of
trimmed crystal atoms whose Cartesian distance from that selected atom is less
than the packing-density threshold, minus one to remove the selected atom's
central-cell copy.

This module performs the exact distance-counting step after the broader crystal
block has already been reduced by crystal.trim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from crystal.translate import TranslatedAtom
from crystal.trim import TrimmedCrystalBlock
from structure.models import PreparedAtom, PreparedStructure


class PackingDensityError(ValueError):
    """Raised when RABDAM cannot calculate packing density."""


@dataclass(frozen=True)
class PackingDensityAtomResult:
    """
    Packing-density result for one selected asymmetric-unit atom.

    packing_density_atom_index:
        One-based position of this atom in the packing-density result list.

    source_atom_index:
        Zero-based reader index of the selected asymmetric-unit atom.

    atom_serial:
        Atom serial number from the input structure, when available.

    neighbour_count:
        Number of trimmed crystal atoms within the packing-density threshold of
        this selected atom, after subtracting one for the central-cell copy of
        the atom itself.
    """

    packing_density_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    neighbour_count: int


@dataclass(frozen=True)
class PackingDensityResult:
    """
    Packing-density counts for the selected BDamage atoms.

    atom_results:
        One result per selected asymmetric-unit atom, in selected-atom order.

    packing_density_threshold:
        Distance cutoff in Angstroms used for neighbour counting.

    selected_atom_count:
        Number of selected asymmetric-unit atoms that were scored.

    neighbour_atom_count:
        Number of trimmed crystal atoms searched for each selected atom.
    """

    atom_results: tuple[PackingDensityAtomResult, ...]
    packing_density_threshold: float
    selected_atom_count: int
    neighbour_atom_count: int


def calculate_bdamage_packing_density(
    *,
    prepared_structure: PreparedStructure,
    trimmed_block: TrimmedCrystalBlock,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Calculate packing density for the BDamage-selected atoms.

    This convenience wrapper uses prepared_structure.selected_atoms as the atoms
    that receive packing-density counts and trimmed_block.atoms as the local
    crystal neighbour cloud.
    """

    return calculate_packing_density(
        selected_atoms=prepared_structure.selected_atoms,
        neighbour_atoms=trimmed_block.atoms,
        packing_density_threshold=packing_density_threshold,
    )


def calculate_packing_density(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_atoms: Iterable[TranslatedAtom],
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbour atoms within packing_density_threshold of each selected atom.
    """

    if (
        not math.isfinite(packing_density_threshold)
        or packing_density_threshold <= 0
    ):
        raise PackingDensityError(
            "packing_density_threshold must be a finite positive number, "
            f"got {packing_density_threshold!r}."
        )

    selected_atom_tuple = tuple(selected_atoms)
    if not selected_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density for an empty selected-atom list."
        )

    neighbour_atom_tuple = tuple(neighbour_atoms)
    if not neighbour_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density with an empty neighbour-atom list."
        )

    threshold_squared = float(packing_density_threshold) ** 2

    atom_results = tuple(
        PackingDensityAtomResult(
            packing_density_atom_index=selected_atom_index,
            source_atom_index=selected_atom.record.source_atom_index,
            atom_serial=selected_atom.record.atom_serial,
            neighbour_count=_count_neighbours_excluding_selected_atom_self_copy(
                selected_atom=selected_atom,
                neighbour_atoms=neighbour_atom_tuple,
                threshold_squared=threshold_squared,
            ),
        )
        for selected_atom_index, selected_atom in enumerate(selected_atom_tuple, start=1)
    )

    return PackingDensityResult(
        atom_results=atom_results,
        packing_density_threshold=float(packing_density_threshold),
        selected_atom_count=len(selected_atom_tuple),
        neighbour_atom_count=len(neighbour_atom_tuple),
    )


def _count_neighbours_excluding_selected_atom_self_copy(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: tuple[TranslatedAtom, ...],
    threshold_squared: float,
) -> int:
    """
    Count neighbours, then remove the selected atom's central-cell copy.
    """

    raw_count = _count_neighbours_within_threshold_squared(
        selected_atom=selected_atom,
        neighbour_atoms=neighbour_atoms,
        threshold_squared=threshold_squared,
    )
    if not _selected_atom_self_copy_is_counted(
        selected_atom=selected_atom,
        neighbour_atoms=neighbour_atoms,
        threshold_squared=threshold_squared,
    ):
        raise PackingDensityError(
            "Cannot subtract the selected atom's central-cell copy from the "
            "packing-density count because that copy was not counted. Check "
            "that the neighbour cloud contains the selected atom's central-cell "
            "image."
        )

    return raw_count - 1


def _selected_atom_self_copy_is_counted(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: Iterable[TranslatedAtom],
    threshold_squared: float,
) -> bool:
    """
    Return True when the selected atom's central-cell copy is counted.
    """

    selected_x = selected_atom.record.x
    selected_y = selected_atom.record.y
    selected_z = selected_atom.record.z

    return any(
        neighbour_atom.source_atom_index == selected_atom.record.source_atom_index
        and neighbour_atom.is_identity_symmetry_operation
        and neighbour_atom.translation_a == 0
        and neighbour_atom.translation_b == 0
        and neighbour_atom.translation_c == 0
        and squared_distance_to_translated_atom(
            selected_x=selected_x,
            selected_y=selected_y,
            selected_z=selected_z,
            neighbour_atom=neighbour_atom,
        )
        < threshold_squared
        for neighbour_atom in neighbour_atoms
    )


def _count_neighbours_within_threshold_squared(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: Iterable[TranslatedAtom],
    threshold_squared: float,
) -> int:
    """
    Count neighbours whose squared Cartesian distance is < threshold_squared.

    Squared distances are used to avoid square-root calculations while producing
    the same inclusion result as comparing true Euclidean distances.
    """

    if not math.isfinite(threshold_squared) or threshold_squared < 0:
        raise PackingDensityError(
            "threshold_squared must be a finite non-negative number, "
            f"got {threshold_squared!r}."
        )

    selected_x = selected_atom.record.x
    selected_y = selected_atom.record.y
    selected_z = selected_atom.record.z

    count = 0
    for neighbour_atom in neighbour_atoms:
        if squared_distance_to_translated_atom(
            selected_x=selected_x,
            selected_y=selected_y,
            selected_z=selected_z,
            neighbour_atom=neighbour_atom,
        ) < threshold_squared:
            count += 1

    return count


def squared_distance_to_translated_atom(
    *,
    selected_x: float,
    selected_y: float,
    selected_z: float,
    neighbour_atom: TranslatedAtom,
) -> float:
    """
    Return squared Cartesian distance from a selected atom to a neighbour atom.
    """

    dx = selected_x - neighbour_atom.x
    dy = selected_y - neighbour_atom.y
    dz = selected_z - neighbour_atom.z

    return float(dx * dx + dy * dy + dz * dz)


def packing_density_counts_as_tuple(
    result: PackingDensityResult,
) -> tuple[int, ...]:
    """Return only neighbour counts from a packing-density result."""

    return tuple(atom_result.neighbour_count for atom_result in result.atom_results)
