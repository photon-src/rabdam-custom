"""
Read local structure files for RABDAM 3.

This module takes a resolved local structure input and reads the structure file
using Gemmi.

It converts the file into RABDAM-owned data objects.
"""

from dataclasses import dataclass
from pathlib import Path

import gemmi

from input.resolver import (
    InputResolutionError,
    ResolvedStructureInput,
    StructureFileFormat,
)


class StructureReadError(InputResolutionError):
    """Raised when RABDAM cannot read a local structure file."""


@dataclass(frozen=True)
class AtomRecord:
    """
    One atom record read from a structure file.

    source_atom_index:
        Zero-based index assigned by reader.py in the order atoms are read.
        Used internally for stable atom identity and deduplication.
    """

    source_atom_index: int
    model_number: int
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_name: str
    element: str
    altloc: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    atom_serial: int | None
    record_type: str


@dataclass(frozen=True)
class StructureMetadata:
    """Basic metadata read from a structure file."""

    source_path: Path
    structure_id: str | None
    file_format: StructureFileFormat
    space_group: str | None
    unit_cell_a: float | None
    unit_cell_b: float | None
    unit_cell_c: float | None
    unit_cell_alpha: float | None
    unit_cell_beta: float | None
    unit_cell_gamma: float | None


@dataclass(frozen=True)
class StructureData:
    """Structure data passed from the input layer to later RABDAM stages."""

    atoms: list[AtomRecord]
    metadata: StructureMetadata


def read_structure(resolved_input: ResolvedStructureInput) -> StructureData:
    """
    Read a resolved local structure file.
    """

    if not resolved_input.is_local_file:
        raise StructureReadError(
            "read_structure expected a local file input, "
            f"but got {resolved_input.source_type.value!r}."
        )

    if resolved_input.local_path is None:
        raise StructureReadError("Local structure input is missing local_path.")

    path = resolved_input.local_path

    if not path.exists():
        raise StructureReadError(f"Structure file does not exist: {path!s}")

    if not path.is_file():
        raise StructureReadError(f"Expected a structure file, but got: {path!s}")

    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:
        raise StructureReadError(
            f"Gemmi could not read structure file {path!s}: {error}"
        ) from error

    if len(structure) != 1:
        raise StructureReadError(
            f"Expected a single-model structure, but found {len(structure)} models."
        )

    atoms = _extract_atom_records(structure)

    if not atoms:
        raise StructureReadError(
            f"No atom records were found in structure file: {path!s}"
        )

    metadata = _extract_metadata(
        structure=structure,
        resolved_input=resolved_input,
        path=path,
    )

    return StructureData(
        atoms=atoms,
        metadata=metadata,
    )


def _extract_atom_records(structure: gemmi.Structure) -> list[AtomRecord]:
    """Extract atom records from a Gemmi structure."""

    atom_records: list[AtomRecord] = []

    for model_index, model in enumerate(structure, start=1):
        for chain in model:
            chain_id = chain.name

            for residue in chain:
                residue_name = residue.name
                residue_number = _get_residue_number(residue)
                insertion_code = _get_insertion_code(residue)
                record_type = _get_record_type(residue)

                for atom in residue:
                    position = atom.pos

                    atom_records.append(
                        AtomRecord(
                            source_atom_index=len(atom_records),
                            model_number=model_index,
                            chain_id=chain_id,
                            residue_name=residue_name,
                            residue_number=residue_number,
                            insertion_code=insertion_code,
                            atom_name=atom.name.strip(),
                            element=atom.element.name,
                            altloc=_clean_altloc(atom.altloc),
                            x=float(position.x),
                            y=float(position.y),
                            z=float(position.z),
                            occupancy=float(atom.occ),
                            b_factor=float(atom.b_iso),
                            atom_serial=_get_atom_serial(atom),
                            record_type=record_type,
                        )
                    )

    return atom_records


def _extract_metadata(
    *,
    structure: gemmi.Structure,
    resolved_input: ResolvedStructureInput,
    path: Path,
) -> StructureMetadata:
    """Extract basic metadata from a Gemmi structure."""

    cell = structure.cell
    space_group = _get_space_group_name(structure)

    return StructureMetadata(
        source_path=path.resolve(),
        structure_id=resolved_input.structure_id,
        file_format=resolved_input.file_format,
        space_group=space_group,
        unit_cell_a=_none_if_zero(cell.a),
        unit_cell_b=_none_if_zero(cell.b),
        unit_cell_c=_none_if_zero(cell.c),
        unit_cell_alpha=_none_if_zero(cell.alpha),
        unit_cell_beta=_none_if_zero(cell.beta),
        unit_cell_gamma=_none_if_zero(cell.gamma),
    )


def _get_space_group_name(structure: gemmi.Structure) -> str | None:
    """Return the space-group name if Gemmi provides one."""

    space_group = getattr(structure, "spacegroup_hm", None)

    if space_group:
        return str(space_group)

    return None


def _get_residue_number(residue: gemmi.Residue) -> int | None:
    """Return the residue sequence number if available."""

    residue_number = getattr(residue.seqid, "num", None)

    if residue_number is None:
        return None

    return int(residue_number)


def _get_insertion_code(residue: gemmi.Residue) -> str:
    """Return the residue insertion code if available."""

    insertion_code = getattr(residue.seqid, "icode", "")

    if insertion_code is None:
        return ""

    return str(insertion_code).strip()


def _get_atom_serial(atom: gemmi.Atom) -> int | None:
    """Return the atom serial number if available."""

    atom_serial = getattr(atom, "serial", None)

    if atom_serial is None:
        return None

    return int(atom_serial)


def _clean_altloc(altloc: str) -> str:
    """Return a clean alternate-location identifier."""

    if altloc is None:
        return ""

    cleaned = str(altloc).replace("\x00", "").strip()

    if cleaned in {".", "?"}:
        return ""

    return cleaned


def _get_record_type(residue: gemmi.Residue) -> str:
    """
    Return ATOM or HETATM for a residue.

    Gemmi stores this information at residue level through het_flag.
    """

    het_flag = str(getattr(residue, "het_flag", "")).strip().upper()

    if het_flag == "H":
        return "HETATM"

    return "ATOM"


def _none_if_zero(value: float) -> float | None:
    """
    Convert zero cell values to None.

    Some malformed or non-crystallographic inputs may not have useful unit-cell
    values.
    """

    value = float(value)

    if value == 0.0:
        return None

    return value
