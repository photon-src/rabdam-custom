"""Structural checks for PDB-REDO Bnet reference-database candidates.

This module inspects a PDB-REDO final mmCIF model and extracts cheap structural
facts needed before running the full RABDAM calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import gemmi

from .discover import PdbRedoCandidate
from structure.classify import is_nucleic_acid_component, is_protein_component


_ASP_GLU_RESIDUE_NAMES = frozenset({"ASP", "GLU", "DAS", "DGL"})
_ASP_GLU_CARBOXYL_OXYGEN_NAMES = frozenset({"OD1", "OD2", "OE1", "OE2"})
_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})
_PROTEIN_POLYMER_TYPES = frozenset(
    {
        "polypeptide(l)",
        "polypeptide(d)",
        "cyclic-pseudo-peptide",
    }
)
_NUCLEIC_ACID_POLYMER_TYPE_KEYWORDS = frozenset(
    {
        "nucleotide",
        "nucleic acid",
    }
)
_NUCLEIC_ACID_POLYMER_TYPES = frozenset(
    {
        "polydeoxyribonucleotide",
        "polyribonucleotide",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
        "peptide nucleic acid",
    }
)
_MISSING_VALUES = frozenset({"", ".", "?", "null", "none", "nan", "na", "n/a"})


class PdbRedoStructureCheckError(ValueError):
    """Raised when PDB-REDO structural checks cannot be performed."""


@dataclass(frozen=True, slots=True)
class PdbRedoStructureChecks:
    """Structural facts extracted from a PDB-REDO final mmCIF model.

    ``has_nonflat_protein_b_factors`` is the RABDAM2-style structural fallback
    check for per-atom B-factors. Prefer explicit PDB-REDO B-factor model
    metadata when available.
    """

    pdb_id: str
    has_protein: bool
    has_nucleic_acid: bool
    experimental_methods: tuple[str, ...]
    asp_glu_carboxyl_oxygen_count: int
    asp_glu_residue_count: int
    asp_glu_residue_keys_with_occupancy_below_one: tuple[str, ...]
    has_nonflat_protein_b_factors: bool
    atom_count: int
    non_hydrogen_atom_count: int
    protein_atom_count: int
    model_count: int
    final_cif_path: Path
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_asp_glu_residue_with_total_occupancy_below_one(self) -> bool:
        """Return whether any Asp/Glu residue has total occupancy below one."""

        return bool(self.asp_glu_residue_keys_with_occupancy_below_one)

    @property
    def is_xray(self) -> bool:
        """Return whether any experimental method looks like X-ray crystallography."""

        return any(
            "x-ray" in method.casefold()
            for method in self.experimental_methods
        )


def read_pdb_redo_structure_checks(
    candidate: PdbRedoCandidate,
) -> PdbRedoStructureChecks:
    """Read structural checks for a discovered PDB-REDO candidate."""

    final_cif_path = candidate.final_cif_path
    if not final_cif_path.is_file():
        raise PdbRedoStructureCheckError(
            f"PDB-REDO final mmCIF does not exist: {final_cif_path}"
        )

    try:
        structure = gemmi.read_structure(str(final_cif_path))
    except (OSError, RuntimeError, ValueError) as error:
        raise PdbRedoStructureCheckError(
            f"Could not parse PDB-REDO final mmCIF: {final_cif_path}"
        ) from error

    try:
        document = gemmi.cif.read_file(str(final_cif_path))
    except (OSError, RuntimeError, ValueError) as error:
        raise PdbRedoStructureCheckError(
            f"Could not parse PDB-REDO final mmCIF as CIF: {final_cif_path}"
        ) from error

    block = document[0] if len(document) else None
    warnings: list[str] = []

    if len(structure) == 0:
        raise PdbRedoStructureCheckError(
            f"PDB-REDO final mmCIF contains no models: {final_cif_path}"
        )

    model = structure[0]
    if len(structure) > 1:
        warnings.append(
            "Structure contains multiple models; structural checks use only "
            "the first model."
        )

    experimental_methods = _read_experimental_methods(block)
    entity_polymer_types = _read_entity_polymer_types(block)

    has_protein = _has_protein_polymer(entity_polymer_types)
    has_nucleic_acid = _has_nucleic_acid_polymer(entity_polymer_types)

    if not entity_polymer_types:
        warnings.append(
            "Could not read _entity_poly.type values; protein/nucleic-acid "
            "classification falls back to residue-name heuristics."
        )
        has_protein = _model_has_protein_like_residue(model)
        has_nucleic_acid = _model_has_nucleic_acid_like_residue(model)

    asp_glu_info = _collect_asp_glu_info(model)
    b_factor_info = _collect_b_factor_info(model)

    return PdbRedoStructureChecks(
        pdb_id=candidate.pdb_id,
        has_protein=has_protein,
        has_nucleic_acid=has_nucleic_acid,
        experimental_methods=experimental_methods,
        asp_glu_carboxyl_oxygen_count=asp_glu_info.carboxyl_oxygen_count,
        asp_glu_residue_count=asp_glu_info.residue_count,
        asp_glu_residue_keys_with_occupancy_below_one=(
            asp_glu_info.residue_keys_with_occupancy_below_one
        ),
        has_nonflat_protein_b_factors=(
            b_factor_info.has_nonflat_protein_b_factors
        ),
        atom_count=b_factor_info.atom_count,
        non_hydrogen_atom_count=b_factor_info.non_hydrogen_atom_count,
        protein_atom_count=b_factor_info.protein_atom_count,
        model_count=len(structure),
        final_cif_path=final_cif_path,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class _AspGluInfo:
    carboxyl_oxygen_count: int
    residue_count: int
    residue_keys_with_occupancy_below_one: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BFactorInfo:
    has_nonflat_protein_b_factors: bool
    atom_count: int
    non_hydrogen_atom_count: int
    protein_atom_count: int


def _collect_asp_glu_info(model: gemmi.Model) -> _AspGluInfo:
    carboxyl_oxygen_count = 0
    residue_count = 0
    bad_occupancy_residue_keys: list[str] = []

    for chain in model:
        for residue in chain:
            residue_name = residue.name.upper()
            if residue_name not in _ASP_GLU_RESIDUE_NAMES:
                continue

            residue_count += 1
            residue_key = _residue_key(model, chain, residue)

            carboxyl_atoms = [
                atom
                for atom in residue
                if atom.name.strip().upper() in _ASP_GLU_CARBOXYL_OXYGEN_NAMES
            ]
            carboxyl_oxygen_count += len(carboxyl_atoms)

            if _residue_has_total_carboxyl_oxygen_occupancy_below_one(
                carboxyl_atoms
            ):
                bad_occupancy_residue_keys.append(residue_key)

    return _AspGluInfo(
        carboxyl_oxygen_count=carboxyl_oxygen_count,
        residue_count=residue_count,
        residue_keys_with_occupancy_below_one=tuple(bad_occupancy_residue_keys),
    )


def _residue_has_total_carboxyl_oxygen_occupancy_below_one(
    atoms: list[gemmi.Atom],
) -> bool:
    """Return whether a residue has any carboxyl oxygen site below occupancy 1.

    For alternate conformers, occupancies are summed by atom name. For example,
    OD1 altloc A at 0.5 plus OD1 altloc B at 0.5 is treated as full occupancy.
    """

    if not atoms:
        return False

    occupancy_by_atom_name: dict[str, float] = {}

    for atom in atoms:
        atom_name = atom.name.strip().upper()
        occupancy = float(atom.occ)

        if not math.isfinite(occupancy):
            return True

        occupancy_by_atom_name[atom_name] = (
            occupancy_by_atom_name.get(atom_name, 0.0) + occupancy
        )

    return any(
        total_occupancy < 0.999
        for total_occupancy in occupancy_by_atom_name.values()
    )


def _collect_b_factor_info(model: gemmi.Model) -> _BFactorInfo:
    atom_count = 0
    non_hydrogen_atom_count = 0
    protein_atom_count = 0
    residue_backbone_b_values: dict[str, list[float]] = {}

    for chain in model:
        for residue in chain:
            is_protein_residue = _is_protein_like_residue_name(residue.name)
            residue_key = _residue_key(model, chain, residue)

            for atom in residue:
                atom_count += 1

                if not _is_hydrogen(atom):
                    non_hydrogen_atom_count += 1

                if is_protein_residue:
                    protein_atom_count += 1
                    atom_name = atom.name.strip().upper()
                    if not _is_hydrogen(atom) and atom_name in _BACKBONE_ATOM_NAMES:
                        residue_backbone_b_values.setdefault(
                            residue_key,
                            [],
                        ).append(float(atom.b_iso))

    has_nonflat_protein_b_factors = _has_per_atom_b_factors_by_backbone(
        residue_backbone_b_values
    )

    return _BFactorInfo(
        has_nonflat_protein_b_factors=has_nonflat_protein_b_factors,
        atom_count=atom_count,
        non_hydrogen_atom_count=non_hydrogen_atom_count,
        protein_atom_count=protein_atom_count,
    )


def _has_per_atom_b_factors_by_backbone(
    residue_backbone_b_values: dict[str, list[float]],
    *,
    threshold_fraction: float = 0.2,
) -> bool:
    """Return the RABDAM2-style per-atom B-factor fallback result.

    RABDAM2 counts residues whose listed backbone atom B-factors are all
    identical. If at least 20% of residues, with a minimum threshold of three
    residues, look per-residue/flat, the structure is treated as not having
    per-atom B-factors.
    """

    total_residue_count = len(residue_backbone_b_values)
    if total_residue_count == 0:
        return False

    per_residue_b_factor_count = 0
    for b_values in residue_backbone_b_values.values():
        finite_values = [value for value in b_values if math.isfinite(value)]
        if finite_values and len(set(finite_values)) == 1:
            per_residue_b_factor_count += 1

    threshold = max(3.0, total_residue_count * threshold_fraction)
    if per_residue_b_factor_count >= threshold:
        return False

    return True


def _read_experimental_methods(block: gemmi.cif.Block | None) -> tuple[str, ...]:
    if block is None:
        return ()

    methods: list[str] = []
    scalar_method = block.find_value("_exptl.method")
    if scalar_method is not None:
        methods.append(_strip_cif_quotes(scalar_method))

    for value in _find_loop_column_values(block, "_exptl.method"):
        methods.append(_strip_cif_quotes(value))

    return _deduplicate_strings(methods)


def _read_entity_polymer_types(block: gemmi.cif.Block | None) -> tuple[str, ...]:
    if block is None:
        return ()

    values = [
        _strip_cif_quotes(value)
        for value in _find_loop_column_values(block, "_entity_poly.type")
    ]

    scalar_value = block.find_value("_entity_poly.type")
    if scalar_value is not None:
        values.append(_strip_cif_quotes(scalar_value))

    return _deduplicate_strings(values)


def _find_loop_column_values(
    block: gemmi.cif.Block,
    tag: str,
) -> tuple[str, ...]:
    values: list[str] = []

    loop = block.find_loop(tag)
    if loop is None:
        return ()

    for value in loop:
        if _is_missing_text(value):
            continue
        values.append(value)

    return tuple(values)


def _has_protein_polymer(entity_polymer_types: tuple[str, ...]) -> bool:
    normalized_types = {
        _normalize_polymer_type(value) for value in entity_polymer_types
    }
    return bool(normalized_types & _PROTEIN_POLYMER_TYPES)


def _has_nucleic_acid_polymer(entity_polymer_types: tuple[str, ...]) -> bool:
    normalized_types = {
        _normalize_polymer_type(value) for value in entity_polymer_types
    }
    return any(
        polymer_type in _NUCLEIC_ACID_POLYMER_TYPES
        or any(
            keyword in polymer_type
            for keyword in _NUCLEIC_ACID_POLYMER_TYPE_KEYWORDS
        )
        for polymer_type in normalized_types
    )


def _normalize_polymer_type(value: str) -> str:
    return _strip_cif_quotes(value).strip().casefold()


def _model_has_protein_like_residue(model: gemmi.Model) -> bool:
    return any(
        _is_protein_like_residue_name(residue.name)
        for chain in model
        for residue in chain
    )


def _model_has_nucleic_acid_like_residue(model: gemmi.Model) -> bool:
    return any(
        _is_nucleic_acid_like_residue_name(residue.name)
        for chain in model
        for residue in chain
    )


def _is_protein_like_residue_name(residue_name: str) -> bool:
    return is_protein_component(residue_name)


def _is_nucleic_acid_like_residue_name(residue_name: str) -> bool:
    return is_nucleic_acid_component(residue_name)


def _is_hydrogen(atom: gemmi.Atom) -> bool:
    if atom.element.name.casefold() == "h":
        return True

    return atom.name.strip().upper().startswith("H")


def _residue_key(
    model: gemmi.Model,
    chain: gemmi.Chain,
    residue: gemmi.Residue,
) -> str:
    sequence_id = residue.seqid
    insertion_code = sequence_id.icode.strip() if sequence_id.icode else ""
    sequence_number = sequence_id.num
    insertion_suffix = f"^{insertion_code}" if insertion_code else ""

    return (
        f"model={model.name};"
        f"chain={chain.name};"
        f"residue={residue.name};"
        f"seqid={sequence_number}{insertion_suffix}"
    )


def _strip_cif_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_missing_text(value: str) -> bool:
    return _strip_cif_quotes(value).strip().casefold() in _MISSING_VALUES


def _deduplicate_strings(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        text = _strip_cif_quotes(value).strip()
        if not text or _is_missing_text(text):
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        output.append(text)

    return tuple(output)


__all__ = [
    "PdbRedoStructureCheckError",
    "PdbRedoStructureChecks",
    "read_pdb_redo_structure_checks",
]
