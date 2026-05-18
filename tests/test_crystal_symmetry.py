import unittest
from pathlib import Path

import gemmi

from crystal.symmetry import (
    CrystalSymmetryError,
    apply_operation_without_wrapping,
    coordinates_as_tuples,
    expand_atoms_by_symmetry,
    normalize_space_group_name,
    unit_cell_from_metadata,
)
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from structure.models import PreparedAtom


def make_metadata(
    *,
    space_group: str | None = "P 1",
    unit_cell_a: float | None = 10.0,
    unit_cell_b: float | None = 10.0,
    unit_cell_c: float | None = 10.0,
    unit_cell_alpha: float | None = 90.0,
    unit_cell_beta: float | None = 90.0,
    unit_cell_gamma: float | None = 90.0,
) -> StructureMetadata:
    return StructureMetadata(
        source_path=Path("model.cif"),
        structure_id=None,
        file_format=StructureFileFormat.MMCIF,
        space_group=space_group,
        unit_cell_a=unit_cell_a,
        unit_cell_b=unit_cell_b,
        unit_cell_c=unit_cell_c,
        unit_cell_alpha=unit_cell_alpha,
        unit_cell_beta=unit_cell_beta,
        unit_cell_gamma=unit_cell_gamma,
    )


def make_prepared_atom(
    *,
    source_atom_index: int = 0,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
) -> PreparedAtom:
    record = AtomRecord(
        source_atom_index=source_atom_index,
        model_number=1,
        chain_id="A",
        residue_name="ALA",
        residue_number=1,
        insertion_code="",
        atom_name="CA",
        element="C",
        altloc="",
        x=x,
        y=y,
        z=z,
        occupancy=1.0,
        b_factor=10.0,
        atom_serial=source_atom_index + 1,
        record_type="ATOM",
    )

    return PreparedAtom(
        record=record,
        is_hydrogen=False,
        is_protein=True,
        is_nucleic_acid=False,
        is_solvent=False,
        is_hetatm=False,
    )


class CrystalSymmetryTests(unittest.TestCase):
    def assert_coordinates_almost_equal(
        self,
        actual: tuple[tuple[float, float, float], ...],
        expected: tuple[tuple[float, float, float], ...],
    ) -> None:
        self.assertEqual(len(actual), len(expected))

        for actual_coordinates, expected_coordinates in zip(actual, expected):
            for actual_value, expected_value in zip(
                actual_coordinates,
                expected_coordinates,
            ):
                self.assertAlmostEqual(actual_value, expected_value)

    def test_expands_identity_space_group_atom_to_identity_copy(self) -> None:
        expanded = expand_atoms_by_symmetry(
            atoms=(make_prepared_atom(),),
            metadata=make_metadata(space_group="P 1"),
        )

        self.assertEqual(expanded.operation_count, 1)
        self.assertEqual(expanded.space_group_name, "P 1")
        self.assertEqual(expanded.atoms[0].unit_cell_atom_index, 1)
        self.assertEqual(expanded.atoms[0].source_atom_index, 0)
        self.assertEqual(expanded.atoms[0].symmetry_operation, "x,y,z")
        self.assert_coordinates_almost_equal(
            coordinates_as_tuples(expanded),
            ((1.0, 2.0, 3.0),),
        )

    def test_expands_non_identity_space_group_without_wrapping(self) -> None:
        expanded = expand_atoms_by_symmetry(
            atoms=(make_prepared_atom(),),
            metadata=make_metadata(space_group="P 21 21 21"),
        )

        self.assertEqual(expanded.operation_count, 4)
        self.assertEqual(
            tuple(atom.symmetry_operation for atom in expanded.atoms),
            (
                "x,y,z",
                "-x+1/2,-y,z+1/2",
                "x+1/2,-y+1/2,-z",
                "-x,y+1/2,-z+1/2",
            ),
        )
        self.assert_coordinates_almost_equal(
            coordinates_as_tuples(expanded),
            (
                (1.0, 2.0, 3.0),
                (4.0, -2.0, 8.0),
                (6.0, 3.0, -3.0),
                (-1.0, 7.0, 2.0),
            ),
        )

    def test_applies_operation_without_wrapping_values_into_unit_cell(self) -> None:
        transformed = apply_operation_without_wrapping(
            gemmi.Op("x+1,y,z"),
            gemmi.Fractional(0.1, 0.2, 0.3),
        )

        self.assertAlmostEqual(transformed.x, 1.1)
        self.assertAlmostEqual(transformed.y, 0.2)
        self.assertAlmostEqual(transformed.z, 0.3)

    def test_normalizes_space_group_name_quotes_and_extra_spacing(self) -> None:
        self.assertEqual(
            normalize_space_group_name(" 'P 21 21 21' "),
            "P 21 21 21",
        )

    def test_rejects_empty_atom_list(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "empty atom list"):
            expand_atoms_by_symmetry(atoms=(), metadata=make_metadata())

    def test_rejects_missing_unit_cell_value(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "unit-cell parameters"):
            unit_cell_from_metadata(make_metadata(unit_cell_a=None))

    def test_rejects_non_numeric_unit_cell_value(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "non-numeric"):
            unit_cell_from_metadata(
                make_metadata(unit_cell_a="not-a-number")  # type: ignore[arg-type]
            )

    def test_rejects_invalid_unit_cell_length(self) -> None:
        with self.assertRaisesRegex(
            CrystalSymmetryError,
            "Invalid unit-cell lengths",
        ):
            unit_cell_from_metadata(make_metadata(unit_cell_a=-1.0))

    def test_rejects_invalid_unit_cell_angle(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "Invalid unit-cell angles"):
            unit_cell_from_metadata(make_metadata(unit_cell_alpha=180.0))

    def test_rejects_missing_space_group(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "space-group name"):
            expand_atoms_by_symmetry(
                atoms=(make_prepared_atom(),),
                metadata=make_metadata(space_group=None),
            )

    def test_rejects_unknown_space_group(self) -> None:
        with self.assertRaisesRegex(CrystalSymmetryError, "could not recognize"):
            expand_atoms_by_symmetry(
                atoms=(make_prepared_atom(),),
                metadata=make_metadata(space_group="not a space group"),
            )


if __name__ == "__main__":
    unittest.main()
