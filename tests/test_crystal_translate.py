import math
import unittest

from crystal.symmetry import SymmetryExpandedAtom, SymmetryExpandedStructure, UnitCellParameters
from crystal.translate import (
    CrystalTranslationError,
    translate_expanded_unit_cell,
    translated_coordinates_as_tuples,
    translation_vector_for_offsets,
    unit_cell_translation_vectors,
)


def is_close_tuple(left, right, places: int = 9) -> bool:
    return all(round(abs(a - b), places) == 0 for a, b in zip(left, right))


def make_expanded_structure(
    *,
    atoms=(
        SymmetryExpandedAtom(
            unit_cell_atom_index=1,
            source_atom_index=0,
            symmetry_operation_index=1,
            symmetry_operation="x,y,z",
            is_identity_symmetry_operation=True,
            x=1.0,
            y=2.0,
            z=3.0,
        ),
    ),
    unit_cell=UnitCellParameters(
        a=10.0,
        b=20.0,
        c=30.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    ),
):
    return SymmetryExpandedStructure(
        atoms=tuple(atoms),
        unit_cell=unit_cell,
        space_group_name="P 1",
        operation_count=1,
    )


class CrystalTranslateTests(unittest.TestCase):
    def assert_coordinates_almost_equal(
        self,
        actual: tuple[float, float, float],
        expected: tuple[float, float, float],
    ) -> None:
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value)

    def test_orthogonal_translation_vectors(self) -> None:
        vectors = unit_cell_translation_vectors(
            UnitCellParameters(
                a=10.0,
                b=20.0,
                c=30.0,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
            )
        )

        self.assert_coordinates_almost_equal(
            (vectors.a.x, vectors.a.y, vectors.a.z),
            (10.0, 0.0, 0.0),
        )
        self.assert_coordinates_almost_equal(
            (vectors.b.x, vectors.b.y, vectors.b.z),
            (0.0, 20.0, 0.0),
        )
        self.assert_coordinates_almost_equal(
            (vectors.c.x, vectors.c.y, vectors.c.z),
            (0.0, 0.0, 30.0),
        )

    def test_skewed_translation_vectors_match_rabdam2_formula(self) -> None:
        unit_cell = UnitCellParameters(
            a=10.0,
            b=20.0,
            c=30.0,
            alpha=80.0,
            beta=75.0,
            gamma=70.0,
        )
        vectors = unit_cell_translation_vectors(unit_cell)

        alpha = math.radians(unit_cell.alpha)
        beta = math.radians(unit_cell.beta)
        gamma = math.radians(unit_cell.gamma)
        v = math.sqrt(
            1
            - math.cos(alpha) ** 2
            - math.cos(beta) ** 2
            - math.cos(gamma) ** 2
            + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
        )
        expected_c = (
            unit_cell.c * math.cos(beta),
            unit_cell.c * ((math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)),
            unit_cell.c * (v / math.sin(gamma)),
        )

        self.assert_coordinates_almost_equal(
            (vectors.c.x, vectors.c.y, vectors.c.z),
            expected_c,
        )

    def test_translate_expanded_unit_cell_generates_27_copies_by_default(self) -> None:
        block = translate_expanded_unit_cell(make_expanded_structure())

        self.assertEqual(len(block.atoms), 27)
        self.assertEqual(block.translation_range, 1)
        self.assertEqual(block.source_unit_cell_atom_count, 1)

        coords = translated_coordinates_as_tuples(block)
        self.assertTrue(any(is_close_tuple(coord, (1.0, 2.0, 3.0)) for coord in coords))
        self.assertTrue(any(is_close_tuple(coord, (11.0, 2.0, 3.0)) for coord in coords))
        self.assertTrue(any(is_close_tuple(coord, (1.0, -18.0, 33.0)) for coord in coords))

    def test_translate_order_matches_rabdam2_nested_offsets(self) -> None:
        block = translate_expanded_unit_cell(make_expanded_structure())

        first = block.atoms[0]
        self.assertEqual((first.translation_a, first.translation_b, first.translation_c), (-1, -1, -1))
        self.assertTrue(is_close_tuple((first.x, first.y, first.z), (-9.0, -18.0, -27.0)))

        second = block.atoms[1]
        self.assertEqual((second.translation_a, second.translation_b, second.translation_c), (-1, -1, 0))
        self.assertTrue(is_close_tuple((second.x, second.y, second.z), (-9.0, -18.0, 3.0)))

    def test_translation_range_zero_keeps_central_cell_only(self) -> None:
        block = translate_expanded_unit_cell(
            make_expanded_structure(),
            translation_range=0,
        )

        self.assertEqual(len(block.atoms), 1)
        atom = block.atoms[0]
        self.assertEqual((atom.translation_a, atom.translation_b, atom.translation_c), (0, 0, 0))
        self.assertTrue(is_close_tuple((atom.x, atom.y, atom.z), (1.0, 2.0, 3.0)))

    def test_metadata_is_preserved_from_symmetry_expanded_atom(self) -> None:
        source = SymmetryExpandedAtom(
            unit_cell_atom_index=7,
            source_atom_index=3,
            symmetry_operation_index=2,
            symmetry_operation="-x,y,z",
            is_identity_symmetry_operation=False,
            x=1.0,
            y=2.0,
            z=3.0,
        )
        block = translate_expanded_unit_cell(make_expanded_structure(atoms=(source,)), translation_range=0)
        translated = block.atoms[0]

        self.assertEqual(translated.unit_cell_atom_index, 7)
        self.assertEqual(translated.source_atom_index, 3)
        self.assertEqual(translated.symmetry_operation_index, 2)
        self.assertFalse(translated.is_identity_symmetry_operation)

    def test_translation_vector_for_offsets_combines_abc_vectors(self) -> None:
        vectors = unit_cell_translation_vectors(
            UnitCellParameters(
                a=10.0,
                b=20.0,
                c=30.0,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
            )
        )
        shift = translation_vector_for_offsets(
            vectors,
            a_offset=1,
            b_offset=-1,
            c_offset=1,
        )

        self.assert_coordinates_almost_equal(
            (shift.x, shift.y, shift.z),
            (10.0, -20.0, 30.0),
        )

    def test_empty_atom_list_raises(self) -> None:
        with self.assertRaises(CrystalTranslationError):
            translate_expanded_unit_cell(make_expanded_structure(atoms=()))

    def test_negative_translation_range_raises(self) -> None:
        with self.assertRaises(CrystalTranslationError):
            translate_expanded_unit_cell(
                make_expanded_structure(),
                translation_range=-1,
            )

    def test_invalid_unit_cell_length_raises(self) -> None:
        with self.assertRaisesRegex(CrystalTranslationError, "unit-cell lengths"):
            unit_cell_translation_vectors(
                UnitCellParameters(
                    a=0.0,
                    b=20.0,
                    c=30.0,
                    alpha=90.0,
                    beta=90.0,
                    gamma=90.0,
                )
            )

    def test_invalid_unit_cell_angle_raises(self) -> None:
        with self.assertRaisesRegex(CrystalTranslationError, "unit-cell angles"):
            unit_cell_translation_vectors(
                UnitCellParameters(
                    a=10.0,
                    b=20.0,
                    c=30.0,
                    alpha=90.0,
                    beta=180.0,
                    gamma=90.0,
                )
            )

    def test_invalid_unit_cell_geometry_raises(self) -> None:
        with self.assertRaisesRegex(CrystalTranslationError, "unit-cell geometry"):
            unit_cell_translation_vectors(
                UnitCellParameters(
                    a=10.0,
                    b=20.0,
                    c=30.0,
                    alpha=10.0,
                    beta=10.0,
                    gamma=170.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
