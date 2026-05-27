from pathlib import Path
import unittest

import numpy as np

from crystal.symmetry import (
    SymmetryExpandedAtom,
    SymmetryExpandedStructure,
    UnitCellParameters,
)
from crystal.translate import (
    CartesianVector,
    TranslatedAtom,
    TranslatedCrystalBlock,
    UnitCellTranslationVectors,
    translate_expanded_unit_cell,
)
from crystal.trim import (
    ArrayTrimmedCrystalBlock,
    CartesianBounds,
    CrystalTrimError,
    bounds_from_prepared_atoms,
    expand_bounds,
    translated_atom_is_inside_bounds,
    trim_expanded_unit_cell_to_reference_atoms,
    trim_translated_block_for_bdamage,
    trim_translated_block_to_reference_atoms,
    trimmed_coordinates_as_tuples,
)
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationReport,
)


def make_prepared_atom(
    *,
    source_atom_index: int,
    x: float,
    y: float,
    z: float,
    atom_name: str = "CA",
    residue_name: str = "ALA",
) -> PreparedAtom:
    record = AtomRecord(
        source_atom_index=source_atom_index,
        model_number=1,
        chain_id="A",
        residue_name=residue_name,
        residue_number=1,
        insertion_code="",
        atom_name=atom_name,
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


def make_translated_atom(
    *,
    translated_atom_index: int,
    x: float,
    y: float,
    z: float,
) -> TranslatedAtom:
    return TranslatedAtom(
        translated_atom_index=translated_atom_index,
        unit_cell_atom_index=translated_atom_index,
        source_atom_index=translated_atom_index - 1,
        symmetry_operation_index=1,
        is_identity_symmetry_operation=True,
        translation_a=0,
        translation_b=0,
        translation_c=0,
        x=x,
        y=y,
        z=z,
    )


def make_translated_block(atoms: tuple[TranslatedAtom, ...]) -> TranslatedCrystalBlock:
    unit_cell = UnitCellParameters(
        a=10.0,
        b=20.0,
        c=30.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )
    vectors = UnitCellTranslationVectors(
        a=CartesianVector(10.0, 0.0, 0.0),
        b=CartesianVector(0.0, 20.0, 0.0),
        c=CartesianVector(0.0, 0.0, 30.0),
    )
    return TranslatedCrystalBlock(
        atoms=atoms,
        unit_cell=unit_cell,
        translation_vectors=vectors,
        translation_range=1,
        source_unit_cell_atom_count=len(atoms),
    )


def make_expanded_structure(
    *,
    atoms: tuple[SymmetryExpandedAtom, ...],
    unit_cell: UnitCellParameters | None = None,
) -> SymmetryExpandedStructure:
    return SymmetryExpandedStructure(
        atoms=atoms,
        unit_cell=unit_cell
        or UnitCellParameters(
            a=10.0,
            b=20.0,
            c=30.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        ),
        space_group_name="P 1",
        operation_count=1,
    )


def make_expanded_atom(
    *,
    unit_cell_atom_index: int,
    source_atom_index: int,
    x: float,
    y: float,
    z: float,
    symmetry_operation_index: int = 1,
    symmetry_operation: str = "x,y,z",
    is_identity_symmetry_operation: bool = True,
) -> SymmetryExpandedAtom:
    return SymmetryExpandedAtom(
        unit_cell_atom_index=unit_cell_atom_index,
        source_atom_index=source_atom_index,
        symmetry_operation_index=symmetry_operation_index,
        symmetry_operation=symmetry_operation,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        x=x,
        y=y,
        z=z,
    )


class CrystalTrimTests(unittest.TestCase):
    def assert_fused_block_matches_object_block(
        self,
        *,
        fused: ArrayTrimmedCrystalBlock,
        object_block,
    ) -> None:
        self.assertEqual(fused.original_atom_count, object_block.original_atom_count)
        self.assertEqual(fused.reference_bounds, object_block.reference_bounds)
        self.assertEqual(fused.trim_bounds, object_block.trim_bounds)
        self.assertEqual(fused.padding, object_block.padding)
        self.assertEqual(
            trimmed_coordinates_as_tuples(fused),
            trimmed_coordinates_as_tuples(object_block),
        )
        self.assertEqual(
            tuple(fused.source_atom_indices.tolist()),
            tuple(atom.source_atom_index for atom in object_block.atoms),
        )
        self.assertEqual(
            tuple(bool(flag) for flag in fused.is_identity_symmetry_operation.tolist()),
            tuple(atom.is_identity_symmetry_operation for atom in object_block.atoms),
        )
        self.assertEqual(
            tuple(tuple(row) for row in fused.translation_offsets.tolist()),
            tuple(
                (atom.translation_a, atom.translation_b, atom.translation_c)
                for atom in object_block.atoms
            ),
        )

    def test_bounds_from_prepared_atoms(self) -> None:
        atoms = (
            make_prepared_atom(source_atom_index=0, x=1.0, y=4.0, z=-2.0),
            make_prepared_atom(source_atom_index=1, x=3.0, y=-5.0, z=8.0),
            make_prepared_atom(source_atom_index=2, x=-1.0, y=7.0, z=0.0),
        )

        bounds = bounds_from_prepared_atoms(atoms)

        self.assertEqual(
            bounds,
            CartesianBounds(
                x_min=-1.0,
                x_max=3.0,
                y_min=-5.0,
                y_max=7.0,
                z_min=-2.0,
                z_max=8.0,
            ),
        )

    def test_expand_bounds_adds_padding_to_each_side(self) -> None:
        bounds = CartesianBounds(
            x_min=1.0,
            x_max=3.0,
            y_min=-5.0,
            y_max=7.0,
            z_min=-2.0,
            z_max=8.0,
        )

        expanded = expand_bounds(bounds, 2.5)

        self.assertEqual(
            expanded,
            CartesianBounds(
                x_min=-1.5,
                x_max=5.5,
                y_min=-7.5,
                y_max=9.5,
                z_min=-4.5,
                z_max=10.5,
            ),
        )

    def test_translated_atom_is_inside_bounds_is_inclusive(self) -> None:
        bounds = CartesianBounds(
            x_min=0.0,
            x_max=10.0,
            y_min=0.0,
            y_max=10.0,
            z_min=0.0,
            z_max=10.0,
        )

        self.assertTrue(
            translated_atom_is_inside_bounds(
                make_translated_atom(translated_atom_index=1, x=0.0, y=10.0, z=5.0),
                bounds,
            )
        )
        self.assertFalse(
            translated_atom_is_inside_bounds(
                make_translated_atom(translated_atom_index=2, x=-0.001, y=5.0, z=5.0),
                bounds,
            )
        )

    def test_trim_translated_block_to_reference_atoms_keeps_atoms_inside_padded_box(self) -> None:
        reference_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
            make_prepared_atom(source_atom_index=1, x=10.0, y=10.0, z=10.0),
        )
        translated_block = make_translated_block(
            (
                make_translated_atom(translated_atom_index=1, x=-1.0, y=5.0, z=5.0),
                make_translated_atom(translated_atom_index=2, x=5.0, y=5.0, z=5.0),
                make_translated_atom(translated_atom_index=3, x=11.0, y=10.0, z=10.0),
                make_translated_atom(translated_atom_index=4, x=12.1, y=5.0, z=5.0),
                make_translated_atom(translated_atom_index=5, x=5.0, y=-2.1, z=5.0),
            )
        )

        trimmed = trim_translated_block_to_reference_atoms(
            translated_block=translated_block,
            reference_atoms=reference_atoms,
            padding=2.0,
        )

        self.assertEqual(trimmed.original_atom_count, 5)
        self.assertEqual([atom.translated_atom_index for atom in trimmed.atoms], [1, 2, 3])
        self.assertEqual(
            trimmed.trim_bounds,
            CartesianBounds(
                x_min=-2.0,
                x_max=12.0,
                y_min=-2.0,
                y_max=12.0,
                z_min=-2.0,
                z_max=12.0,
            ),
        )
        self.assertEqual(
            trimmed_coordinates_as_tuples(trimmed),
            ((-1.0, 5.0, 5.0), (5.0, 5.0, 5.0), (11.0, 10.0, 10.0)),
        )

    def test_trim_for_bdamage_uses_selected_atoms_as_reference_bounds(self) -> None:
        cleaned_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
            make_prepared_atom(source_atom_index=1, x=100.0, y=100.0, z=100.0),
        )
        selected_atoms = (cleaned_atoms[0],)
        prepared_structure = PreparedStructure(
            cleaned_atoms=cleaned_atoms,
            selected_atoms=selected_atoms,
            metadata=StructureMetadata(
                source_path=Path("test.pdb"),
                structure_id=None,
                file_format=StructureFileFormat.PDB,
                space_group="P 1",
                unit_cell_a=10.0,
                unit_cell_b=20.0,
                unit_cell_c=30.0,
                unit_cell_alpha=90.0,
                unit_cell_beta=90.0,
                unit_cell_gamma=90.0,
            ),
            report=StructurePreparationReport(
                input_atom_count=2,
                cleaned_atom_count=2,
                selected_atom_count=1,
                removed_hydrogen_count=0,
                removed_invalid_coordinate_count=0,
                removed_invalid_occupancy_count=0,
                removed_invalid_b_factor_count=0,
                removed_altloc_count=0,
            ),
        )
        translated_block = make_translated_block(
            (
                make_translated_atom(translated_atom_index=1, x=0.5, y=0.5, z=0.5),
                make_translated_atom(translated_atom_index=2, x=100.0, y=100.0, z=100.0),
            )
        )

        trimmed = trim_translated_block_for_bdamage(
            translated_block=translated_block,
            prepared_structure=prepared_structure,
            padding=1.0,
        )

        self.assertEqual([atom.translated_atom_index for atom in trimmed.atoms], [1])
        self.assertEqual(
            trimmed.reference_bounds,
            CartesianBounds(
                x_min=0.0,
                x_max=0.0,
                y_min=0.0,
                y_max=0.0,
                z_min=0.0,
                z_max=0.0,
            ),
        )

    def test_fused_trim_matches_object_path_default_range(self) -> None:
        expanded_structure = make_expanded_structure(
            atoms=(
                make_expanded_atom(
                    unit_cell_atom_index=1,
                    source_atom_index=0,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                ),
                make_expanded_atom(
                    unit_cell_atom_index=2,
                    source_atom_index=1,
                    x=8.0,
                    y=3.0,
                    z=3.0,
                    symmetry_operation_index=2,
                    symmetry_operation="-x,y,z",
                    is_identity_symmetry_operation=False,
                ),
                make_expanded_atom(
                    unit_cell_atom_index=3,
                    source_atom_index=2,
                    x=-9.0,
                    y=2.0,
                    z=3.0,
                ),
            ),
        )
        reference_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
            make_prepared_atom(source_atom_index=1, x=10.0, y=5.0, z=5.0),
        )
        object_block = trim_translated_block_to_reference_atoms(
            translated_block=translate_expanded_unit_cell(expanded_structure),
            reference_atoms=reference_atoms,
            padding=3.0,
        )

        fused = trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded_structure,
            reference_atoms=reference_atoms,
            padding=3.0,
        )

        self.assert_fused_block_matches_object_block(
            fused=fused,
            object_block=object_block,
        )

    def test_fused_trim_matches_object_path_for_translation_range_zero(self) -> None:
        expanded_structure = make_expanded_structure(
            atoms=(
                make_expanded_atom(
                    unit_cell_atom_index=1,
                    source_atom_index=0,
                    x=0.0,
                    y=5.0,
                    z=10.0,
                ),
                make_expanded_atom(
                    unit_cell_atom_index=2,
                    source_atom_index=1,
                    x=11.0,
                    y=5.0,
                    z=10.0,
                ),
            ),
        )
        reference_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
            make_prepared_atom(source_atom_index=1, x=10.0, y=10.0, z=10.0),
        )
        object_block = trim_translated_block_to_reference_atoms(
            translated_block=translate_expanded_unit_cell(
                expanded_structure,
                translation_range=0,
            ),
            reference_atoms=reference_atoms,
            padding=0.0,
        )

        fused = trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded_structure,
            reference_atoms=reference_atoms,
            padding=0.0,
            translation_range=0,
        )

        self.assert_fused_block_matches_object_block(
            fused=fused,
            object_block=object_block,
        )
        self.assertEqual(fused.atom_count, 1)
        np.testing.assert_allclose(fused.coordinates[0], np.asarray((0.0, 5.0, 10.0)))

    def test_fused_trim_matches_object_path_for_non_orthogonal_unit_cell(self) -> None:
        expanded_structure = make_expanded_structure(
            atoms=(
                make_expanded_atom(
                    unit_cell_atom_index=1,
                    source_atom_index=0,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                ),
                make_expanded_atom(
                    unit_cell_atom_index=2,
                    source_atom_index=1,
                    x=7.0,
                    y=8.0,
                    z=9.0,
                ),
            ),
            unit_cell=UnitCellParameters(
                a=10.0,
                b=12.0,
                c=14.0,
                alpha=80.0,
                beta=75.0,
                gamma=70.0,
            ),
        )
        reference_atoms = (
            make_prepared_atom(source_atom_index=0, x=-5.0, y=-5.0, z=-5.0),
            make_prepared_atom(source_atom_index=1, x=15.0, y=15.0, z=15.0),
        )
        object_block = trim_translated_block_to_reference_atoms(
            translated_block=translate_expanded_unit_cell(expanded_structure),
            reference_atoms=reference_atoms,
            padding=2.0,
        )

        fused = trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded_structure,
            reference_atoms=reference_atoms,
            padding=2.0,
        )

        self.assert_fused_block_matches_object_block(
            fused=fused,
            object_block=object_block,
        )

    def test_empty_reference_atoms_raises(self) -> None:
        with self.assertRaises(CrystalTrimError):
            bounds_from_prepared_atoms(())

    def test_negative_padding_raises(self) -> None:
        with self.assertRaises(CrystalTrimError):
            expand_bounds(
                CartesianBounds(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
                -1.0,
            )

    def test_trim_negative_padding_raises(self) -> None:
        reference_atoms = (make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),)
        translated_block = make_translated_block(
            (make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),)
        )

        with self.assertRaises(CrystalTrimError):
            trim_translated_block_to_reference_atoms(
                translated_block=translated_block,
                reference_atoms=reference_atoms,
                padding=-1.0,
            )

    def test_empty_translated_block_raises(self) -> None:
        reference_atoms = (make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),)
        translated_block = make_translated_block(())

        with self.assertRaises(CrystalTrimError):
            trim_translated_block_to_reference_atoms(
                translated_block=translated_block,
                reference_atoms=reference_atoms,
                padding=1.0,
            )

    def test_trimmed_empty_result_raises(self) -> None:
        reference_atoms = (make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),)
        translated_block = make_translated_block(
            (make_translated_atom(translated_atom_index=1, x=100.0, y=100.0, z=100.0),)
        )

        with self.assertRaises(CrystalTrimError):
            trim_translated_block_to_reference_atoms(
                translated_block=translated_block,
                reference_atoms=reference_atoms,
                padding=1.0,
            )


if __name__ == "__main__":
    unittest.main()
