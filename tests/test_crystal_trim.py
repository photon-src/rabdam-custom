from pathlib import Path
import unittest

from crystal.symmetry import UnitCellParameters
from crystal.translate import (
    CartesianVector,
    TranslatedAtom,
    TranslatedCrystalBlock,
    UnitCellTranslationVectors,
)
from crystal.trim import (
    CartesianBounds,
    CrystalTrimError,
    bounds_from_prepared_atoms,
    expand_bounds,
    translated_atom_is_inside_bounds,
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


class CrystalTrimTests(unittest.TestCase):
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

    def test_empty_reference_atoms_raises(self) -> None:
        with self.assertRaises(CrystalTrimError):
            bounds_from_prepared_atoms(())

    def test_negative_padding_raises(self) -> None:
        with self.assertRaises(CrystalTrimError):
            expand_bounds(
                CartesianBounds(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
                -1.0,
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