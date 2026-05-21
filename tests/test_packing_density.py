from pathlib import Path
import math
import unittest

from crystal.symmetry import UnitCellParameters
from crystal.translate import (
    CartesianVector,
    TranslatedAtom,
    TranslatedCrystalBlock,
    UnitCellTranslationVectors,
)
from crystal.trim import CartesianBounds, TrimmedCrystalBlock
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from packing.density import (
    PackingDensityError,
    _count_neighbours_within_threshold_squared,
    calculate_bdamage_packing_density,
    calculate_packing_density,
    packing_density_counts_as_tuple,
    squared_distance_to_translated_atom,
)
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
    atom_serial: int | None = None,
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
        atom_serial=atom_serial,
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
    source_atom_index: int | None = None,
    is_identity_symmetry_operation: bool = True,
) -> TranslatedAtom:
    return TranslatedAtom(
        translated_atom_index=translated_atom_index,
        unit_cell_atom_index=translated_atom_index,
        source_atom_index=(
            translated_atom_index - 1
            if source_atom_index is None
            else source_atom_index
        ),
        symmetry_operation_index=1,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        translation_a=0,
        translation_b=0,
        translation_c=0,
        x=x,
        y=y,
        z=z,
    )


def make_trimmed_block(atoms: tuple[TranslatedAtom, ...]) -> TrimmedCrystalBlock:
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
    translated_block = TranslatedCrystalBlock(
        atoms=atoms,
        unit_cell=unit_cell,
        translation_vectors=vectors,
        translation_range=1,
        source_unit_cell_atom_count=len(atoms),
    )
    bounds = CartesianBounds(
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        z_min=-10.0,
        z_max=10.0,
    )
    return TrimmedCrystalBlock(
        atoms=translated_block.atoms,
        reference_bounds=bounds,
        trim_bounds=bounds,
        padding=7.5,
        original_atom_count=len(atoms),
    )


def make_prepared_structure(selected_atoms: tuple[PreparedAtom, ...]) -> PreparedStructure:
    return PreparedStructure(
        cleaned_atoms=selected_atoms,
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
            input_atom_count=len(selected_atoms),
            cleaned_atom_count=len(selected_atoms),
            selected_atom_count=len(selected_atoms),
            removed_hydrogen_count=0,
            removed_invalid_coordinate_count=0,
            removed_invalid_occupancy_count=0,
            removed_invalid_b_factor_count=0,
            removed_altloc_count=0,
        ),
    )


class PackingDensityTests(unittest.TestCase):
    def test_squared_distance_to_translated_atom(self) -> None:
        neighbour = make_translated_atom(
            translated_atom_index=1,
            x=4.0,
            y=6.0,
            z=8.0,
        )

        distance_squared = squared_distance_to_translated_atom(
            selected_x=1.0,
            selected_y=2.0,
            selected_z=3.0,
            neighbour_atom=neighbour,
        )

        self.assertEqual(distance_squared, 50.0)

    def test_count_neighbours_excludes_atoms_on_threshold_boundary(self) -> None:
        selected_atom = make_prepared_atom(
            source_atom_index=0,
            x=0.0,
            y=0.0,
            z=0.0,
        )
        neighbours = (
            make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),
            make_translated_atom(translated_atom_index=2, x=3.0, y=4.0, z=0.0),
            make_translated_atom(translated_atom_index=3, x=5.1, y=0.0, z=0.0),
        )

        count = _count_neighbours_within_threshold_squared(
            selected_atom=selected_atom,
            neighbour_atoms=neighbours,
            threshold_squared=25.0,
        )

        self.assertEqual(count, 1)

    def test_calculate_packing_density_counts_each_selected_atom_after_self_correction(self) -> None:
        selected_atoms = (
            make_prepared_atom(source_atom_index=0, atom_serial=101, x=0.0, y=0.0, z=0.0),
            make_prepared_atom(source_atom_index=1, atom_serial=102, x=10.0, y=0.0, z=0.0),
        )
        neighbour_atoms = (
            make_translated_atom(
                translated_atom_index=1,
                source_atom_index=0,
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            make_translated_atom(translated_atom_index=2, x=2.0, y=2.0, z=0.0),
            make_translated_atom(
                translated_atom_index=3,
                source_atom_index=1,
                x=10.0,
                y=0.0,
                z=0.0,
            ),
            make_translated_atom(translated_atom_index=4, x=13.0, y=0.0, z=0.0),
        )

        result = calculate_packing_density(
            selected_atoms=selected_atoms,
            neighbour_atoms=neighbour_atoms,
            packing_density_threshold=5.0,
        )

        self.assertEqual(result.selected_atom_count, 2)
        self.assertEqual(result.neighbour_atom_count, 4)
        self.assertEqual(result.packing_density_threshold, 5.0)
        self.assertEqual(packing_density_counts_as_tuple(result), (1, 1))
        self.assertEqual(result.atom_results[0].packing_density_atom_index, 1)
        self.assertEqual(result.atom_results[0].source_atom_index, 0)
        self.assertEqual(result.atom_results[0].atom_serial, 101)
        self.assertEqual(result.atom_results[1].packing_density_atom_index, 2)
        self.assertEqual(result.atom_results[1].source_atom_index, 1)
        self.assertEqual(result.atom_results[1].atom_serial, 102)

    def test_calculate_packing_density_requires_counted_self_copy(self) -> None:
        selected_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
        )
        neighbour_atoms = (
            make_translated_atom(
                translated_atom_index=1,
                source_atom_index=1,
                x=1.0,
                y=0.0,
                z=0.0,
            ),
            make_translated_atom(
                translated_atom_index=2,
                source_atom_index=2,
                x=2.0,
                y=0.0,
                z=0.0,
            ),
            make_translated_atom(
                translated_atom_index=3,
                source_atom_index=3,
                x=3.0,
                y=0.0,
                z=0.0,
            ),
        )

        with self.assertRaisesRegex(
            PackingDensityError,
            "central-cell copy",
        ):
            calculate_packing_density(
                selected_atoms=selected_atoms,
                neighbour_atoms=neighbour_atoms,
                packing_density_threshold=5.0,
            )

    def test_calculate_bdamage_packing_density_wrapper_uses_selected_atoms_and_trimmed_block(self) -> None:
        selected_atoms = (
            make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
        )
        prepared_structure = make_prepared_structure(selected_atoms)
        trimmed_block = make_trimmed_block(
            (
                make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),
                make_translated_atom(translated_atom_index=2, x=0.0, y=6.0, z=0.0),
            )
        )

        result = calculate_bdamage_packing_density(
            prepared_structure=prepared_structure,
            trimmed_block=trimmed_block,
            packing_density_threshold=7.5,
        )

        self.assertEqual(packing_density_counts_as_tuple(result), (1,))
        self.assertEqual(result.selected_atom_count, 1)
        self.assertEqual(result.neighbour_atom_count, 2)

    def test_non_positive_threshold_raises(self) -> None:
        with self.assertRaises(PackingDensityError):
            calculate_packing_density(
                selected_atoms=(make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),),
                neighbour_atoms=(make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),),
                packing_density_threshold=0.0,
            )

        with self.assertRaises(PackingDensityError):
            calculate_packing_density(
                selected_atoms=(make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),),
                neighbour_atoms=(make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),),
                packing_density_threshold=-1.0,
            )

    def test_non_finite_threshold_raises(self) -> None:
        selected_atoms = (make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),)
        neighbour_atoms = (make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),)

        for threshold in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(PackingDensityError):
                    calculate_packing_density(
                        selected_atoms=selected_atoms,
                        neighbour_atoms=neighbour_atoms,
                        packing_density_threshold=threshold,
                    )

    def test_empty_selected_atoms_raises(self) -> None:
        with self.assertRaises(PackingDensityError):
            calculate_packing_density(
                selected_atoms=(),
                neighbour_atoms=(make_translated_atom(translated_atom_index=1, x=0.0, y=0.0, z=0.0),),
                packing_density_threshold=7.5,
            )

    def test_empty_neighbour_atoms_raises(self) -> None:
        with self.assertRaises(PackingDensityError):
            calculate_packing_density(
                selected_atoms=(make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),),
                neighbour_atoms=(),
                packing_density_threshold=7.5,
            )

    def test_negative_threshold_squared_raises(self) -> None:
        with self.assertRaises(PackingDensityError):
            _count_neighbours_within_threshold_squared(
                selected_atom=make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
                neighbour_atoms=(),
                threshold_squared=-1.0,
            )

    def test_non_finite_threshold_squared_raises(self) -> None:
        for threshold_squared in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold_squared=threshold_squared):
                with self.assertRaises(PackingDensityError):
                    _count_neighbours_within_threshold_squared(
                        selected_atom=make_prepared_atom(source_atom_index=0, x=0.0, y=0.0, z=0.0),
                        neighbour_atoms=(),
                        threshold_squared=threshold_squared,
                    )


if __name__ == "__main__":
    unittest.main()
