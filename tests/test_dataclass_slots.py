import pickle
import unittest
from pathlib import Path

import numpy as np

from bdamage.score import BDamageAtomInput, BDamageAtomResult, BDamageScoreResult
from rabdam.workflow import BDamageWorkflowOptions, BDamageWorkflowResult
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
)
from crystal.trim import (
    ArrayTrimmedCrystalBlock,
    CartesianBounds,
    TrimmedCrystalBlock,
)
from input.reader import AtomRecord, StructureData, StructureMetadata
from input.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)
from packing.density import PackingDensityAtomResult, PackingDensityResult
from structure.altlocs import AltlocSelectionResult
from structure.filters import AtomFilterCounts, AtomFilterResult
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationOptions,
    StructurePreparationReport,
)


def make_atom_record() -> AtomRecord:
    return AtomRecord(
        source_atom_index=0,
        model_number=1,
        chain_id="A",
        residue_name="ALA",
        residue_number=1,
        insertion_code="",
        atom_name="CA",
        element="C",
        altloc="",
        x=1.0,
        y=2.0,
        z=3.0,
        occupancy=1.0,
        b_factor=10.0,
        atom_serial=1,
        record_type="ATOM",
    )


class DataclassSlotsTests(unittest.TestCase):
    def assert_slotted_pickle_round_trip(self, instance: object) -> None:
        self.assertFalse(hasattr(instance, "__dict__"))

        restored = pickle.loads(pickle.dumps(instance))

        self.assertFalse(hasattr(restored, "__dict__"))
        self.assertEqual(restored, instance)

    def test_public_value_dataclasses_are_slotted_and_pickleable(self) -> None:
        atom = make_atom_record()
        metadata = StructureMetadata(
            source_path=Path("example.cif"),
            structure_id="example",
            file_format=StructureFileFormat.MMCIF,
            space_group="P 1",
            unit_cell_a=10.0,
            unit_cell_b=11.0,
            unit_cell_c=12.0,
            unit_cell_alpha=90.0,
            unit_cell_beta=90.0,
            unit_cell_gamma=90.0,
        )
        prepared_atom = PreparedAtom(
            record=atom,
            is_hydrogen=False,
            is_protein=True,
            is_nucleic_acid=False,
            is_solvent=False,
            is_hetatm=False,
        )
        preparation_report = StructurePreparationReport(
            input_atom_count=1,
            cleaned_atom_count=1,
            selected_atom_count=1,
            removed_hydrogen_count=0,
            removed_invalid_coordinate_count=0,
            removed_invalid_occupancy_count=0,
            removed_invalid_b_factor_count=0,
            removed_altloc_count=0,
        )
        prepared_structure = PreparedStructure(
            cleaned_atoms=(prepared_atom,),
            selected_atoms=(prepared_atom,),
            metadata=metadata,
            report=preparation_report,
        )
        unit_cell = UnitCellParameters(
            a=10.0,
            b=11.0,
            c=12.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
        )
        expanded_atom = SymmetryExpandedAtom(
            unit_cell_atom_index=1,
            source_atom_index=0,
            symmetry_operation_index=1,
            symmetry_operation="x,y,z",
            is_identity_symmetry_operation=True,
            x=1.0,
            y=2.0,
            z=3.0,
        )
        expanded_structure = SymmetryExpandedStructure(
            atoms=(expanded_atom,),
            unit_cell=unit_cell,
            space_group_name="P 1",
            operation_count=1,
        )
        vectors = UnitCellTranslationVectors(
            a=CartesianVector(10.0, 0.0, 0.0),
            b=CartesianVector(0.0, 11.0, 0.0),
            c=CartesianVector(0.0, 0.0, 12.0),
        )
        translated_atom = TranslatedAtom(
            translated_atom_index=1,
            unit_cell_atom_index=1,
            source_atom_index=0,
            symmetry_operation_index=1,
            is_identity_symmetry_operation=True,
            translation_a=0,
            translation_b=0,
            translation_c=0,
            x=1.0,
            y=2.0,
            z=3.0,
        )
        translated_block = TranslatedCrystalBlock(
            atoms=(translated_atom,),
            unit_cell=unit_cell,
            translation_vectors=vectors,
            translation_range=1,
            source_unit_cell_atom_count=1,
        )
        bounds = CartesianBounds(
            x_min=0.0,
            x_max=4.0,
            y_min=0.0,
            y_max=4.0,
            z_min=0.0,
            z_max=4.0,
        )
        trimmed_block = TrimmedCrystalBlock(
            atoms=(translated_atom,),
            reference_bounds=bounds,
            trim_bounds=bounds,
            padding=7.0,
            original_atom_count=1,
        )
        packing_atom_result = PackingDensityAtomResult(
            packing_density_atom_index=1,
            source_atom_index=0,
            atom_serial=1,
            neighbour_count=2,
        )
        packing_result = PackingDensityResult(
            atom_results=(packing_atom_result,),
            packing_density_threshold=7.0,
            selected_atom_count=1,
            neighbour_atom_count=1,
        )
        bdamage_input = BDamageAtomInput(
            bdamage_atom_index=1,
            source_atom_index=0,
            atom_serial=1,
            b_factor=10.0,
            packing_density=2,
        )
        bdamage_atom_result = BDamageAtomResult(
            bdamage_atom_index=1,
            source_atom_index=0,
            atom_serial=1,
            b_factor=10.0,
            packing_density=2,
            average_b_factor=10.0,
            bdamage=1.0,
            sorted_packing_density_index=1,
        )
        bdamage_score_result = BDamageScoreResult(
            atom_results=(bdamage_atom_result,),
            window_size=1,
            selected_atom_count=1,
        )
        workflow_options = BDamageWorkflowOptions()

        instances = (
            ResolvedStructureInput(
                original_input="example.cif",
                source_type=StructureSourceType.LOCAL_FILE,
                file_format=StructureFileFormat.MMCIF,
                local_path=Path("example.cif"),
            ),
            atom,
            metadata,
            StructureData(atoms=(atom,), metadata=metadata),
            StructurePreparationOptions(),
            prepared_atom,
            preparation_report,
            prepared_structure,
            AltlocSelectionResult(atoms=(atom,), removed_count=0),
            AtomFilterCounts(),
            AtomFilterResult(atoms=(atom,), counts=AtomFilterCounts()),
            unit_cell,
            expanded_atom,
            expanded_structure,
            CartesianVector(1.0, 2.0, 3.0),
            vectors,
            translated_atom,
            translated_block,
            bounds,
            trimmed_block,
            packing_atom_result,
            packing_result,
            bdamage_input,
            bdamage_atom_result,
            bdamage_score_result,
            workflow_options,
            BDamageWorkflowResult(
                prepared_structure=prepared_structure,
                symmetry_expanded_structure=expanded_structure,
                translated_block=None,
                trimmed_block=trimmed_block,
                packing_density_result=packing_result,
                bdamage_score_result=bdamage_score_result,
                options=workflow_options,
                window_size=1,
            ),
        )

        for instance in instances:
            with self.subTest(dataclass_type=type(instance).__name__):
                self.assert_slotted_pickle_round_trip(instance)

    def test_array_trimmed_block_is_slotted_and_pickleable(self) -> None:
        bounds = CartesianBounds(
            x_min=0.0,
            x_max=4.0,
            y_min=0.0,
            y_max=4.0,
            z_min=0.0,
            z_max=4.0,
        )
        vectors = UnitCellTranslationVectors(
            a=CartesianVector(10.0, 0.0, 0.0),
            b=CartesianVector(0.0, 11.0, 0.0),
            c=CartesianVector(0.0, 0.0, 12.0),
        )
        block = ArrayTrimmedCrystalBlock(
            coordinates=np.asarray([(1.0, 2.0, 3.0)], dtype=np.float64),
            source_atom_indices=np.asarray([0], dtype=np.int64),
            is_identity_symmetry_operation=np.asarray([True], dtype=np.bool_),
            translation_offsets=np.asarray([(0, 0, 0)], dtype=np.int64),
            reference_bounds=bounds,
            trim_bounds=bounds,
            padding=7.0,
            original_atom_count=1,
            translation_vectors=vectors,
            translation_range=1,
            source_unit_cell_atom_count=1,
        )

        self.assertFalse(hasattr(block, "__dict__"))

        restored = pickle.loads(pickle.dumps(block))

        self.assertFalse(hasattr(restored, "__dict__"))
        np.testing.assert_array_equal(restored.coordinates, block.coordinates)
        np.testing.assert_array_equal(
            restored.source_atom_indices,
            block.source_atom_indices,
        )
        np.testing.assert_array_equal(
            restored.is_identity_symmetry_operation,
            block.is_identity_symmetry_operation,
        )
        np.testing.assert_array_equal(
            restored.translation_offsets,
            block.translation_offsets,
        )
        self.assertEqual(restored.reference_bounds, block.reference_bounds)
        self.assertEqual(restored.trim_bounds, block.trim_bounds)
        self.assertEqual(restored.padding, block.padding)
        self.assertEqual(restored.original_atom_count, block.original_atom_count)


if __name__ == "__main__":
    unittest.main()
