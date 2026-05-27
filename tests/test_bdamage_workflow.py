from pathlib import Path
import unittest

from rabdam.workflow import (
    BDamageWorkflowOptions,
    calculate_bdamage_for_prepared_structure,
)
from crystal.translate import TranslatedCrystalBlock, translate_expanded_unit_cell
from crystal.trim import ArrayTrimmedCrystalBlock, trim_translated_block_for_bdamage
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from packing.density import (
    calculate_bdamage_packing_density,
    packing_density_counts_as_tuple,
)
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationReport,
)


def make_prepared_atom(
    *,
    source_atom_index: int,
    atom_serial: int,
    x: float,
    y: float,
    z: float,
    b_factor: float,
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
        b_factor=b_factor,
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


def make_prepared_structure() -> PreparedStructure:
    atoms = (
        make_prepared_atom(
            source_atom_index=0,
            atom_serial=1,
            x=0.0,
            y=0.0,
            z=0.0,
            b_factor=10.0,
        ),
        make_prepared_atom(
            source_atom_index=1,
            atom_serial=2,
            x=2.0,
            y=0.0,
            z=0.0,
            b_factor=20.0,
        ),
        make_prepared_atom(
            source_atom_index=2,
            atom_serial=3,
            x=9.0,
            y=0.0,
            z=0.0,
            b_factor=30.0,
        ),
    )
    return PreparedStructure(
        cleaned_atoms=atoms,
        selected_atoms=atoms,
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
            input_atom_count=len(atoms),
            cleaned_atom_count=len(atoms),
            selected_atom_count=len(atoms),
            removed_hydrogen_count=0,
            removed_invalid_coordinate_count=0,
            removed_invalid_occupancy_count=0,
            removed_invalid_b_factor_count=0,
            removed_altloc_count=0,
        ),
    )


def make_workflow_options(
    *,
    materialize_translated_block: bool = False,
) -> BDamageWorkflowOptions:
    return BDamageWorkflowOptions(
        packing_density_threshold=5.0,
        minimum_window_size=1,
        materialize_translated_block=materialize_translated_block,
    )


class BDamageWorkflowTests(unittest.TestCase):
    def test_default_workflow_uses_fused_trim_without_materializing_translation(self) -> None:
        prepared_structure = make_prepared_structure()

        result = calculate_bdamage_for_prepared_structure(
            prepared_structure=prepared_structure,
            options=make_workflow_options(),
        )

        self.assertIsNone(result.translated_block)
        self.assertIsInstance(result.trimmed_block, ArrayTrimmedCrystalBlock)
        self.assertEqual(
            result.trimmed_block.original_atom_count,
            len(result.symmetry_expanded_structure.atoms) * 27,
        )
        self.assertEqual(
            result.packing_density_result.neighbour_atom_count,
            result.trimmed_block.atom_count,
        )

    def test_debug_workflow_can_materialize_full_translated_block(self) -> None:
        prepared_structure = make_prepared_structure()

        result = calculate_bdamage_for_prepared_structure(
            prepared_structure=prepared_structure,
            options=make_workflow_options(materialize_translated_block=True),
        )

        self.assertIsInstance(result.translated_block, TranslatedCrystalBlock)
        self.assertIsInstance(result.trimmed_block, ArrayTrimmedCrystalBlock)
        assert result.translated_block is not None
        self.assertEqual(
            len(result.translated_block.atoms),
            result.trimmed_block.original_atom_count,
        )

    def test_fast_workflow_packing_density_matches_object_path(self) -> None:
        prepared_structure = make_prepared_structure()
        result = calculate_bdamage_for_prepared_structure(
            prepared_structure=prepared_structure,
            options=make_workflow_options(),
        )
        object_translated_block = translate_expanded_unit_cell(
            result.symmetry_expanded_structure,
            translation_range=result.options.translation_range,
        )
        object_trimmed_block = trim_translated_block_for_bdamage(
            translated_block=object_translated_block,
            prepared_structure=prepared_structure,
            padding=result.options.packing_density_threshold,
        )

        object_packing_density = calculate_bdamage_packing_density(
            prepared_structure=prepared_structure,
            trimmed_block=object_trimmed_block,
            packing_density_threshold=result.options.packing_density_threshold,
        )

        self.assertEqual(
            result.trimmed_block.atom_count,
            len(object_trimmed_block.atoms),
        )
        self.assertEqual(
            packing_density_counts_as_tuple(result.packing_density_result),
            packing_density_counts_as_tuple(object_packing_density),
        )


if __name__ == "__main__":
    unittest.main()
