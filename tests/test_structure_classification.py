import unittest

from input.reader import AtomRecord
from structure.classify import classify_atom
from structure.models import StructurePreparationOptions
from structure.selection import select_bdamage_atoms


def make_atom(
    residue_name: str,
    *,
    record_type: str = "ATOM",
    atom_serial: int = 1,
) -> AtomRecord:
    return AtomRecord(
        source_atom_index=atom_serial - 1,
        model_number=1,
        chain_id="A",
        residue_name=residue_name,
        residue_number=1,
        insertion_code="",
        atom_name="CA",
        element="C",
        altloc="",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        b_factor=10.0,
        atom_serial=atom_serial,
        record_type=record_type,
    )


class ClassificationTests(unittest.TestCase):
    def classify(self, residue_name: str, record_type: str = "ATOM"):
        return classify_atom(make_atom(residue_name, record_type=record_type))

    def test_standard_components_keep_existing_classification(self) -> None:
        self.assertTrue(self.classify("ALA").is_protein)
        self.assertTrue(self.classify("A").is_nucleic_acid)
        self.assertTrue(self.classify("HOH", record_type="HETATM").is_solvent)

    def test_gemmi_tabulated_protein_like_components(self) -> None:
        for residue_name in ("MSE", "SEC", "PYL", "DAS", "DGL"):
            with self.subTest(residue_name=residue_name):
                self.assertTrue(self.classify(residue_name).is_protein)

    def test_explicit_protein_like_overrides(self) -> None:
        for residue_name in ("HID", "HIE", "HIP", "ASH", "GLH"):
            with self.subTest(residue_name=residue_name):
                self.assertTrue(self.classify(residue_name).is_protein)

    def test_gemmi_tabulated_nucleic_acid_components(self) -> None:
        for residue_name in ("PSU", "OMG"):
            with self.subTest(residue_name=residue_name):
                self.assertTrue(self.classify(residue_name).is_nucleic_acid)

    def test_explicit_nucleic_acid_overrides(self) -> None:
        for residue_name in ("5MC", "YYG"):
            with self.subTest(residue_name=residue_name):
                self.assertTrue(self.classify(residue_name).is_nucleic_acid)

    def test_unknown_ligand_is_not_polymer_like(self) -> None:
        atom = self.classify("LIG", record_type="HETATM")

        self.assertFalse(atom.is_protein)
        self.assertFalse(atom.is_nucleic_acid)


class SelectionTests(unittest.TestCase):
    def selected_residue_names(
        self,
        residue_names: tuple[str, ...],
        *,
        record_type: str = "ATOM",
        options: StructurePreparationOptions = StructurePreparationOptions(),
    ) -> tuple[str, ...]:
        atoms = tuple(
            classify_atom(
                make_atom(
                    residue_name,
                    record_type=record_type,
                    atom_serial=index,
                )
            )
            for index, residue_name in enumerate(residue_names, start=1)
        )

        return tuple(
            atom.record.residue_name
            for atom in select_bdamage_atoms(atoms, options)
        )

    def test_protein_atom_records_are_selected_by_default(self) -> None:
        self.assertEqual(self.selected_residue_names(("ALA",)), ("ALA",))

    def test_protein_like_hetatm_records_are_excluded_by_default(self) -> None:
        self.assertEqual(
            self.selected_residue_names(("MSE",), record_type="HETATM"),
            (),
        )

    def test_protein_like_hetatm_records_can_be_selected(self) -> None:
        options = StructurePreparationOptions(
            include_protein_like_hetatm_in_selection=True
        )

        self.assertEqual(
            self.selected_residue_names(
                ("MSE",),
                record_type="HETATM",
                options=options,
            ),
            ("MSE",),
        )

    def test_component_remove_overrides_protein_like_hetatm_option(self) -> None:
        options = StructurePreparationOptions(
            include_protein_like_hetatm_in_selection=True,
            remove_component_names=frozenset({"MSE"}),
        )

        self.assertEqual(
            self.selected_residue_names(
                ("MSE",),
                record_type="HETATM",
                options=options,
            ),
            (),
        )

    def test_component_add_overrides_default_protein_like_hetatm_exclusion(self) -> None:
        options = StructurePreparationOptions(
            include_protein_like_hetatm_in_selection=False,
            add_component_names=frozenset({"MSE"}),
        )

        self.assertEqual(
            self.selected_residue_names(
                ("MSE",),
                record_type="HETATM",
                options=options,
            ),
            ("MSE",),
        )

    def test_non_protein_hetatm_records_follow_hetatm_option(self) -> None:
        self.assertEqual(
            self.selected_residue_names(("LIG",), record_type="HETATM"),
            (),
        )

        options = StructurePreparationOptions(include_hetatm_in_selection=True)

        self.assertEqual(
            self.selected_residue_names(
                ("LIG",),
                record_type="HETATM",
                options=options,
            ),
            ("LIG",),
        )


if __name__ == "__main__":
    unittest.main()
