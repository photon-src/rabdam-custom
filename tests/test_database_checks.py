from pathlib import Path
import tempfile
import unittest

from database.checks import read_pdb_redo_structure_checks
from database.discover import PdbRedoCandidate


def write_candidate_files(root: Path, final_cif_text: str) -> PdbRedoCandidate:
    root.mkdir(parents=True, exist_ok=True)
    final_cif_path = root / "1abc_final.cif"
    final_cif_path.write_text(final_cif_text, encoding="utf-8")
    data_json_path = root / "data.json"
    data_json_path.write_text("{}", encoding="utf-8")

    return PdbRedoCandidate(
        pdb_id="1abc",
        entry_dir=root,
        final_cif_path=final_cif_path,
        data_json_path=data_json_path,
    )


def mmcif_text(
    *,
    methods: tuple[str, ...] = ("X-RAY DIFFRACTION",),
    entity_polymer_types: tuple[str, ...] | None = ("polypeptide(L)",),
    atom_rows: tuple[str, ...],
) -> str:
    lines = ["data_1abc"]

    if len(methods) == 1:
        lines.append(f"_exptl.method {_quote_cif(methods[0])}")
    elif methods:
        lines.extend(("loop_", "_exptl.method"))
        lines.extend(_quote_cif(method) for method in methods)
        lines.append("#")

    if entity_polymer_types is not None:
        lines.extend(("loop_", "_entity_poly.entity_id", "_entity_poly.type"))
        for entity_id, polymer_type in enumerate(entity_polymer_types, start=1):
            lines.append(f"{entity_id} {_quote_cif(polymer_type)}")
        lines.append("#")

    lines.extend(
        (
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        )
    )
    lines.extend(atom_rows)
    lines.append("#")

    return "\n".join(lines) + "\n"


def atom_site_row(
    atom_id: int,
    *,
    comp_id: str = "ALA",
    atom_name: str = "CA",
    element: str = "C",
    alt_id: str = ".",
    asym_id: str = "A",
    entity_id: int = 1,
    seq_id: int = 1,
    occupancy: str = "1.00",
    b_factor: str = "10.00",
    group: str = "ATOM",
    model_num: int = 1,
) -> str:
    coordinate = float(atom_id)
    return (
        f"{group} {atom_id} {element} {atom_name} {alt_id} {comp_id} "
        f"{asym_id} {entity_id} {seq_id} ? "
        f"{coordinate:.1f} 0.0 0.0 {occupancy} {b_factor} "
        f"{seq_id} {comp_id} {asym_id} {atom_name} {model_num}"
    )


def backbone_atom_rows(
    start_atom_id: int,
    *,
    residue_count: int,
    identical_backbone_b_factors: bool = False,
) -> tuple[str, ...]:
    rows: list[str] = []
    atom_id = start_atom_id

    for residue_index in range(1, residue_count + 1):
        for atom_offset, atom_name in enumerate(("N", "CA", "C", "O")):
            b_factor = (
                "10.00"
                if identical_backbone_b_factors
                else f"{10.0 + residue_index + atom_offset / 10.0:.2f}"
            )
            rows.append(
                atom_site_row(
                    atom_id,
                    seq_id=residue_index,
                    atom_name=atom_name,
                    element="N" if atom_name == "N" else "C",
                    b_factor=b_factor,
                )
            )
            atom_id += 1

    return tuple(rows)


def _quote_cif(value: str) -> str:
    return f"'{value}'"


class PdbRedoStructureChecksTests(unittest.TestCase):
    def test_reads_protein_only_xray_entry(self) -> None:
        atoms = backbone_atom_rows(1, residue_count=6)

        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(atom_rows=atoms),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertTrue(checks.is_xray)
        self.assertTrue(checks.has_protein)
        self.assertFalse(checks.has_nucleic_acid)
        self.assertEqual(checks.experimental_methods, ("X-RAY DIFFRACTION",))
        self.assertEqual(checks.atom_count, 24)
        self.assertEqual(checks.non_hydrogen_atom_count, 24)
        self.assertEqual(checks.protein_atom_count, 24)
        self.assertEqual(checks.model_count, 1)
        self.assertTrue(checks.has_nonflat_protein_b_factors)
        self.assertEqual(checks.warnings, ())

    def test_peptide_nucleic_acid_is_nucleic_acid_not_protein(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    entity_polymer_types=("peptide nucleic acid",),
                    atom_rows=(
                        atom_site_row(
                            1,
                            comp_id="5MC",
                            atom_name="P",
                            element="P",
                        ),
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertFalse(checks.has_protein)
        self.assertTrue(checks.has_nucleic_acid)
        self.assertEqual(checks.warnings, ())

    def test_missing_entity_polymer_types_falls_back_to_component_classifier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    entity_polymer_types=None,
                    atom_rows=(
                        atom_site_row(1, comp_id="ASH"),
                        atom_site_row(
                            2,
                            comp_id="5MC",
                            atom_name="P",
                            element="P",
                            entity_id=2,
                            seq_id=2,
                        ),
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertTrue(checks.has_protein)
        self.assertTrue(checks.has_nucleic_acid)
        self.assertEqual(
            checks.warnings,
            (
                "Could not read _entity_poly.type values; protein/nucleic-acid "
                "classification falls back to residue-name heuristics.",
            ),
        )

    def test_altloc_carboxyl_oxygen_occupancies_are_summed_by_atom_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    atom_rows=(
                        atom_site_row(
                            1,
                            comp_id="ASP",
                            atom_name="OD1",
                            element="O",
                            alt_id="A",
                            occupancy="0.50",
                        ),
                        atom_site_row(
                            2,
                            comp_id="ASP",
                            atom_name="OD1",
                            element="O",
                            alt_id="B",
                            occupancy="0.50",
                        ),
                        atom_site_row(
                            3,
                            comp_id="ASP",
                            atom_name="OD2",
                            element="O",
                        ),
                        atom_site_row(
                            4,
                            comp_id="GLU",
                            atom_name="OE1",
                            element="O",
                            seq_id=2,
                            occupancy="0.50",
                        ),
                        atom_site_row(
                            5,
                            comp_id="GLU",
                            atom_name="OE2",
                            element="O",
                            seq_id=2,
                        ),
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertEqual(checks.asp_glu_residue_count, 2)
        self.assertEqual(checks.asp_glu_carboxyl_oxygen_count, 5)
        self.assertEqual(
            checks.asp_glu_residue_keys_with_occupancy_below_one,
            ("model=1;chain=A;residue=GLU;seqid=2",),
        )

    def test_d_isomer_asp_glu_residues_count_for_bnet_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    atom_rows=(
                        atom_site_row(
                            1,
                            comp_id="DAS",
                            atom_name="OD1",
                            element="O",
                        ),
                        atom_site_row(
                            2,
                            comp_id="DAS",
                            atom_name="OD2",
                            element="O",
                            occupancy="0.50",
                        ),
                        atom_site_row(
                            3,
                            comp_id="DGL",
                            atom_name="OE1",
                            element="O",
                            seq_id=2,
                        ),
                        atom_site_row(
                            4,
                            comp_id="DGL",
                            atom_name="OE2",
                            element="O",
                            seq_id=2,
                        ),
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertEqual(checks.asp_glu_residue_count, 2)
        self.assertEqual(checks.asp_glu_carboxyl_oxygen_count, 4)
        self.assertEqual(
            checks.asp_glu_residue_keys_with_occupancy_below_one,
            ("model=1;chain=A;residue=DAS;seqid=1",),
        )

    def test_backbone_b_factor_fallback_matches_rabdam2_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    atom_rows=backbone_atom_rows(
                        1,
                        residue_count=20,
                        identical_backbone_b_factors=True,
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertFalse(checks.has_nonflat_protein_b_factors)

    def test_non_xray_method_is_not_xray(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    methods=("ELECTRON MICROSCOPY",),
                    atom_rows=(atom_site_row(1),),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertFalse(checks.is_xray)
        self.assertEqual(checks.experimental_methods, ("ELECTRON MICROSCOPY",))

    def test_multi_model_entries_use_first_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                mmcif_text(
                    atom_rows=(
                        atom_site_row(
                            1,
                            comp_id="ASP",
                            atom_name="OD1",
                            element="O",
                        ),
                        atom_site_row(
                            2,
                            comp_id="ASP",
                            atom_name="OD2",
                            element="O",
                        ),
                        atom_site_row(
                            3,
                            comp_id="GLU",
                            atom_name="OE1",
                            element="O",
                            seq_id=2,
                            occupancy="0.50",
                            model_num=2,
                        ),
                        atom_site_row(
                            4,
                            comp_id="GLU",
                            atom_name="OE2",
                            element="O",
                            seq_id=2,
                            model_num=2,
                        ),
                    ),
                ),
            )

            checks = read_pdb_redo_structure_checks(candidate)

        self.assertEqual(checks.atom_count, 2)
        self.assertEqual(checks.non_hydrogen_atom_count, 2)
        self.assertEqual(checks.protein_atom_count, 2)
        self.assertEqual(checks.model_count, 2)
        self.assertEqual(checks.asp_glu_residue_count, 1)
        self.assertEqual(checks.asp_glu_carboxyl_oxygen_count, 2)
        self.assertEqual(
            checks.asp_glu_residue_keys_with_occupancy_below_one,
            (),
        )
        self.assertEqual(
            checks.warnings,
            (
                "Structure contains multiple models; structural checks use "
                "only the first model.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
