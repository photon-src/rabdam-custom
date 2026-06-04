import csv
import json
from pathlib import Path
import tempfile
import unittest

from bnet.reference import load_bnet_reference_database
from database.output import (
    ACCEPTED_DETAILS_FIELDNAMES,
    ACCEPTED_FIELDNAMES,
    ALL_SCORES_FIELDNAMES,
    REJECTED_FIELDNAMES,
    TEMPERATURE_CACHE_FIELDNAMES,
    BnetDatabaseCsvWriter,
    BnetDatabaseOutputError,
    TemperatureCacheCsvWriter,
    load_temperature_cache,
    write_sorted_reference_csv_from_accepted_csv,
    write_simple_bnet_database_csv,
)
from database.process import (
    AcceptedBnetReferenceRow,
    PdbRedoProcessResult,
    PdbRedoProcessStage,
    RejectedBnetReferenceRow,
)


def make_accepted_row(
    pdb_id: str = "1abc",
    *,
    resolution_angstrom: float = 1.5,
    bnet: float = 1.2,
) -> AcceptedBnetReferenceRow:
    return AcceptedBnetReferenceRow(
        pdb_id=pdb_id,
        resolution_angstrom=resolution_angstrom,
        bnet=bnet,
        r_work=0.2,
        r_free=0.25,
        temperature_k=100.0,
        wilson_b=12.5,
        b_factor_restraint_weight=0.8,
        bnet_site_count=24,
        asp_glu_carboxyl_oxygen_count=24,
        asp_glu_residue_count=12,
        median_bdamage=1.1,
        left_area=0.4,
        right_area=0.6,
        atom_count=100,
        non_hydrogen_atom_count=90,
        protein_atom_count=80,
        selected_atom_count=70,
        bdamage_window_size=11,
        has_protein=True,
        has_nucleic_acid=False,
        is_xray=True,
        has_nonflat_protein_b_factors=True,
        uses_per_atom_b_factors=True,
        b_factor_model_source="data_json:properties.BREFTYPE",
        b_factor_refinement_type="ISOT",
        b_factor_refinement_type_source="data_json:properties.BREFTYPE",
        has_asp_glu_residue_with_total_occupancy_below_one=False,
        experimental_methods=("X-RAY DIFFRACTION",),
        metadata_warnings=("metadata warning",),
        structure_check_warnings=("structure warning",),
        resolution_source="data_json:resolution",
        r_work_source="data_json:rwork",
        r_free_source="data_json:rfree",
        temperature_source="data_json:temperature",
        temperature_values_k=(100.0,),
        temperature_sources=("data_json:temperature",),
        growth_temperature_k=293.0,
        growth_temperature_values_k=(293.0,),
        growth_temperature_source="data_json:exptl_crystal_grow.temp",
        growth_temperature_sources=("data_json:exptl_crystal_grow.temp",),
        wilson_b_source="data_json:wilson_b",
        b_factor_restraint_weight_source="data_json:b_factor_restraint_weight",
        final_cif_path=Path("/tmp/1abc_final.cif"),
        data_json_path=Path("/tmp/data.json"),
    )


def make_rejected_row() -> RejectedBnetReferenceRow:
    return RejectedBnetReferenceRow(
        pdb_id="2def",
        stage=PdbRedoProcessStage.RABDAM,
        reason="rabdam_error",
        message="Calculation failed",
        final_cif_path=Path("/tmp/2def_final.cif"),
        data_json_path=Path("/tmp/data.json"),
        exception_type="ValueError",
        traceback_text="Traceback text",
        resolution_angstrom=2.0,
        r_free=0.3,
        temperature_k=100.0,
        asp_glu_carboxyl_oxygen_count=22,
        bnet=None,
        metadata_warnings=("metadata warning",),
        structure_check_warnings=("structure warning",),
        uses_per_atom_b_factors=False,
        b_factor_model_source="data_json:properties.BREFTYPE",
        b_factor_refinement_type="OVER",
        b_factor_refinement_type_source="data_json:properties.BREFTYPE",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class BnetDatabaseOutputTests(unittest.TestCase):
    def test_writer_appends_minimal_details_and_rejected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            details_path = root / "accepted_details.csv"
            rejected_path = root / "rejected.csv"

            writer = BnetDatabaseCsvWriter(
                accepted_path,
                accepted_details_csv_path=details_path,
                rejected_csv_path=rejected_path,
            )
            accepted_row = make_accepted_row()
            rejected_row = make_rejected_row()

            writer.write_result(
                PdbRedoProcessResult(
                    pdb_id=accepted_row.pdb_id,
                    accepted=accepted_row,
                )
            )
            writer.write_result(
                PdbRedoProcessResult(
                    pdb_id=rejected_row.pdb_id,
                    rejected=rejected_row,
                )
            )

            accepted_rows = read_csv_rows(accepted_path)
            detail_rows = read_csv_rows(details_path)
            rejected_rows = read_csv_rows(rejected_path)

        self.assertEqual(writer.accepted_count, 1)
        self.assertEqual(writer.rejected_count, 1)
        self.assertEqual(accepted_rows[0][""], "0")
        self.assertEqual(accepted_rows[0]["PDB code"], "1ABC")
        self.assertEqual(detail_rows[0]["PDB code"], "1ABC")
        self.assertEqual(detail_rows[0]["has_nonflat_protein_b_factors"], "true")
        self.assertEqual(detail_rows[0]["uses_per_atom_b_factors"], "true")
        self.assertEqual(detail_rows[0]["b_factor_refinement_type"], "ISOT")
        self.assertEqual(
            json.loads(detail_rows[0]["temperature_values_K"]),
            [100.0],
        )
        self.assertEqual(detail_rows[0]["growth_temperature_K"], "293")
        self.assertEqual(
            json.loads(detail_rows[0]["growth_temperature_values_K"]),
            [293.0],
        )
        self.assertEqual(
            json.loads(detail_rows[0]["metadata_warnings"]),
            ["metadata warning"],
        )
        self.assertEqual(rejected_rows[0]["PDB code"], "2DEF")
        self.assertEqual(rejected_rows[0]["traceback_text"], "Traceback text")
        self.assertEqual(rejected_rows[0]["uses_per_atom_b_factors"], "false")
        self.assertEqual(rejected_rows[0]["b_factor_refinement_type"], "OVER")
        self.assertEqual(
            json.loads(rejected_rows[0]["structure_check_warnings"]),
            ["structure warning"],
        )

    def test_accepted_output_round_trips_through_reference_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted_path = Path(temp_dir) / "accepted.csv"
            writer = BnetDatabaseCsvWriter(accepted_path)

            writer.write_accepted(make_accepted_row("1abc", bnet=1.2))
            writer.write_accepted(make_accepted_row("2def", bnet=2.4))

            database = load_bnet_reference_database(
                accepted_path,
                database_id="test_reference",
            )

        self.assertEqual(database.pdb_ids, ("1ABC", "2DEF"))
        self.assertEqual([entry.bnet for entry in database.entries], [1.2, 2.4])

    def test_writer_validates_existing_headers_before_appending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted_path = Path(temp_dir) / "accepted.csv"
            accepted_path.write_text(
                "pdb_id,resolution_angstrom,bnet\n"
                "1ABC,1.5,1.2\n",
                encoding="utf-8",
            )

            with self.assertRaises(BnetDatabaseOutputError):
                BnetDatabaseCsvWriter(accepted_path)

    def test_writer_rejects_mismatched_resume_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            details_path = root / "accepted_details.csv"
            accepted_path.write_text(
                ",".join(ACCEPTED_FIELDNAMES)
                + "\n0,1ABC,1.5,1.2\n",
                encoding="utf-8",
            )
            details_path.write_text(
                ",".join(ACCEPTED_DETAILS_FIELDNAMES) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(BnetDatabaseOutputError):
                BnetDatabaseCsvWriter(
                    accepted_path,
                    accepted_details_csv_path=details_path,
                )

    def test_writer_overwrite_replaces_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            rejected_path = root / "rejected.csv"
            accepted_path.write_text("bad\n", encoding="utf-8")
            rejected_path.write_text("bad\n", encoding="utf-8")

            writer = BnetDatabaseCsvWriter(
                accepted_path,
                rejected_csv_path=rejected_path,
                overwrite=True,
            )

            accepted_header = accepted_path.read_text(encoding="utf-8").splitlines()[0]
            rejected_header = rejected_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(writer.accepted_count, 0)
        self.assertEqual(writer.rejected_count, 0)
        self.assertEqual(accepted_header, ",".join(ACCEPTED_FIELDNAMES))
        self.assertEqual(rejected_header, ",".join(REJECTED_FIELDNAMES))

    def test_rejected_output_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted_path = Path(temp_dir) / "accepted.csv"
            writer = BnetDatabaseCsvWriter(accepted_path)

            writer.write_rejected(make_rejected_row())

        self.assertEqual(writer.rejected_count, 0)
        self.assertIsNone(writer.paths.rejected_csv_path)

    def test_writer_records_all_scores_for_accepted_ineligible_and_failure_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            all_scores_path = root / "all_scores.csv"
            writer = BnetDatabaseCsvWriter(
                accepted_path,
                rejected_csv_path=root / "rejected.csv",
                all_scores_csv_path=all_scores_path,
            )

            accepted_row = make_accepted_row()
            ineligible_row = RejectedBnetReferenceRow(
                pdb_id="2def",
                stage=PdbRedoProcessStage.FINAL_ELIGIBILITY,
                reason="cannot_verify_temperature",
                message="Collection temperature could not be verified.",
                final_cif_path=Path("/tmp/2def_final.cif"),
                data_json_path=Path("/tmp/data.json"),
                resolution_angstrom=1.6,
                r_free=0.22,
                asp_glu_carboxyl_oxygen_count=30,
                bnet=2.5,
                temperature_cache_status="missing",
                temperature_cache_message="No temperature metadata found.",
                uses_per_atom_b_factors=True,
                b_factor_model_source="structure_backbone_b_factor_check",
            )
            failure_row = make_rejected_row()

            writer.write_result(
                PdbRedoProcessResult(
                    pdb_id=accepted_row.pdb_id,
                    accepted=accepted_row,
                )
            )
            writer.write_result(
                PdbRedoProcessResult(
                    pdb_id=ineligible_row.pdb_id,
                    rejected=ineligible_row,
                )
            )
            writer.write_result(
                PdbRedoProcessResult(
                    pdb_id=failure_row.pdb_id,
                    rejected=failure_row,
                )
            )

            rows = read_csv_rows(all_scores_path)

        self.assertEqual(writer.all_scores_count, 3)
        self.assertEqual(tuple(rows[0]), ALL_SCORES_FIELDNAMES)
        self.assertEqual(rows[0]["PDB code"], "1ABC")
        self.assertEqual(rows[0]["status"], "accepted")
        self.assertEqual(rows[0]["is_reference_eligible"], "true")
        self.assertEqual(rows[0]["growth_temperature_K"], "293")
        self.assertEqual(rows[0]["b_factor_refinement_type"], "ISOT")
        self.assertEqual(rows[1]["PDB code"], "2DEF")
        self.assertEqual(rows[1]["status"], "rejected")
        self.assertEqual(rows[1]["reason"], "cannot_verify_temperature")
        self.assertEqual(rows[1]["Bnet"], "2.5")
        self.assertEqual(rows[1]["uses_per_atom_b_factors"], "true")
        self.assertEqual(rows[1]["is_reference_eligible"], "false")
        self.assertEqual(rows[2]["stage"], PdbRedoProcessStage.RABDAM.value)
        self.assertEqual(rows[2]["exception_type"], "ValueError")
        self.assertEqual(rows[2]["Bnet"], "")

    def test_batch_writer_sorts_by_bnet_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted_path = Path(temp_dir) / "accepted.csv"

            write_simple_bnet_database_csv(
                (
                    make_accepted_row("1abc", bnet=1.2),
                    make_accepted_row("2def", bnet=2.4),
                ),
                accepted_path,
            )
            rows = read_csv_rows(accepted_path)
            database = load_bnet_reference_database(
                accepted_path,
                database_id="sorted_reference",
            )

        self.assertEqual([row[""] for row in rows], ["0", "1"])
        self.assertEqual(database.pdb_ids, ("2DEF", "1ABC"))
        self.assertEqual([entry.bnet for entry in database.entries], [2.4, 1.2])

    def test_final_reference_csv_is_sorted_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            final_path = root / "database.csv"
            writer = BnetDatabaseCsvWriter(accepted_path)
            writer.write_accepted(make_accepted_row("1abc", bnet=1.2))
            writer.write_accepted(make_accepted_row("2def", bnet=2.4))

            count = write_sorted_reference_csv_from_accepted_csv(
                accepted_path,
                final_path,
            )
            rows = read_csv_rows(final_path)
            database = load_bnet_reference_database(
                final_path,
                database_id="final_reference",
            )

        self.assertEqual(count, 2)
        self.assertEqual(tuple(rows[0]), ACCEPTED_FIELDNAMES)
        self.assertEqual([row["PDB code"] for row in rows], ["2DEF", "1ABC"])
        self.assertEqual(database.pdb_ids, ("2DEF", "1ABC"))

    def test_temperature_cache_writer_round_trips_compact_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "temperature_cache.csv"
            writer = TemperatureCacheCsvWriter(cache_path)
            row = RejectedBnetReferenceRow(
                pdb_id="1abc",
                stage=PdbRedoProcessStage.FINAL_ELIGIBILITY,
                reason="cannot_verify_temperature",
                message="Temperature recovered from RCSB.",
                final_cif_path=Path("/tmp/1abc_final.cif"),
                temperature_k=100.0,
                temperature_values_k=(100.0, 110.0),
                temperature_source="rcsb_mmcif:_diffrn.ambient_temp",
                temperature_cache_status="found",
            )

            writer.write_result(PdbRedoProcessResult(pdb_id="1abc", rejected=row))
            rows = read_csv_rows(cache_path)
            cache = load_temperature_cache(cache_path)

        self.assertEqual(tuple(rows[0]), TEMPERATURE_CACHE_FIELDNAMES)
        self.assertEqual(rows[0]["PDB code"], "1ABC")
        self.assertEqual(
            json.loads(rows[0]["collection_temperature_values_K"]),
            [100.0, 110.0],
        )
        self.assertIn("1abc", cache)
        self.assertEqual(cache["1abc"].temperature_values_k, (100.0, 110.0))


if __name__ == "__main__":
    unittest.main()
