import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bnet.reference import (
    BnetReferenceDatabase,
    BnetReferenceEntry,
    BnetReferenceError,
    BnetReferenceMetadata,
    load_bnet_reference_database,
    write_bnet_reference_database,
)


class BnetReferenceTests(unittest.TestCase):
    def test_loads_normalized_reference_csv_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "reference.csv"
            manifest_path = Path(temp_dir) / "reference.manifest.json"
            csv_path.write_text(
                "pdb_id,resolution_angstrom,bnet,ignored\n"
                "1ABC,1.5,1.2,x\n"
                "2DEF,2.1,2.4,y\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "database_id": "test_reference",
                        "schema_version": "1.0",
                        "metric_kind": "protein_cryo_asp_glu_bnet",
                        "source": "unit test",
                    }
                ),
                encoding="utf-8",
            )

            database = load_bnet_reference_database(csv_path)

        self.assertEqual(database.entry_count, 2)
        self.assertEqual(database.pdb_ids, ("1ABC", "2DEF"))
        self.assertEqual(database.metadata.database_id, "test_reference")
        self.assertEqual(database.metadata.source, "unit test")
        self.assertEqual([entry.bnet for entry in database.entries], [1.2, 2.4])
        np.testing.assert_allclose(database.resolution_values, [1.5, 2.1])
        np.testing.assert_allclose(database.bnet_values, [1.2, 2.4])
        self.assertFalse(database.resolution_values.flags.writeable)
        self.assertFalse(database.bnet_values.flags.writeable)

    def test_loads_standard_reference_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "reference.csv"
            csv_path.write_text(
                ",PDB code,Resolution (A),Bnet\n"
                "0,1ABC,1.5,1.2\n",
                encoding="utf-8",
            )

            database = load_bnet_reference_database(
                csv_path,
                database_id="standard_reference",
            )

        self.assertEqual(database.metadata.database_id, "standard_reference")
        self.assertEqual(database.entries[0], BnetReferenceEntry("1ABC", 1.5, 1.2))

    def test_load_reference_csv_rejects_ambiguous_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "ambiguous.csv"
            csv_path.write_text(
                "pdb_id,PDB code,Resolution (A),Bnet\n"
                "1ABC,2DEF,1.5,1.2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BnetReferenceError, "ambiguous 'pdb_id'"):
                load_bnet_reference_database(csv_path)

    def test_write_reference_database_round_trips_csv_and_manifest(self) -> None:
        database = BnetReferenceDatabase(
            entries=(
                BnetReferenceEntry("1ABC", 1.5, 1.2),
                BnetReferenceEntry("2DEF", 2.1, 2.4),
            ),
            metadata=BnetReferenceMetadata(
                database_id="round_trip",
                source="unit test",
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "nested" / "reference.csv"

            write_bnet_reference_database(database, csv_path)
            restored = load_bnet_reference_database(csv_path)

        self.assertEqual(restored.entries, database.entries)
        self.assertEqual(restored.metadata.database_id, "round_trip")
        self.assertEqual(restored.metadata.source, "unit test")
        self.assertEqual(restored.metadata.source_path, str(csv_path))
        self.assertEqual(
            restored.metadata.manifest_path,
            str(csv_path.with_suffix(".manifest.json")),
        )

    def test_reference_database_rejects_invalid_entries(self) -> None:
        invalid_entries = (
            lambda: BnetReferenceEntry("", 1.5, 1.2),
            lambda: BnetReferenceEntry("1ABC", True, 1.2),
            lambda: BnetReferenceEntry("1ABC", object(), 1.2),
            lambda: BnetReferenceEntry("1ABC", 0.0, 1.2),
            lambda: BnetReferenceEntry("1ABC", 1.5, -1.0),
            lambda: BnetReferenceDatabase(entries=()),
            lambda: BnetReferenceDatabase(
                entries=(
                    BnetReferenceEntry("1ABC", 1.5, 1.2),
                    BnetReferenceEntry("1abc", 1.6, 1.3),
                )
            ),
        )

        for make_invalid in invalid_entries:
            with self.subTest(make_invalid=make_invalid):
                with self.assertRaises(BnetReferenceError):
                    make_invalid()

    def test_load_reference_csv_reports_row_and_column_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_column_path = Path(temp_dir) / "missing.csv"
            invalid_row_path = Path(temp_dir) / "invalid.csv"
            missing_column_path.write_text(
                "pdb_id,bnet\n"
                "1ABC,1.2\n",
                encoding="utf-8",
            )
            invalid_row_path.write_text(
                "pdb_id,resolution_angstrom,bnet\n"
                "1ABC,not-resolution,1.2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BnetReferenceError, "resolution_angstrom"):
                load_bnet_reference_database(missing_column_path)

            with self.assertRaisesRegex(BnetReferenceError, "row 2"):
                load_bnet_reference_database(invalid_row_path)

    def test_manifest_must_be_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "reference.csv"
            manifest_path = Path(temp_dir) / "reference.manifest.json"
            csv_path.write_text(
                "pdb_id,resolution_angstrom,bnet\n"
                "1ABC,1.5,1.2\n",
                encoding="utf-8",
            )
            manifest_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(BnetReferenceError, "JSON object"):
                load_bnet_reference_database(csv_path)
