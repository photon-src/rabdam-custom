from pathlib import Path
import tempfile
import unittest

from database.discover import (
    PdbRedoDiscoveryError,
    PdbRedoDiscoverySkipReason,
    discover_pdb_redo_candidates,
    iter_pdb_redo_candidates,
)


def write_entry(entry_dir: Path, pdb_id: str, *, data_json: bool = True) -> Path:
    entry_dir.mkdir(parents=True, exist_ok=True)
    final_cif_path = entry_dir / f"{pdb_id}_final.cif"
    final_cif_path.write_text("data_test\n", encoding="utf-8")
    if data_json:
        (entry_dir / "data.json").write_text("{}", encoding="utf-8")
    return final_cif_path


class PdbRedoDiscoveryTests(unittest.TestCase):
    def test_discovers_flat_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_entry(root / "1abc", "1abc")
            write_entry(root / "2def", "2def")

            result = discover_pdb_redo_candidates(root, recursive=False)

        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.skipped_count, 0)
        self.assertEqual(result.pdb_ids, ("1abc", "2def"))
        self.assertEqual(
            [candidate.data_json_path.name for candidate in result.candidates],
            ["data.json", "data.json"],
        )

    def test_discovers_nested_candidates_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_entry(root / "nested" / "aa" / "1abc", "1abc")

            result = discover_pdb_redo_candidates(root)

        self.assertEqual(result.pdb_ids, ("1abc",))

    def test_missing_data_json_can_be_required_or_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_cif_path = write_entry(root / "1abc", "1abc", data_json=False)

            required_result = discover_pdb_redo_candidates(root)
            optional_result = discover_pdb_redo_candidates(
                root,
                require_data_json=False,
            )

        self.assertEqual(required_result.candidates, ())
        self.assertEqual(required_result.skipped_count, 1)
        self.assertEqual(
            required_result.skipped[0].reason,
            PdbRedoDiscoverySkipReason.MISSING_DATA_JSON,
        )
        self.assertEqual(required_result.skipped[0].path, final_cif_path)
        self.assertEqual(optional_result.pdb_ids, ("1abc",))
        self.assertIsNone(optional_result.candidates[0].data_json_path)

    def test_skips_invalid_pdb_id_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_cif_path = write_entry(root / "bad", "abcd")

            result = discover_pdb_redo_candidates(root)

        self.assertEqual(result.candidates, ())
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(
            result.skipped[0].reason,
            PdbRedoDiscoverySkipReason.INVALID_PDB_ID,
        )
        self.assertEqual(result.skipped[0].path, final_cif_path)

    def test_duplicate_pdb_ids_are_resolved_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = write_entry(root / "a" / "1abc", "1abc")
            skipped_path = write_entry(root / "b" / "1abc", "1abc")

            result = discover_pdb_redo_candidates(root)

        self.assertEqual(result.pdb_ids, ("1abc",))
        self.assertEqual(result.candidates[0].final_cif_path, accepted_path)
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(
            result.skipped[0].reason,
            PdbRedoDiscoverySkipReason.DUPLICATE_PDB_ID,
        )
        self.assertEqual(result.skipped[0].path, skipped_path)

    def test_iter_candidates_yields_only_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_entry(root / "1abc", "1abc")
            write_entry(root / "2def", "2def", data_json=False)

            candidates = tuple(iter_pdb_redo_candidates(root))

        self.assertEqual(tuple(candidate.pdb_id for candidate in candidates), ("1abc",))

    def test_missing_root_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "missing"

            with self.assertRaises(PdbRedoDiscoveryError):
                discover_pdb_redo_candidates(missing_root)
