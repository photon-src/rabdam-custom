import json
from pathlib import Path
import tempfile
import unittest

from database.discover import PdbRedoCandidate
from database.metadata import (
    PdbRedoMetadataError,
    read_pdb_redo_metadata,
)


def write_candidate_files(
    root: Path,
    *,
    data: object | None = None,
    final_cif_text: str | None = None,
    data_json: bool = True,
) -> PdbRedoCandidate:
    root.mkdir(parents=True, exist_ok=True)
    final_cif_path = root / "1abc_final.cif"
    final_cif_path.write_text(final_cif_text or "data_1abc\n#\n", encoding="utf-8")

    data_json_path = root / "data.json"
    if data_json:
        data_json_path.write_text(
            json.dumps({} if data is None else data),
            encoding="utf-8",
        )

    return PdbRedoCandidate(
        pdb_id="1abc",
        entry_dir=root,
        final_cif_path=final_cif_path,
        data_json_path=data_json_path if data_json else None,
    )


def mmcif_metadata_text(
    *,
    resolution: str = "1.50",
    r_work: str = "0.20",
    r_free: str = "0.25",
    temperature: str = "100",
    wilson_b: str = "12.5",
    b_restraint_weight: str = "0.8",
) -> str:
    return (
        "data_1abc\n"
        f"_refine.ls_d_res_high {resolution}\n"
        f"_refine.ls_r_factor_r_work {r_work}\n"
        f"_refine.ls_r_factor_r_free {r_free}\n"
        f"_diffrn.ambient_temp {temperature}\n"
        f"_reflns.B_iso_Wilson_estimate {wilson_b}\n"
        f"_refine.pdbx_adp_restraints_weight {b_restraint_weight}\n"
        "#\n"
    )


class PdbRedoMetadataTests(unittest.TestCase):
    def test_reads_data_json_metadata_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data={
                    "refine": {
                        "ls_d_res_high": "1.4",
                        "ls_R_factor_R_work": "0.19",
                        "ls_R_factor_R_free": "0.24",
                    },
                    "diffrn": {"ambient_temp": "100"},
                    "reflns": {"B_iso_Wilson_estimate": "13.5"},
                    "BFactorRestraintWeight": "0.9",
                },
            )

            metadata = read_pdb_redo_metadata(candidate)

        self.assertEqual(metadata.pdb_id, "1abc")
        self.assertEqual(metadata.resolution_angstrom, 1.4)
        self.assertEqual(metadata.r_work, 0.19)
        self.assertEqual(metadata.r_free, 0.24)
        self.assertEqual(metadata.temperature_k, 100.0)
        self.assertEqual(metadata.wilson_b, 13.5)
        self.assertEqual(metadata.b_factor_restraint_weight, 0.9)
        self.assertEqual(
            metadata.resolution_source,
            "data_json:refine.ls_d_res_high",
        )
        self.assertEqual(
            metadata.r_free_source,
            "data_json:refine.ls_R_factor_R_free",
        )
        self.assertEqual(
            metadata.b_factor_restraint_weight_source,
            "data_json:BFactorRestraintWeight",
        )
        self.assertEqual(metadata.warnings, ())

    def test_uses_mmcif_fallback_for_missing_json_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data={"resolution": "?"},
                final_cif_text=mmcif_metadata_text(),
            )

            metadata = read_pdb_redo_metadata(candidate)

        self.assertEqual(metadata.resolution_angstrom, 1.5)
        self.assertEqual(metadata.r_work, 0.2)
        self.assertEqual(metadata.r_free, 0.25)
        self.assertEqual(metadata.temperature_k, 100.0)
        self.assertEqual(metadata.wilson_b, 12.5)
        self.assertEqual(metadata.b_factor_restraint_weight, 0.8)
        self.assertEqual(metadata.resolution_source, "mmcif:_refine.ls_d_res_high")
        self.assertEqual(metadata.r_work_source, "mmcif:_refine.ls_r_factor_r_work")

    def test_data_json_values_take_priority_over_mmcif_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data={"resolution": "1.2"},
                final_cif_text=mmcif_metadata_text(resolution="2.2"),
            )

            metadata = read_pdb_redo_metadata(candidate)

        self.assertEqual(metadata.resolution_angstrom, 1.2)
        self.assertEqual(metadata.resolution_source, "data_json:resolution")

    def test_missing_data_json_warns_and_uses_mmcif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data_json=False,
                final_cif_text=mmcif_metadata_text(),
            )

            metadata = read_pdb_redo_metadata(candidate)

        self.assertEqual(metadata.resolution_angstrom, 1.5)
        self.assertIsNone(metadata.data_json_path)
        self.assertEqual(
            metadata.warnings,
            ("data.json is missing; metadata will rely on mmCIF fallback.",),
        )

    def test_mmcif_fallback_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data_json=False,
                final_cif_text=mmcif_metadata_text(),
            )

            metadata = read_pdb_redo_metadata(candidate, use_mmcif_fallback=False)

        self.assertIsNone(metadata.resolution_angstrom)
        self.assertIsNone(metadata.resolution_source)

    def test_invalid_data_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = write_candidate_files(root)
            data_json_path = candidate.data_json_path
            self.assertIsNotNone(data_json_path)
            assert data_json_path is not None
            data_json_path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(PdbRedoMetadataError):
                read_pdb_redo_metadata(candidate)

    def test_data_json_must_be_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(Path(temp_dir), data=[])

            with self.assertRaises(PdbRedoMetadataError):
                read_pdb_redo_metadata(candidate)

    def test_missing_and_non_finite_values_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data={
                    "resolution": "?",
                    "rfree": "nan",
                    "rwork": "inf",
                    "temperature": False,
                },
            )

            metadata = read_pdb_redo_metadata(
                candidate,
                use_mmcif_fallback=False,
            )

        self.assertIsNone(metadata.resolution_angstrom)
        self.assertIsNone(metadata.r_free)
        self.assertIsNone(metadata.r_work)
        self.assertIsNone(metadata.temperature_k)
        self.assertIsNone(metadata.resolution_source)

    def test_stereochemistry_target_values_are_not_b_factor_restraint_weight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = write_candidate_files(
                Path(temp_dir),
                data_json=False,
                final_cif_text=(
                    "data_1abc\n"
                    "_refine.pdbx_stereochemistry_target_values 3.2\n"
                    "#\n"
                ),
            )

            metadata = read_pdb_redo_metadata(candidate)

        self.assertIsNone(metadata.b_factor_restraint_weight)
        self.assertIsNone(metadata.b_factor_restraint_weight_source)
