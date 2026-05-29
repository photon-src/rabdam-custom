from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from database.build import (
    BnetDatabaseBuildOptions,
    _WorkerOptions,
    _process_candidate_worker,
    _process_candidates_parallel,
    build_bnet_reference_database,
    parse_args,
)
from database.discover import PdbRedoCandidate, PdbRedoDiscoveryResult
from database.output import ACCEPTED_FIELDNAMES, REJECTED_FIELDNAMES
from database.process import (
    AcceptedBnetReferenceRow,
    PdbRedoProcessResult,
    PdbRedoProcessStage,
    PdbRedoRejectReason,
    RejectedBnetReferenceRow,
)


def make_candidate(pdb_id: str) -> PdbRedoCandidate:
    entry_dir = Path("/tmp/pdb-redo") / pdb_id
    return PdbRedoCandidate(
        pdb_id=pdb_id,
        entry_dir=entry_dir,
        final_cif_path=entry_dir / f"{pdb_id}_final.cif",
        data_json_path=entry_dir / "data.json",
    )


def make_accepted_result(pdb_id: str) -> PdbRedoProcessResult:
    return PdbRedoProcessResult(
        pdb_id=pdb_id,
        accepted=AcceptedBnetReferenceRow(
            pdb_id=pdb_id,
            resolution_angstrom=1.5,
            bnet=1.2,
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
            has_asp_glu_residue_with_total_occupancy_below_one=False,
            experimental_methods=("X-RAY DIFFRACTION",),
            final_cif_path=Path(f"/tmp/{pdb_id}_final.cif"),
            data_json_path=Path("/tmp/data.json"),
        ),
    )


def make_rejected_result(pdb_id: str) -> PdbRedoProcessResult:
    return PdbRedoProcessResult(
        pdb_id=pdb_id,
        rejected=RejectedBnetReferenceRow(
            pdb_id=pdb_id,
            stage=PdbRedoProcessStage.DOMAIN_FILTER,
            reason=PdbRedoRejectReason.NOT_XRAY.value,
            message="Not X-ray.",
            final_cif_path=Path(f"/tmp/{pdb_id}_final.cif"),
            data_json_path=Path("/tmp/data.json"),
        ),
    )


def make_worker_options(*, include_traceback: bool = False) -> _WorkerOptions:
    return _WorkerOptions(
        require_xray=True,
        require_single_model=True,
        require_protein=True,
        reject_nucleic_acid=True,
        include_traceback=include_traceback,
    )


class FakeFuture:
    def __init__(self, result: PdbRedoProcessResult):
        self._result = result

    def result(self) -> PdbRedoProcessResult:
        return self._result


class FakeProcessPoolExecutor:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers

    def __enter__(self) -> "FakeProcessPoolExecutor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(
        self,
        _callback: object,
        candidate: PdbRedoCandidate,
        _worker_options: _WorkerOptions,
    ) -> FakeFuture:
        return FakeFuture(make_rejected_result(candidate.pdb_id))


def fake_wait(
    futures: object,
    *,
    return_when: object,
) -> tuple[set[object], set[object]]:
    del return_when
    return set(futures), set()


class BnetDatabaseBuildTests(unittest.TestCase):
    def test_build_resumes_only_discovered_processed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted_path = root / "accepted.csv"
            rejected_path = root / "rejected.csv"
            accepted_path.write_text(
                ",".join(ACCEPTED_FIELDNAMES)
                + "\n0,1ABC,1.5,1.2\n",
                encoding="utf-8",
            )
            rejected_path.write_text(
                ",".join(REJECTED_FIELDNAMES)
                + "\n9ZZZ,domain_filter,not_xray,msg,,,,,/tmp/a.cif,,,,,\n",
                encoding="utf-8",
            )
            discovery = PdbRedoDiscoveryResult(
                candidates=(make_candidate("1abc"), make_candidate("2def")),
                skipped=(),
            )

            with (
                patch(
                    "database.build.discover_pdb_redo_candidates",
                    return_value=discovery,
                ),
                patch(
                    "database.build._process_candidates_parallel",
                    return_value=iter((make_accepted_result("2def"),)),
                ) as process_mock,
                patch("database.build._print_build_start"),
                patch("database.build._print_progress"),
                patch("database.build._print_build_summary"),
            ):
                summary = build_bnet_reference_database(
                    BnetDatabaseBuildOptions(
                        pdb_redo_root=root,
                        accepted_csv_path=accepted_path,
                        rejected_csv_path=rejected_path,
                        jobs=1,
                    )
                )

        attempted_candidates = process_mock.call_args.args[0]
        self.assertEqual(
            [candidate.pdb_id for candidate in attempted_candidates],
            ["2def"],
        )
        self.assertEqual(summary.discovered_candidate_count, 2)
        self.assertEqual(summary.already_processed_count, 1)
        self.assertEqual(summary.attempted_count, 1)
        self.assertEqual(summary.accepted_count, 1)

    def test_parallel_processing_yields_results_in_candidate_order(self) -> None:
        candidates = (
            make_candidate("1abc"),
            make_candidate("2def"),
            make_candidate("3ghi"),
        )

        with (
            patch("database.build.ProcessPoolExecutor", FakeProcessPoolExecutor),
            patch("database.build.wait", fake_wait),
        ):
            results = tuple(
                _process_candidates_parallel(
                    candidates,
                    worker_options=make_worker_options(),
                    jobs=2,
                    max_tasks_in_flight=3,
                )
            )

        self.assertEqual(
            [result.pdb_id for result in results],
            ["1abc", "2def", "3ghi"],
        )

    def test_worker_unexpected_exception_becomes_rejected_result(self) -> None:
        candidate = make_candidate("1abc")

        with patch(
            "database.build.process_pdb_redo_candidate",
            side_effect=RuntimeError("boom"),
        ):
            result = _process_candidate_worker(
                candidate,
                make_worker_options(include_traceback=True),
            )

        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.UNEXPECTED_ERROR,
        )
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.UNEXPECTED_WORKER_ERROR.value,
        )
        self.assertEqual(result.rejected.exception_type, "RuntimeError")
        self.assertIn("RuntimeError: boom", result.rejected.message)
        self.assertIn("RuntimeError: boom", result.rejected.traceback_text or "")

    def test_parse_args_maps_cli_flags_to_options(self) -> None:
        options = parse_args(
            [
                "/tmp/pdb-redo",
                "--accepted-csv",
                "/tmp/accepted.csv",
                "--accepted-details-csv",
                "/tmp/details.csv",
                "--rejected-csv",
                "/tmp/rejected.csv",
                "--jobs",
                "3",
                "--max-tasks-in-flight",
                "6",
                "--max-candidates",
                "10",
                "--progress-every",
                "0",
                "--overwrite",
                "--no-resume",
                "--no-recursive-discovery",
                "--allow-missing-data-json",
                "--allow-non-xray",
                "--allow-multiple-models",
                "--allow-no-protein",
                "--allow-nucleic-acid",
                "--include-traceback",
            ]
        )

        self.assertEqual(options.pdb_redo_root, Path("/tmp/pdb-redo"))
        self.assertEqual(options.accepted_csv_path, Path("/tmp/accepted.csv"))
        self.assertEqual(
            options.accepted_details_csv_path,
            Path("/tmp/details.csv"),
        )
        self.assertEqual(options.rejected_csv_path, Path("/tmp/rejected.csv"))
        self.assertEqual(options.jobs, 3)
        self.assertEqual(options.max_tasks_in_flight, 6)
        self.assertEqual(options.max_candidates, 10)
        self.assertEqual(options.progress_every, 0)
        self.assertTrue(options.overwrite)
        self.assertFalse(options.resume)
        self.assertFalse(options.recursive_discovery)
        self.assertFalse(options.require_data_json)
        self.assertFalse(options.require_xray)
        self.assertFalse(options.require_single_model)
        self.assertFalse(options.require_protein)
        self.assertFalse(options.reject_nucleic_acid)
        self.assertTrue(options.include_traceback)


if __name__ == "__main__":
    unittest.main()
