"""Build a PDB-REDO Bnet reference database.

This module orchestrates the full local build:

1. Discover local PDB-REDO candidates.
2. Optionally skip candidates already present in existing output CSVs.
3. Process candidates in parallel.
4. Write accepted/rejected rows from the main process only.

Workers never write output files. This keeps the CSVs safe and resumable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import monotonic
import traceback

from .discover import (
    PdbRedoCandidate,
    PdbRedoDiscoveryResult,
    discover_pdb_redo_candidates,
)
from .output import BnetDatabaseCsvWriter
from .process import (
    PdbRedoProcessResult,
    PdbRedoProcessStage,
    PdbRedoRejectReason,
    RejectedBnetReferenceRow,
    process_pdb_redo_candidate,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_DATABASE_CSV_PATH = DEFAULT_OUTPUT_DIR / "database.csv"
DEFAULT_ACCEPTED_CSV_PATH = DEFAULT_DATABASE_CSV_PATH


@dataclass(frozen=True, slots=True)
class BnetDatabaseBuildOptions:
    """Options controlling a PDB-REDO Bnet database build."""

    pdb_redo_root: Path
    accepted_csv_path: Path = DEFAULT_ACCEPTED_CSV_PATH
    accepted_details_csv_path: Path | None = None
    rejected_csv_path: Path | None = None

    require_data_json: bool = True
    recursive_discovery: bool = True
    overwrite: bool = False
    resume: bool = True

    max_candidates: int | None = None
    progress_every: int = 100

    jobs: int = max(1, (os.cpu_count() or 2) - 1)
    max_tasks_in_flight: int | None = None

    require_xray: bool = True
    require_single_model: bool = True
    require_protein: bool = True
    reject_nucleic_acid: bool = True

    include_traceback: bool = False


@dataclass(frozen=True, slots=True)
class BnetDatabaseBuildSummary:
    """Summary of a completed Bnet database build run."""

    discovered_candidate_count: int
    discovery_skipped_count: int
    already_processed_count: int
    attempted_count: int
    accepted_count: int
    rejected_count: int
    elapsed_seconds: float

    @property
    def processed_count(self) -> int:
        """Return accepted plus rejected count from this build run."""

        return self.accepted_count + self.rejected_count


@dataclass(frozen=True, slots=True)
class _WorkerOptions:
    """Pickle-friendly subset of processing options used by worker processes."""

    require_xray: bool
    require_single_model: bool
    require_protein: bool
    reject_nucleic_acid: bool
    include_traceback: bool


def build_bnet_reference_database(
    options: BnetDatabaseBuildOptions,
) -> BnetDatabaseBuildSummary:
    """Build a Bnet reference database from a local PDB-REDO mirror."""

    start_time = monotonic()

    discovery = discover_pdb_redo_candidates(
        options.pdb_redo_root,
        require_data_json=options.require_data_json,
        recursive=options.recursive_discovery,
    )

    discovered_pdb_ids = frozenset(
        candidate.pdb_id.casefold() for candidate in discovery.candidates
    )
    existing_processed_pdb_ids = (
        _read_processed_pdb_ids(
            options.accepted_csv_path,
            options.rejected_csv_path,
        )
        if options.resume and not options.overwrite
        else frozenset()
    )
    already_processed_pdb_ids = existing_processed_pdb_ids & discovered_pdb_ids

    writer = BnetDatabaseCsvWriter(
        options.accepted_csv_path,
        accepted_details_csv_path=options.accepted_details_csv_path,
        rejected_csv_path=options.rejected_csv_path,
        overwrite=options.overwrite,
    )

    candidates = _filter_candidates(
        discovery.candidates,
        already_processed_pdb_ids=already_processed_pdb_ids,
        max_candidates=options.max_candidates,
    )

    _print_build_start(
        discovery=discovery,
        candidates_to_attempt=len(candidates),
        already_processed_count=len(already_processed_pdb_ids),
        options=options,
    )

    worker_options = _WorkerOptions(
        require_xray=options.require_xray,
        require_single_model=options.require_single_model,
        require_protein=options.require_protein,
        reject_nucleic_acid=options.reject_nucleic_acid,
        include_traceback=options.include_traceback,
    )

    attempted_count = 0
    accepted_this_run = 0
    rejected_this_run = 0

    for result in _process_candidates_parallel(
        candidates,
        worker_options=worker_options,
        jobs=options.jobs,
        max_tasks_in_flight=options.max_tasks_in_flight,
    ):
        attempted_count += 1
        writer.write_result(result)

        if result.is_accepted:
            accepted_this_run += 1
        else:
            rejected_this_run += 1
            if _is_unexpected_worker_error(result):
                _print_unexpected_worker_error(result)

        if _should_print_progress(attempted_count, options.progress_every):
            _print_progress(
                attempted_count=attempted_count,
                total_count=len(candidates),
                accepted_count=accepted_this_run,
                rejected_count=rejected_this_run,
                result=result,
            )

    elapsed_seconds = monotonic() - start_time

    summary = BnetDatabaseBuildSummary(
        discovered_candidate_count=discovery.candidate_count,
        discovery_skipped_count=discovery.skipped_count,
        already_processed_count=len(already_processed_pdb_ids),
        attempted_count=attempted_count,
        accepted_count=accepted_this_run,
        rejected_count=rejected_this_run,
        elapsed_seconds=elapsed_seconds,
    )

    _print_build_summary(summary, writer)

    return summary


def _process_candidates_parallel(
    candidates: tuple[PdbRedoCandidate, ...],
    *,
    worker_options: _WorkerOptions,
    jobs: int,
    max_tasks_in_flight: int | None,
) -> Iterator[PdbRedoProcessResult]:
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")

    if not candidates:
        return

    if jobs == 1:
        for candidate in candidates:
            yield _process_candidate_worker(candidate, worker_options)
        return

    tasks_in_flight_limit = max_tasks_in_flight
    if tasks_in_flight_limit is None:
        tasks_in_flight_limit = jobs * 2

    if tasks_in_flight_limit < jobs:
        raise ValueError(
            "max_tasks_in_flight must be greater than or equal to jobs."
        )

    candidate_iter = iter(candidates)

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_candidate = {}
        completed_results: dict[int, PdbRedoProcessResult] = {}
        next_submit_index = 0
        next_yield_index = 0

        for candidate in _take(candidate_iter, tasks_in_flight_limit):
            _submit_candidate_task(
                executor=executor,
                future_to_candidate=future_to_candidate,
                completed_results=completed_results,
                result_index=next_submit_index,
                candidate=candidate,
                worker_options=worker_options,
            )
            next_submit_index += 1

        while future_to_candidate or next_yield_index in completed_results:
            while next_yield_index in completed_results:
                yield completed_results.pop(next_yield_index)
                next_yield_index += 1

            if not future_to_candidate:
                break

            done, _pending = wait(
                future_to_candidate,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                result_index, candidate = future_to_candidate.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = _unexpected_worker_error_result(
                        candidate,
                        error,
                        include_traceback=worker_options.include_traceback,
                    )
                completed_results[result_index] = result

                try:
                    next_candidate = next(candidate_iter)
                except StopIteration:
                    pass
                else:
                    _submit_candidate_task(
                        executor=executor,
                        future_to_candidate=future_to_candidate,
                        completed_results=completed_results,
                        result_index=next_submit_index,
                        candidate=next_candidate,
                        worker_options=worker_options,
                    )
                    next_submit_index += 1


def _submit_candidate_task(
    *,
    executor: ProcessPoolExecutor,
    future_to_candidate: dict[object, tuple[int, PdbRedoCandidate]],
    completed_results: dict[int, PdbRedoProcessResult],
    result_index: int,
    candidate: PdbRedoCandidate,
    worker_options: _WorkerOptions,
) -> None:
    try:
        future = executor.submit(
            _process_candidate_worker,
            candidate,
            worker_options,
        )
    except Exception as error:
        completed_results[result_index] = _unexpected_worker_error_result(
            candidate,
            error,
            include_traceback=worker_options.include_traceback,
        )
        return

    future_to_candidate[future] = (result_index, candidate)


def _process_candidate_worker(
    candidate: PdbRedoCandidate,
    worker_options: _WorkerOptions,
) -> PdbRedoProcessResult:
    """Worker-process entry point.

    Keep this top-level so it is pickleable on macOS/Windows spawn mode.
    """

    try:
        return process_pdb_redo_candidate(
            candidate,
            require_xray=worker_options.require_xray,
            require_single_model=worker_options.require_single_model,
            require_protein=worker_options.require_protein,
            reject_nucleic_acid=worker_options.reject_nucleic_acid,
            include_traceback=worker_options.include_traceback,
        )
    except Exception as error:
        return _unexpected_worker_error_result(
            candidate,
            error,
            include_traceback=worker_options.include_traceback,
        )


def _unexpected_worker_error_result(
    candidate: PdbRedoCandidate,
    error: BaseException,
    *,
    include_traceback: bool,
) -> PdbRedoProcessResult:
    error_type = type(error).__name__
    return PdbRedoProcessResult(
        pdb_id=candidate.pdb_id,
        rejected=RejectedBnetReferenceRow(
            pdb_id=candidate.pdb_id,
            stage=PdbRedoProcessStage.UNEXPECTED_ERROR,
            reason=PdbRedoRejectReason.UNEXPECTED_WORKER_ERROR.value,
            message=(
                "Unexpected worker exception while processing candidate: "
                f"{error_type}: {error}"
            ),
            final_cif_path=candidate.final_cif_path,
            data_json_path=candidate.data_json_path,
            exception_type=error_type,
            traceback_text=traceback.format_exc() if include_traceback else None,
        ),
    )


def _take(
    iterator: Iterator[PdbRedoCandidate],
    count: int,
) -> Iterator[PdbRedoCandidate]:
    for _ in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            return


def _filter_candidates(
    candidates: tuple[PdbRedoCandidate, ...],
    *,
    already_processed_pdb_ids: frozenset[str],
    max_candidates: int | None,
) -> tuple[PdbRedoCandidate, ...]:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.pdb_id.casefold() not in already_processed_pdb_ids
    ]

    if max_candidates is not None:
        if max_candidates < 0:
            raise ValueError("max_candidates must be non-negative or None.")
        filtered = filtered[:max_candidates]

    return tuple(filtered)


def _is_unexpected_worker_error(result: PdbRedoProcessResult) -> bool:
    return (
        result.rejected is not None
        and result.rejected.reason
        == PdbRedoRejectReason.UNEXPECTED_WORKER_ERROR.value
    )


def _print_unexpected_worker_error(result: PdbRedoProcessResult) -> None:
    if result.rejected is None:
        return

    print(
        f"Unexpected worker error for {result.pdb_id.upper()}: "
        f"{result.rejected.message}",
        file=sys.stderr,
        flush=True,
    )
    if result.rejected.traceback_text:
        print(result.rejected.traceback_text, file=sys.stderr, flush=True)


def _read_processed_pdb_ids(
    accepted_csv_path: Path,
    rejected_csv_path: Path | None,
) -> frozenset[str]:
    pdb_ids: set[str] = set()

    pdb_ids.update(_read_pdb_ids_from_csv(accepted_csv_path))
    if rejected_csv_path is not None:
        pdb_ids.update(_read_pdb_ids_from_csv(rejected_csv_path))

    return frozenset(pdb_ids)


def _read_pdb_ids_from_csv(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return set()

        pdb_code_column = _find_pdb_code_column(reader.fieldnames)
        if pdb_code_column is None:
            return set()

        return {
            row[pdb_code_column].strip().casefold()
            for row in reader
            if row.get(pdb_code_column, "").strip()
        }


def _find_pdb_code_column(fieldnames: Iterable[str]) -> str | None:
    for fieldname in fieldnames:
        normalized = _normalize_column_name(fieldname)
        if normalized in {"pdbcode", "pdbid", "pdb"}:
            return fieldname

    return None


def _normalize_column_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _should_print_progress(attempted_count: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False

    return attempted_count == 1 or attempted_count % progress_every == 0


def _print_build_start(
    *,
    discovery: PdbRedoDiscoveryResult,
    candidates_to_attempt: int,
    already_processed_count: int,
    options: BnetDatabaseBuildOptions,
) -> None:
    print("Starting PDB-REDO Bnet reference database build")
    print(f"  PDB-REDO root: {options.pdb_redo_root}")
    print(f"  Accepted CSV: {options.accepted_csv_path}")

    if options.accepted_details_csv_path is not None:
        print(f"  Accepted details CSV: {options.accepted_details_csv_path}")
    if options.rejected_csv_path is not None:
        print(f"  Rejected CSV: {options.rejected_csv_path}")

    print(f"  Discovered candidates: {discovery.candidate_count}")
    print(f"  Discovery skipped: {discovery.skipped_count}")
    print(f"  Already processed in discovered set: {already_processed_count}")
    print(f"  Candidates to attempt: {candidates_to_attempt}")
    print(f"  Jobs: {options.jobs}")

    if options.max_tasks_in_flight is not None:
        print(f"  Max tasks in flight: {options.max_tasks_in_flight}")
    else:
        print(f"  Max tasks in flight: {options.jobs * 2}")

    print()


def _print_progress(
    *,
    attempted_count: int,
    total_count: int,
    accepted_count: int,
    rejected_count: int,
    result: PdbRedoProcessResult,
) -> None:
    status = "accepted" if result.is_accepted else "rejected"
    reason = ""

    if result.rejected is not None:
        reason = f" reason={result.rejected.reason}"

    print(
        f"[{attempted_count}/{total_count}] "
        f"{result.pdb_id.upper()} {status}{reason} "
        f"(accepted={accepted_count}, rejected={rejected_count})",
        flush=True,
    )


def _print_build_summary(
    summary: BnetDatabaseBuildSummary,
    writer: BnetDatabaseCsvWriter,
) -> None:
    print()
    print("Finished PDB-REDO Bnet reference database build")
    print(f"  Attempted this run: {summary.attempted_count}")
    print(f"  Accepted this run: {summary.accepted_count}")
    print(f"  Rejected this run: {summary.rejected_count}")
    print(f"  Accepted CSV total rows: {writer.accepted_count}")
    print(f"  Rejected CSV total rows: {writer.rejected_count}")
    print(f"  Elapsed seconds: {summary.elapsed_seconds:.1f}")

    if summary.elapsed_seconds > 0:
        rate = summary.attempted_count / summary.elapsed_seconds
        print(f"  Rate: {rate:.3f} entries/s")


def parse_args(argv: list[str] | None = None) -> BnetDatabaseBuildOptions:
    """Parse command-line arguments into build options."""

    parser = argparse.ArgumentParser(
        description="Build a Bnet reference database from a local PDB-REDO mirror.",
    )

    parser.add_argument(
        "pdb_redo_root",
        type=Path,
        help="Root directory of the local PDB-REDO mirror.",
    )
    parser.add_argument(
        "--accepted-csv",
        type=Path,
        default=DEFAULT_ACCEPTED_CSV_PATH,
        help=(
            "Output accepted Bnet database CSV. "
            f"Default: {DEFAULT_ACCEPTED_CSV_PATH}"
        ),
    )
    parser.add_argument(
        "--accepted-details-csv",
        type=Path,
        default=None,
        help="Optional detailed accepted-row CSV.",
    )
    parser.add_argument(
        "--rejected-csv",
        type=Path,
        default=None,
        help="Optional rejected-row CSV.",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--max-tasks-in-flight",
        type=int,
        default=None,
        help=(
            "Maximum submitted but unfinished worker tasks. "
            "Defaults to 2 * jobs."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Process at most this many unprocessed candidates.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N completed candidates. Use 0 to disable.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSVs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Do not skip PDB IDs already present in the accepted CSV and, "
            "when --rejected-csv is supplied, the rejected CSV."
        ),
    )
    parser.add_argument(
        "--no-recursive-discovery",
        action="store_true",
        help="Only inspect immediate child directories of the PDB-REDO root.",
    )
    parser.add_argument(
        "--allow-missing-data-json",
        action="store_true",
        help="Discover candidates even when data.json is missing.",
    )

    parser.add_argument(
        "--allow-non-xray",
        action="store_true",
        help="Do not reject entries that are not marked as X-ray crystallography.",
    )
    parser.add_argument(
        "--allow-multiple-models",
        action="store_true",
        help="Do not reject entries containing multiple models.",
    )
    parser.add_argument(
        "--allow-no-protein",
        action="store_true",
        help="Do not reject entries without a protein polymer.",
    )
    parser.add_argument(
        "--allow-nucleic-acid",
        action="store_true",
        help="Do not reject entries containing nucleic-acid polymers.",
    )
    parser.add_argument(
        "--include-traceback",
        action="store_true",
        help="Include traceback text in rejected CSV rows for exceptions.",
    )

    args = parser.parse_args(argv)

    return BnetDatabaseBuildOptions(
        pdb_redo_root=args.pdb_redo_root,
        accepted_csv_path=args.accepted_csv,
        accepted_details_csv_path=args.accepted_details_csv,
        rejected_csv_path=args.rejected_csv,
        require_data_json=not args.allow_missing_data_json,
        recursive_discovery=not args.no_recursive_discovery,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        max_candidates=args.max_candidates,
        progress_every=args.progress_every,
        jobs=args.jobs,
        max_tasks_in_flight=args.max_tasks_in_flight,
        require_xray=not args.allow_non_xray,
        require_single_model=not args.allow_multiple_models,
        require_protein=not args.allow_no_protein,
        reject_nucleic_acid=not args.allow_nucleic_acid,
        include_traceback=args.include_traceback,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    options = parse_args(argv)

    try:
        build_bnet_reference_database(options)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user.", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BnetDatabaseBuildOptions",
    "BnetDatabaseBuildSummary",
    "DEFAULT_ACCEPTED_CSV_PATH",
    "DEFAULT_DATABASE_CSV_PATH",
    "DEFAULT_OUTPUT_DIR",
    "build_bnet_reference_database",
    "main",
    "parse_args",
]
