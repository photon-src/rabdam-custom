"""Command-line interface for RABDAM."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import TextIO, TypeVar

from bdamage.score import (
    BDamageScoreError,
    bdamage_scores_as_tuple,
)
from rabdam import __version__
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    BDamageWorkflowResult,
    calculate_bdamage_for_structure_data,
)
from crystal.symmetry import CrystalSymmetryError
from crystal.translate import CrystalTranslationError
from crystal.trim import CrystalTrimError
from input.rcsb import ensure_local_structure_file
from input.reader import read_structure
from input.resolver import ResolvedStructureInput, resolve_structure_input
from packing.density import (
    PackingDensityError,
    packing_density_counts_as_tuple,
)
from structure.models import StructurePreparationOptions


DEFAULT_OUTPUT_CSV = Path("rabdam_BDamage.csv")
DEFAULT_CACHE_DIR = Path(".rabdam_cache") / "rcsb"
MISSING_STRUCTURE_INPUT_MESSAGE = """rabdam: error: missing required argument: structure_input

Usage:
  rabdam STRUCTURE_INPUT [options]

Examples:
  rabdam example.cif
  rabdam 1LYZ
  rabdam example.cif -o rabdam_BDamage.csv

Run 'rabdam --help' for all options."""

CSV_FIELDNAMES = [
    "REC",
    "ATMNUM",
    "ATMNAME",
    "CONFORMER",
    "RESNAME",
    "CHAIN",
    "RESNUM",
    "INSCODE",
    "XPOS",
    "YPOS",
    "ZPOS",
    "OCC",
    "BFAC",
    "ELEMENT",
    "CHARGE",
    "PD",
    "AVRG_BF",
    "BDAM",
]

RABDAM_CLI_ERRORS = (
    BDamageScoreError,
    BDamageWorkflowError,
    CrystalSymmetryError,
    CrystalTranslationError,
    CrystalTrimError,
    OSError,
    PackingDensityError,
    ValueError,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RabdamCliResult:
    """Result returned by the CLI runner."""

    local_input: ResolvedStructureInput
    output_csv: Path
    workflow_result: BDamageWorkflowResult


def _build_parser() -> argparse.ArgumentParser:
    """Build the RABDAM command-line parser."""

    defaults = BDamageWorkflowOptions()
    parser = argparse.ArgumentParser(
        prog="rabdam",
        usage="%(prog)s STRUCTURE_INPUT [options]",
        description="Run the RABDAM BDamage workflow for a structure file or PDB ID.",
        add_help=False,
    )
    input_group = parser.add_argument_group("input")
    general = parser.add_argument_group("general options")
    output_cache = parser.add_argument_group("output and cache options")
    bdamage = parser.add_argument_group("BDamage calculation parameters")
    selection = parser.add_argument_group("selection options")
    advanced = parser.add_argument_group("advanced/debugging options")

    input_group.add_argument(
        "structure_input",
        nargs="?",
        metavar="STRUCTURE_INPUT",
        help="Path to a .pdb/.cif/.mmcif file, or a 4-character PDB ID such as 1LYZ.",
    )
    general.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    general.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the program version and exit.",
    )
    general.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress and summary output.",
    )
    output_cache.add_argument(
        "-o",
        "--output-csv",
        metavar="PATH",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path for the per-atom BDamage CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    output_cache.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Directory for downloaded RCSB mmCIF files. Default: {DEFAULT_CACHE_DIR}",
    )
    output_cache.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Download RCSB inputs again even when a cached mmCIF exists.",
    )
    bdamage.add_argument(
        "--packing-density-threshold",
        metavar="FLOAT",
        type=positive_float,
        default=defaults.packing_density_threshold,
        help=(
            "Distance cutoff in Angstroms for neighbour counting. "
            f"Default: {defaults.packing_density_threshold}"
        ),
    )
    bdamage.add_argument(
        "--window-size-fraction",
        metavar="FLOAT",
        type=fraction_float,
        default=defaults.window_size_fraction,
        help=(
            "Fraction of selected atoms used to derive the BDamage window size. "
            f"Default: {defaults.window_size_fraction}"
        ),
    )
    bdamage.add_argument(
        "--minimum-window-size",
        metavar="INT",
        type=positive_int,
        default=defaults.minimum_window_size,
        help=f"Minimum BDamage sliding-window atom count. Default: {defaults.minimum_window_size}",
    )
    bdamage.add_argument(
        "--translation-range",
        metavar="INT",
        type=non_negative_int,
        default=defaults.translation_range,
        help=(
            "Number of unit-cell translations to include in each crystal direction. "
            f"Default: {defaults.translation_range}"
        ),
    )
    selection.add_argument(
        "--keep-hydrogens",
        action="store_true",
        help="Keep hydrogen atoms during structure preparation.",
    )
    selection.add_argument(
        "--include-hetatm",
        action="store_true",
        help="Include HETATM records in the BDamage atom selection.",
    )
    selection.add_argument(
        "--include-nucleic-acid",
        action="store_true",
        help="Include nucleic-acid atoms in the BDamage atom selection.",
    )
    selection.add_argument(
        "--remove-atom-serial",
        metavar="INT",
        action="append",
        default=[],
        type=positive_int,
        help="Remove an atom serial from the BDamage selection. May be used repeatedly.",
    )
    selection.add_argument(
        "--add-atom-serial",
        metavar="INT",
        action="append",
        default=[],
        type=positive_int,
        help="Add an atom serial to the BDamage selection. May be used repeatedly.",
    )
    selection.add_argument(
        "--remove-component",
        metavar="NAME",
        action="append",
        default=[],
        type=component_name,
        help="Remove a residue/component name from the BDamage selection.",
    )
    selection.add_argument(
        "--add-component",
        metavar="NAME",
        action="append",
        default=[],
        type=component_name,
        help="Add a residue/component name to the BDamage selection.",
    )
    advanced.add_argument(
        "--materialize-translated-block",
        action="store_true",
        help="Build the full translated crystal block for debugging.",
    )
    advanced.add_argument(
        "--allow-non-protein-selection",
        action="store_true",
        help="Allow BDamage scoring when the selected atoms contain no protein atoms.",
    )
    advanced.add_argument(
        "--preview-count",
        metavar="INT",
        type=non_negative_int,
        default=10,
        help="Number of leading packing-density and BDamage values to print. Default: 10",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = _build_parser()

    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    """Return a positive finite float for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def fraction_float(value: str) -> float:
    """Return a float in the open interval 0 to 1 for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a number greater than 0 and less than 1, got {value!r}"
        )
    return parsed


def positive_int(value: str) -> int:
    """Return a positive integer for argparse."""

    parsed = _int_arg(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    """Return a non-negative integer for argparse."""

    parsed = _int_arg(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def component_name(value: str) -> str:
    """Return a normalized residue/component name for argparse."""

    cleaned = value.strip().upper()
    if not cleaned:
        raise argparse.ArgumentTypeError("component name cannot be empty")
    return cleaned


def workflow_options_from_args(args: argparse.Namespace) -> BDamageWorkflowOptions:
    """Build workflow options from parsed command-line arguments."""

    return BDamageWorkflowOptions(
        packing_density_threshold=args.packing_density_threshold,
        window_size_fraction=args.window_size_fraction,
        minimum_window_size=args.minimum_window_size,
        translation_range=args.translation_range,
        materialize_translated_block=args.materialize_translated_block,
    )


def preparation_options_from_args(
    args: argparse.Namespace,
) -> StructurePreparationOptions:
    """Build structure-preparation options from parsed command-line arguments."""

    return StructurePreparationOptions(
        remove_hydrogens=not args.keep_hydrogens,
        include_hetatm_in_selection=args.include_hetatm,
        include_nucleic_acid_in_selection=args.include_nucleic_acid,
        require_protein_selection=not args.allow_non_protein_selection,
        remove_atom_serials=frozenset(args.remove_atom_serial),
        add_atom_serials=frozenset(args.add_atom_serial),
        remove_component_names=frozenset(args.remove_component),
        add_component_names=frozenset(args.add_component),
    )


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> RabdamCliResult:
    """Run the BDamage workflow from parsed command-line arguments."""

    workflow_options = workflow_options_from_args(args)
    preparation_options = preparation_options_from_args(args)

    resolved = run_stage(
        "Resolving input",
        lambda: resolve_structure_input(args.structure_input),
        stream=stderr,
        quiet=args.quiet,
    )
    local_input = run_stage(
        "Ensuring local structure file",
        lambda: ensure_local_structure_file(
            resolved,
            cache_dir=Path(args.cache_dir).expanduser(),
            overwrite=args.overwrite_cache,
        ),
        stream=stderr,
        quiet=args.quiet,
    )
    structure_data = run_stage(
        "Reading structure",
        lambda: read_structure(local_input),
        stream=stderr,
        quiet=args.quiet,
    )
    workflow_result = calculate_bdamage_for_structure_data(
        structure_data,
        workflow_options=workflow_options,
        preparation_options=preparation_options,
        stage_runner=lambda label, callback: run_stage(
            label,
            callback,
            stream=stderr,
            quiet=args.quiet,
        ),
    )

    output_csv = Path(args.output_csv).expanduser()
    run_stage(
        "Writing BDamage CSV",
        lambda: write_bdamage_csv(
            output_path=output_csv,
            prepared_structure=workflow_result.prepared_structure,
            bdamage_score_result=workflow_result.bdamage_score_result,
        ),
        stream=stderr,
        quiet=args.quiet,
    )

    cli_result = RabdamCliResult(
        local_input=local_input,
        output_csv=output_csv,
        workflow_result=workflow_result,
    )

    if not args.quiet:
        print_summary(cli_result, preview_count=args.preview_count, stream=stdout)

    return cli_result


def run_stage(
    label: str,
    callback: Callable[[], T],
    *,
    stream: TextIO,
    quiet: bool,
) -> T:
    """Run one CLI stage and optionally print elapsed time."""

    if quiet:
        return callback()

    print(f"{label}...", file=stream, flush=True)
    start = perf_counter()
    try:
        result = callback()
    except Exception:
        elapsed = perf_counter() - start
        print(f"{label} failed after {elapsed:.1f}s", file=stream, flush=True)
        raise

    elapsed = perf_counter() - start
    print(f"{label} done in {elapsed:.1f}s", file=stream, flush=True)
    return result


def print_total_runtime(start: float, *, stream: TextIO) -> None:
    """Print the total elapsed CLI runtime."""

    elapsed = perf_counter() - start
    print(f"Total runtime: {elapsed:.1f}s", file=stream, flush=True)


def print_missing_structure_input_error(*, stream: TextIO) -> None:
    """Print a compact recovery message for missing structure input."""

    print(MISSING_STRUCTURE_INPUT_MESSAGE, file=stream)


def write_bdamage_csv(
    *,
    output_path: Path,
    prepared_structure,
    bdamage_score_result,
) -> None:
    """Write per-atom BDamage results as a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for atom, score in zip(
            prepared_structure.selected_atoms,
            bdamage_score_result.atom_results,
            strict=True,
        ):
            record = atom.record
            writer.writerow(
                {
                    "REC": record.record_type,
                    "ATMNUM": record.atom_serial,
                    "ATMNAME": record.atom_name,
                    "CONFORMER": record.altloc or ".",
                    "RESNAME": record.residue_name,
                    "CHAIN": record.chain_id or ".",
                    "RESNUM": record.residue_number,
                    "INSCODE": record.insertion_code or ".",
                    "XPOS": record.x,
                    "YPOS": record.y,
                    "ZPOS": record.z,
                    "OCC": record.occupancy,
                    "BFAC": record.b_factor,
                    "ELEMENT": record.element,
                    "CHARGE": "?",
                    "PD": score.packing_density,
                    "AVRG_BF": score.average_b_factor,
                    "BDAM": score.bdamage,
                }
            )


def print_summary(
    result: RabdamCliResult,
    *,
    preview_count: int,
    stream: TextIO,
) -> None:
    """Print a compact successful-run summary."""

    workflow_result = result.workflow_result
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output CSV: {result.output_csv}", file=stream)
    print(
        f"Selected atoms: {workflow_result.prepared_structure.report.selected_atom_count}",
        file=stream,
    )
    print(f"Window size: {workflow_result.window_size}", file=stream)
    print(
        f"Symmetry-expanded atoms: {len(workflow_result.symmetry_expanded_structure.atoms)}",
        file=stream,
    )
    print(
        f"Translated atoms before trimming: {workflow_result.trimmed_block.original_atom_count}",
        file=stream,
    )
    print(
        f"Translated block materialized: {workflow_result.translated_block is not None}",
        file=stream,
    )
    print(
        f"Trimmed neighbour atoms: {workflow_result.trimmed_block.atom_count}",
        file=stream,
    )

    if preview_count > 0:
        packing_density_counts = packing_density_counts_as_tuple(
            workflow_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(
            workflow_result.bdamage_score_result
        )[:preview_count]
        print(f"Packing-density counts, first {preview_count}:", file=stream)
        print(packing_density_counts, file=stream)
        print(f"BDamage scores, first {preview_count}:", file=stream)
        print(bdamage_scores, file=stream)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the RABDAM command-line interface."""

    start = perf_counter()
    args = parse_args(argv)

    if args.structure_input is None:
        print_missing_structure_input_error(stream=stderr)
        return 2

    try:
        run_from_args(args, stdout=stdout, stderr=stderr)
    except RABDAM_CLI_ERRORS as error:
        print(f"rabdam: error: {error}", file=stderr)
        return 1
    finally:
        if not args.quiet:
            print_total_runtime(start, stream=stderr)

    return 0


def _float_arg(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a number, got {value!r}"
        ) from error


def _int_arg(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
