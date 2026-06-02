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
from bnet.calculate import (
    ProteinBnetCalculationError,
    ProteinBnetResult,
    calculate_protein_bnet,
)
from bnet.percentile import BnetPercentileResult, calculate_bnet_percentile
from bnet.reference import (
    DEFAULT_REFERENCE_DATABASE_CSV_PATH,
    load_default_bnet_reference_database,
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
from input.rcsb import DEFAULT_RCSB_DOWNLOAD_DIR, ensure_local_structure_file
from input.reader import read_structure
from input.resolver import ResolvedStructureInput, resolve_structure_input
from packing.density import (
    PackingDensityError,
    packing_density_counts_as_tuple,
)
from structure.models import StructurePreparationOptions


DEFAULT_OUTPUT_CSV = Path("rabdam_BDamage.csv")
DEFAULT_DOWNLOAD_DIR = DEFAULT_RCSB_DOWNLOAD_DIR
CITATIONS_MESSAGE = """Please cite:

RABDAM:
  Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
  J Appl Cryst 51, 552-559.

BDamage:
  Gerstel M, Deane CM & Garman EF (2015).
  J Synchrotron Radiat 22, 201-212.

Bnet / Bnet percentile:
  Shelley KL & Garman EF (2022).
  Nat Commun 13, 1314."""
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
    ProteinBnetCalculationError,
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
    bnet_result: ProteinBnetResult
    bnet_percentile_result: BnetPercentileResult | None = None
    bnet_percentile_unavailable_reason: str | None = None


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
    output_download = parser.add_argument_group("output and download options")
    bdamage = parser.add_argument_group("BDamage calculation parameters")
    bnet = parser.add_argument_group("Bnet percentile options")
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
        "--citations",
        action="store_true",
        help="Print citation information and exit.",
    )
    general.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress and summary output.",
    )
    output_download.add_argument(
        "-o",
        "--output-csv",
        metavar="PATH",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path for the per-atom BDamage CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    output_download.add_argument(
        "--download-dir",
        metavar="DIR",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=(
            "Directory for downloaded RCSB mmCIF files. "
            f"Default: {DEFAULT_DOWNLOAD_DIR}"
        ),
    )
    output_download.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Download RCSB inputs again even when the mmCIF file already exists.",
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
    bnet.add_argument(
        "--bnet-resolution-angstrom",
        metavar="FLOAT",
        type=positive_float,
        default=None,
        help=(
            "Resolution in Angstroms for Bnet percentile calculation. "
            "Overrides resolution read from the structure file."
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
        default=0,
        help="Number of leading packing-density and BDamage values to print. Default: 0",
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
    total_runtime_start: float | None = None,
) -> RabdamCliResult:
    """Run the BDamage workflow from parsed command-line arguments."""

    if total_runtime_start is None:
        total_runtime_start = perf_counter()

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
            download_dir=Path(args.download_dir).expanduser(),
            overwrite=args.overwrite_download,
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

    bnet_result = run_stage(
        "Calculating protein Bnet",
        lambda: calculate_protein_bnet(
            prepared_structure=workflow_result.prepared_structure,
            bdamage_score_result=workflow_result.bdamage_score_result,
        ),
        stream=stderr,
        quiet=args.quiet,
    )

    bnet_percentile_result, bnet_percentile_unavailable_reason = run_stage(
        "Calculating Bnet percentile",
        lambda: calculate_default_bnet_percentile(
            bnet_result=bnet_result,
            resolution_angstrom=(
                args.bnet_resolution_angstrom
                or workflow_result.prepared_structure.metadata.resolution_angstrom
            ),
        ),
        stream=stderr,
        quiet=args.quiet,
    )

    cli_result = RabdamCliResult(
        local_input=local_input,
        output_csv=output_csv,
        workflow_result=workflow_result,
        bnet_result=bnet_result,
        bnet_percentile_result=bnet_percentile_result,
        bnet_percentile_unavailable_reason=bnet_percentile_unavailable_reason,
    )

    if not args.quiet:
        total_runtime_seconds = perf_counter() - total_runtime_start
        print_summary(
            cli_result,
            preview_count=args.preview_count,
            total_runtime_seconds=total_runtime_seconds,
            stream=stdout,
        )

    return cli_result


def calculate_default_bnet_percentile(
    *,
    bnet_result: ProteinBnetResult,
    resolution_angstrom: float | None,
) -> tuple[BnetPercentileResult | None, str | None]:
    """Calculate Bnet percentile with the default swappable reference database."""

    reference_database = load_default_bnet_reference_database()
    if reference_database is None:
        return (
            None,
            f"default reference database not found: {DEFAULT_REFERENCE_DATABASE_CSV_PATH}",
        )

    if resolution_angstrom is None:
        return (
            None,
            "structure resolution is unavailable; use --bnet-resolution-angstrom",
        )

    return (
        calculate_bnet_percentile(
            bnet=bnet_result.bnet,
            resolution_angstrom=resolution_angstrom,
            reference_database=reference_database,
        ),
        None,
    )


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
        print(f"  failed after {elapsed:.1f}s", file=stream, flush=True)
        raise

    elapsed = perf_counter() - start
    print(f"  done in {elapsed:.1f}s", file=stream, flush=True)
    return result


def print_missing_structure_input_error(*, stream: TextIO) -> None:
    """Print a compact recovery message for missing structure input."""

    print(MISSING_STRUCTURE_INPUT_MESSAGE, file=stream)


def print_citations(*, stream: TextIO) -> None:
    """Print citation information for RABDAM outputs."""

    print(CITATIONS_MESSAGE, file=stream)


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
    total_runtime_seconds: float,
    stream: TextIO,
) -> None:
    """Print a compact successful-run summary."""

    workflow_result = result.workflow_result
    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output CSV: {result.output_csv}", file=stream)
    print(file=stream)
    print("Crystallographic expansion:", file=stream)
    print(
        "  Symmetry-expanded atoms: "
        f"{len(workflow_result.symmetry_expanded_structure.atoms)}",
        file=stream,
    )
    print(
        "  Translated atoms before trimming: "
        f"{workflow_result.trimmed_block.original_atom_count}",
        file=stream,
    )
    print(
        "  Neighbour-block atoms after trimming: "
        f"{workflow_result.trimmed_block.atom_count}",
        file=stream,
    )
    print(file=stream)
    print("BDamage summary:", file=stream)
    print(
        f"  Selected atoms: {workflow_result.prepared_structure.report.selected_atom_count}",
        file=stream,
    )
    print(f"  Window size: {workflow_result.window_size}", file=stream)
    print(file=stream)
    print("Bnet summary:", file=stream)
    print(f"  Protein Bnet sites: {result.bnet_result.site_count}", file=stream)
    print(f"  Raw protein Bnet: {result.bnet_result.bnet:.4f}", file=stream)
    print_bnet_percentile_summary(result, stream=stream)

    if preview_count > 0:
        packing_density_counts = packing_density_counts_as_tuple(
            workflow_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(
            workflow_result.bdamage_score_result
        )[:preview_count]
        print(file=stream)
        print("Preview:", file=stream)
        print(f"  Packing-density counts, first {preview_count}:", file=stream)
        print(f"    {_format_int_values(packing_density_counts)}", file=stream)
        print(f"  BDamage scores, first {preview_count}:", file=stream)
        print(f"    {_format_float_values(bdamage_scores)}", file=stream)

    if workflow_result.translated_block is not None:
        print(file=stream)
        print("Debug:", file=stream)
        print("  Translated block materialized: True", file=stream)

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


def _format_int_values(values: Sequence[int]) -> str:
    """Return integer preview values as a comma-separated string."""

    return ", ".join(str(value) for value in values)


def print_bnet_percentile_summary(
    result: RabdamCliResult,
    *,
    stream: TextIO,
) -> None:
    """Print Bnet percentile summary lines."""

    percentile_result = result.bnet_percentile_result
    if percentile_result is None:
        reason = result.bnet_percentile_unavailable_reason or "not calculated"
        print(f"  Bnet percentile: unavailable ({reason})", file=stream)
        return

    print(
        f"  Bnet percentile: {percentile_result.percentile_percent:.2f}%",
        file=stream,
    )
    print(
        f"  Bnet reference: {percentile_result.reference_database_id} "
        f"({percentile_result.reference_entry_count} entries)",
        file=stream,
    )
    print(
        f"  Resolution used: {percentile_result.resolution_angstrom:.3g} A",
        file=stream,
    )
    print(
        "  Local reference set: "
        f"{percentile_result.local_reference_count} entries, "
        f"{percentile_result.local_resolution_min:.3g}-"
        f"{percentile_result.local_resolution_max:.3g} A",
        file=stream,
    )


def _format_float_values(values: Sequence[float]) -> str:
    """Return float preview values as a comma-separated string."""

    return ", ".join(f"{value:.3f}" for value in values)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the RABDAM command-line interface."""

    start = perf_counter()
    args = parse_args(argv)

    if args.citations:
        print_citations(stream=stdout)
        return 0

    if args.structure_input is None:
        print_missing_structure_input_error(stream=stderr)
        return 2

    try:
        run_from_args(
            args,
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=start,
        )
    except RABDAM_CLI_ERRORS as error:
        print(f"rabdam: error: {error}", file=stderr)
        return 1

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
