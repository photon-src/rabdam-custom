import argparse
from collections.abc import Callable
import csv
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from bdamage.score import bdamage_scores_as_tuple
from bdamage.score import calculate_bdamage_scores_for_structure
from bdamage.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    bdamage_window_size_from_fraction,
    validate_workflow_options,
)
from crystal.symmetry import expand_prepared_structure_by_symmetry
from crystal.translate import translate_expanded_unit_cell
from crystal.trim import trim_translated_block_for_bdamage
from input.rcsb import ensure_local_structure_file
from input.reader import read_structure
from input.resolver import resolve_structure_input
from packing.density import calculate_bdamage_packing_density
from packing.density import packing_density_counts_as_tuple
from structure.prepare import prepare_structure


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RABDAM workflow for a local structure file or PDB ID."
    )
    parser.add_argument(
        "structure_input",
        nargs="?",
        default="example.cif",
        help="Path to a .pdb/.cif/.mmcif file, or a 4-character PDB ID such as 2blx.",
    )
    parser.add_argument(
        "--output-csv",
        default="rabdam3_BDamage.csv",
        help="Path for the per-atom RABDAM 3 BDamage CSV.",
    )
    return parser.parse_args()


def run_stage(label: str, callback: Callable[[], T]) -> T:
    print(f"{label}...", flush=True)
    start = perf_counter()
    result = callback()
    elapsed = perf_counter() - start
    print(f"{label} done in {elapsed:.1f}s", flush=True)
    return result


def write_bdamage_csv(
    *,
    output_path: Path,
    prepared_structure,
    bdamage_score_result,
) -> None:
    fieldnames = [
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

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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


def main() -> None:
    args = parse_args()
    workflow_options = BDamageWorkflowOptions()
    validate_workflow_options(workflow_options)

    resolved = run_stage(
        "Resolving input",
        lambda: resolve_structure_input(args.structure_input),
    )
    local_input = run_stage(
        "Ensuring local structure file",
        lambda: ensure_local_structure_file(resolved),
    )
    structure_data = run_stage(
        "Reading structure",
        lambda: read_structure(local_input),
    )
    prepared_structure = run_stage(
        "Preparing structure",
        lambda: prepare_structure(structure_data),
    )

    selected_atom_count = len(prepared_structure.selected_atoms)
    if selected_atom_count == 0:
        raise BDamageWorkflowError(
            "Cannot calculate BDamage for a prepared structure with no selected atoms."
        )

    window_size = bdamage_window_size_from_fraction(
        atom_count=selected_atom_count,
        window_size_fraction=workflow_options.window_size_fraction,
        minimum_window_size=workflow_options.minimum_window_size,
    )
    if window_size > selected_atom_count:
        raise BDamageWorkflowError(
            "Calculated BDamage window size is larger than the number of selected atoms: "
            f"window_size={window_size!r}, selected_atom_count={selected_atom_count!r}."
        )

    symmetry_expanded_structure = run_stage(
        "Expanding crystallographic symmetry",
        lambda: expand_prepared_structure_by_symmetry(prepared_structure),
    )
    translated_block = run_stage(
        "Translating neighbouring unit cells",
        lambda: translate_expanded_unit_cell(
            symmetry_expanded_structure,
            translation_range=workflow_options.translation_range,
        ),
    )
    trimmed_block = run_stage(
        "Trimming neighbour block",
        lambda: trim_translated_block_for_bdamage(
            translated_block=translated_block,
            prepared_structure=prepared_structure,
            padding=workflow_options.packing_density_threshold,
        ),
    )
    packing_density_result = run_stage(
        "Calculating packing density",
        lambda: calculate_bdamage_packing_density(
            prepared_structure=prepared_structure,
            trimmed_block=trimmed_block,
            packing_density_threshold=workflow_options.packing_density_threshold,
        ),
    )
    bdamage_score_result = run_stage(
        "Calculating BDamage scores",
        lambda: calculate_bdamage_scores_for_structure(
            prepared_structure=prepared_structure,
            packing_density_result=packing_density_result,
            window_size=window_size,
        ),
    )
    output_csv = Path(args.output_csv)
    run_stage(
        "Writing RABDAM 3 BDamage CSV",
        lambda: write_bdamage_csv(
            output_path=output_csv,
            prepared_structure=prepared_structure,
            bdamage_score_result=bdamage_score_result,
        ),
    )

    print("Input:", local_input.local_path)
    print("Output CSV:", output_csv)
    print("Selected atoms:", prepared_structure.report.selected_atom_count)
    print("Window size:", window_size)
    print("Symmetry-expanded atoms:", len(symmetry_expanded_structure.atoms))
    print("Translated atoms:", len(translated_block.atoms))
    print("Trimmed neighbour atoms:", len(trimmed_block.atoms))
    print("Packing-density counts, first 10:")
    print(packing_density_counts_as_tuple(packing_density_result)[:10])
    print("BDamage scores, first 10:")
    print(bdamage_scores_as_tuple(bdamage_score_result)[:10])


if __name__ == "__main__":
    main()
