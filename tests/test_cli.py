import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bdamage.score import BDamageAtomResult, BDamageScoreResult
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from rabdam.cli import (
    MISSING_STRUCTURE_INPUT_MESSAGE,
    _build_parser,
    main,
    parse_args,
    preparation_options_from_args,
    workflow_options_from_args,
    write_bdamage_csv,
)
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationReport,
)


def make_prepared_structure() -> PreparedStructure:
    atom_record = AtomRecord(
        source_atom_index=0,
        model_number=1,
        chain_id="",
        residue_name="ALA",
        residue_number=12,
        insertion_code="",
        atom_name="CA",
        element="C",
        altloc="",
        x=1.0,
        y=2.0,
        z=3.0,
        occupancy=1.0,
        b_factor=15.0,
        atom_serial=42,
        record_type="ATOM",
    )
    atom = PreparedAtom(
        record=atom_record,
        is_hydrogen=False,
        is_protein=True,
        is_nucleic_acid=False,
        is_solvent=False,
        is_hetatm=False,
    )
    metadata = StructureMetadata(
        source_path=Path("example.cif"),
        structure_id=None,
        file_format=StructureFileFormat.MMCIF,
        space_group="P 1",
        unit_cell_a=10.0,
        unit_cell_b=10.0,
        unit_cell_c=10.0,
        unit_cell_alpha=90.0,
        unit_cell_beta=90.0,
        unit_cell_gamma=90.0,
    )
    report = StructurePreparationReport(
        input_atom_count=1,
        cleaned_atom_count=1,
        selected_atom_count=1,
        removed_hydrogen_count=0,
        removed_invalid_coordinate_count=0,
        removed_invalid_occupancy_count=0,
        removed_invalid_b_factor_count=0,
        removed_altloc_count=0,
    )
    return PreparedStructure(
        cleaned_atoms=(atom,),
        selected_atoms=(atom,),
        metadata=metadata,
        report=report,
    )


class CliArgumentTests(unittest.TestCase):
    def test_builds_workflow_and_preparation_options_from_args(self) -> None:
        args = parse_args(
            [
                "example.cif",
                "--output-csv",
                "out.csv",
                "--cache-dir",
                "cache",
                "--overwrite-cache",
                "--packing-density-threshold",
                "8.5",
                "--window-size-fraction",
                "0.1",
                "--minimum-window-size",
                "3",
                "--translation-range",
                "2",
                "--materialize-translated-block",
                "--keep-hydrogens",
                "--include-hetatm",
                "--include-nucleic-acid",
                "--allow-non-protein-selection",
                "--remove-atom-serial",
                "10",
                "--add-atom-serial",
                "11",
                "--remove-component",
                "hoh",
                "--add-component",
                "lig",
                "--preview-count",
                "0",
                "--quiet",
            ]
        )

        workflow_options = workflow_options_from_args(args)
        preparation_options = preparation_options_from_args(args)

        self.assertEqual(args.structure_input, "example.cif")
        self.assertEqual(args.output_csv, "out.csv")
        self.assertEqual(args.cache_dir, "cache")
        self.assertTrue(args.overwrite_cache)
        self.assertEqual(workflow_options.packing_density_threshold, 8.5)
        self.assertEqual(workflow_options.window_size_fraction, 0.1)
        self.assertEqual(workflow_options.minimum_window_size, 3)
        self.assertEqual(workflow_options.translation_range, 2)
        self.assertTrue(workflow_options.materialize_translated_block)
        self.assertFalse(preparation_options.remove_hydrogens)
        self.assertTrue(preparation_options.include_hetatm_in_selection)
        self.assertTrue(preparation_options.include_nucleic_acid_in_selection)
        self.assertFalse(preparation_options.require_protein_selection)
        self.assertEqual(preparation_options.remove_atom_serials, frozenset({10}))
        self.assertEqual(preparation_options.add_atom_serials, frozenset({11}))
        self.assertEqual(preparation_options.remove_component_names, frozenset({"HOH"}))
        self.assertEqual(preparation_options.add_component_names, frozenset({"LIG"}))
        self.assertEqual(args.preview_count, 0)
        self.assertTrue(args.quiet)

    def test_invalid_input_returns_cli_error(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        exit_code = main(
            ["not-a-structure", "--quiet"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("rabdam: error:", stderr.getvalue())
        self.assertIn("Could not resolve structure input", stderr.getvalue())

    def test_missing_input_returns_short_recovery_error(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch("rabdam.cli.run_from_args") as run_from_args:
            exit_code = main([], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), f"{MISSING_STRUCTURE_INPUT_MESSAGE}\n")
        self.assertNotIn("--materialize-translated-block", stderr.getvalue())
        self.assertNotIn("Total runtime", stderr.getvalue())
        run_from_args.assert_not_called()

    def test_quiet_missing_input_returns_short_recovery_error(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch("rabdam.cli.run_from_args") as run_from_args:
            exit_code = main(["--quiet"], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), f"{MISSING_STRUCTURE_INPUT_MESSAGE}\n")
        self.assertNotIn("Total runtime", stderr.getvalue())
        run_from_args.assert_not_called()

    def test_help_is_grouped_by_option_tier(self) -> None:
        help_text = _build_parser().format_help()

        self.assertIn("usage: rabdam STRUCTURE_INPUT [options]", help_text)
        self.assertIn("input:", help_text)
        self.assertIn("general options:", help_text)
        self.assertIn("output and cache options:", help_text)
        self.assertIn("BDamage calculation parameters:", help_text)
        self.assertIn("selection options:", help_text)
        self.assertIn("advanced/debugging options:", help_text)

        input_start = help_text.index("input:")
        general_start = help_text.index("general options:")
        output_cache_start = help_text.index("output and cache options:")
        bdamage_start = help_text.index("BDamage calculation parameters:")
        selection_start = help_text.index("selection options:")
        advanced_start = help_text.index("advanced/debugging options:")

        input_help = help_text[input_start:general_start]
        general_help = help_text[general_start:output_cache_start]
        output_cache_help = help_text[output_cache_start:bdamage_start]
        bdamage_help = help_text[bdamage_start:selection_start]
        selection_help = help_text[selection_start:advanced_start]
        advanced_help = help_text[advanced_start:]
        normalized_bdamage_help = " ".join(bdamage_help.split())
        normalized_selection_help = " ".join(selection_help.split())
        normalized_advanced_help = " ".join(advanced_help.split())

        self.assertIn("STRUCTURE_INPUT", input_help)
        self.assertIn("-h, --help", general_help)
        self.assertIn("Show this help message and exit.", general_help)
        self.assertIn("--version", general_help)
        self.assertIn("Show the program version and exit.", general_help)
        self.assertIn("-q, --quiet", general_help)
        self.assertIn("-o PATH, --output-csv PATH", output_cache_help)
        self.assertIn("--cache-dir DIR", output_cache_help)
        self.assertIn("--overwrite-cache", output_cache_help)
        self.assertIn("--packing-density-threshold FLOAT", bdamage_help)
        self.assertIn("--translation-range INT", bdamage_help)
        self.assertIn(
            "Number of unit-cell translations to include in each crystal direction.",
            normalized_bdamage_help,
        )
        self.assertIn("--keep-hydrogens", selection_help)
        self.assertIn(
            "Include HETATM records in the BDamage atom selection.",
            normalized_selection_help,
        )
        self.assertIn(
            "Include nucleic-acid atoms in the BDamage atom selection.",
            normalized_selection_help,
        )
        self.assertIn("--remove-component NAME", selection_help)
        self.assertIn("--materialize-translated-block", advanced_help)
        self.assertIn("--allow-non-protein-selection", advanced_help)
        self.assertIn(
            "Allow BDamage scoring when the selected atoms contain no protein atoms.",
            normalized_advanced_help,
        )
        self.assertIn("--preview-count INT", advanced_help)

    def test_success_reports_total_runtime(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch("rabdam.cli.perf_counter", side_effect=[1.0, 3.34]),
            patch("rabdam.cli.run_from_args") as run_from_args,
        ):
            exit_code = main(["example.cif"], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Total runtime: 2.3s", stderr.getvalue())
        run_from_args.assert_called_once()

    def test_quiet_success_suppresses_total_runtime(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch("rabdam.cli.run_from_args") as run_from_args:
            exit_code = main(
                ["example.cif", "--quiet"],
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        run_from_args.assert_called_once()


class BDamageCsvWriterTests(unittest.TestCase):
    def test_write_bdamage_csv_creates_parent_directory_and_rows(self) -> None:
        prepared_structure = make_prepared_structure()
        score_result = BDamageScoreResult(
            atom_results=(
                BDamageAtomResult(
                    bdamage_atom_index=1,
                    source_atom_index=0,
                    atom_serial=42,
                    b_factor=15.0,
                    packing_density=7,
                    average_b_factor=12.5,
                    bdamage=1.2,
                    sorted_packing_density_index=1,
                ),
            ),
            window_size=1,
            selected_atom_count=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "bdamage.csv"

            write_bdamage_csv(
                output_path=output_path,
                prepared_structure=prepared_structure,
                bdamage_score_result=score_result,
            )

            rows = list(csv.DictReader(output_path.read_text().splitlines()))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["REC"], "ATOM")
        self.assertEqual(rows[0]["ATMNUM"], "42")
        self.assertEqual(rows[0]["ATMNAME"], "CA")
        self.assertEqual(rows[0]["CHAIN"], ".")
        self.assertEqual(rows[0]["CONFORMER"], ".")
        self.assertEqual(rows[0]["PD"], "7")
        self.assertEqual(rows[0]["AVRG_BF"], "12.5")
        self.assertEqual(rows[0]["BDAM"], "1.2")


if __name__ == "__main__":
    unittest.main()
