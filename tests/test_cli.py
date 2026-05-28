import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bdamage.score import BDamageAtomResult, BDamageScoreResult
from crystal.symmetry import (
    SymmetryExpandedAtom,
    SymmetryExpandedStructure,
    UnitCellParameters,
)
from crystal.trim import CartesianBounds, TrimmedCrystalBlock
from input.reader import AtomRecord, StructureMetadata
from input.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)
from packing.density import PackingDensityAtomResult, PackingDensityResult
from rabdam.cli import (
    CITATIONS_MESSAGE,
    MISSING_STRUCTURE_INPUT_MESSAGE,
    RabdamCliResult,
    _build_parser,
    main,
    parse_args,
    preparation_options_from_args,
    print_summary,
    run_from_args,
    run_stage,
    workflow_options_from_args,
    write_bdamage_csv,
)
from rabdam.workflow import BDamageWorkflowOptions, BDamageWorkflowResult
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


def make_cli_result(*, translated_block=None) -> RabdamCliResult:
    unit_cell = UnitCellParameters(
        a=10.0,
        b=10.0,
        c=10.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
    )
    symmetry_expanded_structure = SymmetryExpandedStructure(
        atoms=(
            SymmetryExpandedAtom(1, 0, 1, "x,y,z", True, 1.0, 2.0, 3.0),
            SymmetryExpandedAtom(2, 0, 2, "-x,-y,-z", False, 4.0, 5.0, 6.0),
            SymmetryExpandedAtom(3, 0, 3, "x+1/2,y,z", False, 7.0, 8.0, 9.0),
        ),
        unit_cell=unit_cell,
        space_group_name="P 1",
        operation_count=3,
    )
    bounds = CartesianBounds(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
    )
    trimmed_block = TrimmedCrystalBlock(
        atoms=(object(), object()),
        reference_bounds=bounds,
        trim_bounds=bounds,
        padding=7.0,
        original_atom_count=27,
    )
    packing_density_result = PackingDensityResult(
        atom_results=(
            PackingDensityAtomResult(1, 0, 42, 7),
            PackingDensityAtomResult(2, 1, 43, 8),
            PackingDensityAtomResult(3, 2, 44, 9),
        ),
        packing_density_threshold=7.0,
        selected_atom_count=3,
        neighbour_atom_count=2,
    )
    bdamage_score_result = BDamageScoreResult(
        atom_results=(
            BDamageAtomResult(1, 0, 42, 15.0, 7, 12.0, 1.23456, 1),
            BDamageAtomResult(2, 1, 43, 16.0, 8, 13.0, 0.98765, 2),
            BDamageAtomResult(3, 2, 44, 17.0, 9, 14.0, 1.11111, 3),
        ),
        window_size=1,
        selected_atom_count=3,
    )
    workflow_result = BDamageWorkflowResult(
        prepared_structure=make_prepared_structure(),
        symmetry_expanded_structure=symmetry_expanded_structure,
        translated_block=translated_block,
        trimmed_block=trimmed_block,
        packing_density_result=packing_density_result,
        bdamage_score_result=bdamage_score_result,
        options=BDamageWorkflowOptions(),
        window_size=1,
    )
    local_input = ResolvedStructureInput(
        original_input="example.cif",
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=Path("example.cif"),
    )
    return RabdamCliResult(
        local_input=local_input,
        output_csv=Path("out.csv"),
        workflow_result=workflow_result,
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

    def test_preview_count_defaults_to_zero(self) -> None:
        args = parse_args(["example.cif"])

        self.assertEqual(args.preview_count, 0)

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

    def test_citations_prints_citation_information_without_input(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch("rabdam.cli.run_from_args") as run_from_args:
            exit_code = main(["--citations"], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), f"{CITATIONS_MESSAGE}\n")
        self.assertEqual(stderr.getvalue(), "")
        run_from_args.assert_not_called()

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
        self.assertIn("--citations", general_help)
        self.assertIn("Print citation information and exit.", general_help)
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
        self.assertIn(
            "Number of leading packing-density and BDamage values to print. Default: 0",
            normalized_advanced_help,
        )

    def test_success_passes_total_runtime_start_to_runner(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch("rabdam.cli.perf_counter", return_value=1.0),
            patch("rabdam.cli.run_from_args") as run_from_args,
        ):
            exit_code = main(["example.cif"], stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(run_from_args.call_count, 1)
        self.assertEqual(run_from_args.call_args.kwargs["stdout"], stdout)
        self.assertEqual(run_from_args.call_args.kwargs["stderr"], stderr)
        self.assertEqual(run_from_args.call_args.kwargs["total_runtime_start"], 1.0)

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


class CliProgressTests(unittest.TestCase):
    def test_run_stage_prints_indented_completion(self) -> None:
        stream = StringIO()

        with patch("rabdam.cli.perf_counter", side_effect=[1.0, 1.23]):
            result = run_stage(
                "Reading structure",
                lambda: "ok",
                stream=stream,
                quiet=False,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            stream.getvalue(),
            "Reading structure...\n"
            "  done in 0.2s\n",
        )

    def test_run_stage_prints_indented_failure(self) -> None:
        stream = StringIO()

        def fail() -> None:
            raise ValueError("boom")

        with (
            patch("rabdam.cli.perf_counter", side_effect=[1.0, 1.23]),
            self.assertRaises(ValueError),
        ):
            run_stage(
                "Reading structure",
                fail,
                stream=stream,
                quiet=False,
            )

        self.assertEqual(
            stream.getvalue(),
            "Reading structure...\n"
            "  failed after 0.2s\n",
        )


class CliSummaryTests(unittest.TestCase):
    def test_default_summary_is_grouped_without_preview_or_debug(self) -> None:
        stdout = StringIO()

        print_summary(
            make_cli_result(),
            preview_count=0,
            total_runtime_seconds=2.34,
            stream=stdout,
        )

        self.assertEqual(
            stdout.getvalue(),
            "\n"
            "Done.\n"
            "\n"
            "Input: example.cif\n"
            "Output CSV: out.csv\n"
            "\n"
            "Crystallographic expansion:\n"
            "  Symmetry-expanded atoms: 3\n"
            "  Translated atoms before trimming: 27\n"
            "  Neighbour-block atoms after trimming: 2\n"
            "\n"
            "BDamage summary:\n"
            "  Selected atoms: 1\n"
            "  Window size: 1\n"
            "\n"
            "Total runtime: 2.3s\n",
        )
        self.assertNotIn("Preview:", stdout.getvalue())
        self.assertNotIn("Translated block materialized", stdout.getvalue())

    def test_summary_prints_preview_only_when_requested(self) -> None:
        stdout = StringIO()

        print_summary(
            make_cli_result(),
            preview_count=2,
            total_runtime_seconds=2.34,
            stream=stdout,
        )

        self.assertEqual(
            stdout.getvalue(),
            "\n"
            "Done.\n"
            "\n"
            "Input: example.cif\n"
            "Output CSV: out.csv\n"
            "\n"
            "Crystallographic expansion:\n"
            "  Symmetry-expanded atoms: 3\n"
            "  Translated atoms before trimming: 27\n"
            "  Neighbour-block atoms after trimming: 2\n"
            "\n"
            "BDamage summary:\n"
            "  Selected atoms: 1\n"
            "  Window size: 1\n"
            "\n"
            "Preview:\n"
            "  Packing-density counts, first 2:\n"
            "    7, 8\n"
            "  BDamage scores, first 2:\n"
            "    1.235, 0.988\n"
            "\n"
            "Total runtime: 2.3s\n",
        )

    def test_summary_prints_debug_section_for_materialized_block(self) -> None:
        stdout = StringIO()

        print_summary(
            make_cli_result(translated_block=object()),
            preview_count=0,
            total_runtime_seconds=2.34,
            stream=stdout,
        )

        self.assertEqual(
            stdout.getvalue(),
            "\n"
            "Done.\n"
            "\n"
            "Input: example.cif\n"
            "Output CSV: out.csv\n"
            "\n"
            "Crystallographic expansion:\n"
            "  Symmetry-expanded atoms: 3\n"
            "  Translated atoms before trimming: 27\n"
            "  Neighbour-block atoms after trimming: 2\n"
            "\n"
            "BDamage summary:\n"
            "  Selected atoms: 1\n"
            "  Window size: 1\n"
            "\n"
            "Debug:\n"
            "  Translated block materialized: True\n"
            "\n"
            "Total runtime: 2.3s\n",
        )

    def test_run_from_args_includes_total_runtime_in_stdout_summary(self) -> None:
        args = parse_args(["example.cif"])
        stdout = StringIO()
        stderr = StringIO()
        resolved_input = make_cli_result().local_input

        with (
            patch(
                "rabdam.cli.run_stage",
                side_effect=lambda label, callback, **_: callback(),
            ),
            patch("rabdam.cli.resolve_structure_input", return_value=resolved_input),
            patch("rabdam.cli.ensure_local_structure_file", return_value=resolved_input),
            patch("rabdam.cli.read_structure", return_value=object()),
            patch(
                "rabdam.cli.calculate_bdamage_for_structure_data",
                return_value=make_cli_result().workflow_result,
            ),
            patch("rabdam.cli.write_bdamage_csv"),
            patch("rabdam.cli.perf_counter", return_value=3.34),
        ):
            run_from_args(
                args,
                stdout=stdout,
                stderr=stderr,
                total_runtime_start=1.0,
            )

        self.assertIn("Total runtime: 2.3s\n", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")


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
