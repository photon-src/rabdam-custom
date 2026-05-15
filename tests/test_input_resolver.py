import tempfile
import unittest
from pathlib import Path

from input.resolver import (
    InputBatchResolutionError,
    InputResolutionError,
    StructureSourceType,
    resolve_many_structure_inputs,
)


class ResolveManyStructureInputsTests(unittest.TestCase):
    def test_resolves_all_valid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            structure_path = Path(temp_dir) / "model.cif"
            structure_path.write_text("data_model\n")

            resolved_inputs = resolve_many_structure_inputs(
                [str(structure_path), "2blx"]
            )

        self.assertEqual(len(resolved_inputs), 2)
        self.assertEqual(resolved_inputs[0].source_type, StructureSourceType.LOCAL_FILE)
        self.assertEqual(resolved_inputs[1].source_type, StructureSourceType.RCSB_ID)
        self.assertEqual(resolved_inputs[1].structure_id, "2BLX")

    def test_collects_all_resolution_errors(self) -> None:
        with self.assertRaises(InputBatchResolutionError) as context:
            resolve_many_structure_inputs(
                [
                    "not-an-input",
                    "2blx",
                    "",
                    "also-bad",
                ]
            )

        error = context.exception

        self.assertEqual(
            [raw_input for raw_input, _ in error.errors],
            [
                "not-an-input",
                "",
                "also-bad",
            ],
        )
        self.assertIn("One or more structure inputs could not be resolved", str(error))
        self.assertIn("'not-an-input'", str(error))
        self.assertIn("''", str(error))
        self.assertIn("'also-bad'", str(error))

    def test_empty_batch_still_raises_single_resolution_error(self) -> None:
        with self.assertRaises(InputResolutionError) as context:
            resolve_many_structure_inputs([])

        self.assertNotIsInstance(context.exception, InputBatchResolutionError)
        self.assertEqual(str(context.exception), "No structure inputs were provided.")


if __name__ == "__main__":
    unittest.main()
