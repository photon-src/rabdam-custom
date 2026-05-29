import math
import unittest

from bnet.percentile import (
    BNET_PERCENTILE_RANK_METHOD,
    BnetPercentileError,
    calculate_bnet_percentile,
    find_resolution_matched_reference_set,
)
from bnet.reference import (
    BnetReferenceDatabase,
    BnetReferenceEntry,
    BnetReferenceMetadata,
)


def make_reference_database() -> BnetReferenceDatabase:
    return BnetReferenceDatabase(
        entries=(
            BnetReferenceEntry("1AAA", 1.00, 1.0),
            BnetReferenceEntry("1AAB", 1.10, 3.0),
            BnetReferenceEntry("1AAC", 1.10, 5.0),
            BnetReferenceEntry("1AAD", 1.20, 7.0),
            BnetReferenceEntry("1AAE", 2.00, 20.0),
        ),
        metadata=BnetReferenceMetadata(database_id="test_reference"),
    )


class BnetPercentileTests(unittest.TestCase):
    def test_finds_resolution_matched_reference_set_with_boundary_ties(self) -> None:
        reference_database = make_reference_database()

        matched_set = find_resolution_matched_reference_set(
            resolution_angstrom=1.05,
            reference_database=reference_database,
            nearest_resolution_count=2,
        )

        self.assertEqual(matched_set.indices, (0, 1, 2))
        self.assertEqual(matched_set.count, 3)
        self.assertEqual(matched_set.resolution_min, 1.0)
        self.assertEqual(matched_set.resolution_max, 1.1)

    def test_calculates_nearest_reference_bnet_percentile(self) -> None:
        result = calculate_bnet_percentile(
            bnet=3.6,
            resolution_angstrom=1.05,
            reference_database=make_reference_database(),
            nearest_resolution_count=2,
        )

        self.assertEqual(result.rank, 2)
        self.assertEqual(result.rank_method, BNET_PERCENTILE_RANK_METHOD)
        self.assertAlmostEqual(result.percentile_percent, 100.0 * 2 / 3)
        self.assertAlmostEqual(result.percentile_fraction, 2 / 3)
        self.assertEqual(result.nearest_reference_bnet, 3.0)
        self.assertEqual(result.reference_database_id, "test_reference")
        self.assertEqual(result.reference_entry_count, 5)
        self.assertEqual(result.nearest_resolution_count, 2)
        self.assertEqual(result.local_reference_count, 3)
        self.assertEqual(result.local_bnet_min, 1.0)
        self.assertEqual(result.local_bnet_max, 5.0)

    def test_rejects_invalid_inputs(self) -> None:
        reference_database = make_reference_database()
        invalid_calls = (
            lambda: calculate_bnet_percentile(
                bnet=True,
                resolution_angstrom=1.05,
                reference_database=reference_database,
                nearest_resolution_count=2,
            ),
            lambda: calculate_bnet_percentile(
                bnet=object(),  # type: ignore[arg-type]
                resolution_angstrom=1.05,
                reference_database=reference_database,
                nearest_resolution_count=2,
            ),
            lambda: calculate_bnet_percentile(
                bnet=math.nan,
                resolution_angstrom=1.05,
                reference_database=reference_database,
                nearest_resolution_count=2,
            ),
            lambda: calculate_bnet_percentile(
                bnet=1.0,
                resolution_angstrom=0.0,
                reference_database=reference_database,
                nearest_resolution_count=2,
            ),
            lambda: calculate_bnet_percentile(
                bnet=1.0,
                resolution_angstrom=1.05,
                reference_database=reference_database,
                nearest_resolution_count=0,
            ),
            lambda: calculate_bnet_percentile(
                bnet=1.0,
                resolution_angstrom=1.05,
                reference_database=reference_database,
                nearest_resolution_count=6,
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(BnetPercentileError):
                    invalid_call()
