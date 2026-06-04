import unittest

from database.eligibility import (
    BnetEligibilityContext,
    BnetEligibilityReason,
    check_bnet_reference_eligibility,
)


def make_eligible_context() -> BnetEligibilityContext:
    return BnetEligibilityContext(
        resolution_angstrom=1.5,
        r_free=0.2,
        temperature_k=100.0,
        asp_glu_carboxyl_oxygen_count=40,
        has_asp_glu_residue_with_total_occupancy_below_one=False,
        uses_per_atom_b_factors=True,
        bnet=1.5,
    )


class BnetEligibilityTests(unittest.TestCase):
    def test_accepts_eligible_context(self) -> None:
        result = check_bnet_reference_eligibility(make_eligible_context())

        self.assertTrue(result.is_eligible)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.primary_reason, BnetEligibilityReason.ELIGIBLE)

    def test_reports_all_relevant_failure_reasons(self) -> None:
        context = BnetEligibilityContext(
            resolution_angstrom=4.0,
            r_free=0.45,
            temperature_k=200.0,
            asp_glu_carboxyl_oxygen_count=10,
            has_asp_glu_residue_with_total_occupancy_below_one=True,
            uses_per_atom_b_factors=False,
            bnet=None,
        )

        result = check_bnet_reference_eligibility(context)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            tuple(issue.reason for issue in result.issues),
            (
                BnetEligibilityReason.RESOLUTION_WORSE_THAN_LIMIT,
                BnetEligibilityReason.RFREE_TOO_HIGH,
                BnetEligibilityReason.TEMPERATURE_OUTSIDE_CRYO_RANGE,
                BnetEligibilityReason.TOO_FEW_ASP_GLU_CARBOXYL_OXYGENS,
                BnetEligibilityReason.ASP_GLU_OCCUPANCY_LESS_THAN_ONE,
                BnetEligibilityReason.NOT_PER_ATOM_B_FACTOR_MODEL,
                BnetEligibilityReason.MISSING_BNET,
            ),
        )

    def test_rejects_non_finite_and_non_numeric_values(self) -> None:
        context = BnetEligibilityContext(
            resolution_angstrom=object(),  # type: ignore[arg-type]
            r_free=True,  # type: ignore[arg-type]
            temperature_k=float("nan"),
            asp_glu_carboxyl_oxygen_count=40,
            has_asp_glu_residue_with_total_occupancy_below_one=False,
            uses_per_atom_b_factors=True,
            bnet=object(),  # type: ignore[arg-type]
        )

        result = check_bnet_reference_eligibility(context)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            tuple(issue.reason for issue in result.issues),
            (
                BnetEligibilityReason.INVALID_RESOLUTION,
                BnetEligibilityReason.INVALID_RFREE,
                BnetEligibilityReason.INVALID_TEMPERATURE,
                BnetEligibilityReason.INVALID_BNET,
            ),
        )

    def test_rejects_invalid_count_values(self) -> None:
        invalid_counts = (-1, 20.5, True)

        for invalid_count in invalid_counts:
            with self.subTest(invalid_count=invalid_count):
                context = BnetEligibilityContext(
                    resolution_angstrom=1.5,
                    r_free=0.2,
                    temperature_k=100.0,
                    asp_glu_carboxyl_oxygen_count=invalid_count,  # type: ignore[arg-type]
                    has_asp_glu_residue_with_total_occupancy_below_one=False,
                    uses_per_atom_b_factors=True,
                    bnet=1.5,
                )

                result = check_bnet_reference_eligibility(context)

                self.assertFalse(result.is_eligible)
                self.assertEqual(
                    tuple(issue.reason for issue in result.issues),
                    (BnetEligibilityReason.INVALID_ASP_GLU_CARBOXYL_OXYGEN_COUNT,),
                )

    def test_rejects_invalid_boolean_flags(self) -> None:
        context = BnetEligibilityContext(
            resolution_angstrom=1.5,
            r_free=0.2,
            temperature_k=100.0,
            asp_glu_carboxyl_oxygen_count=40,
            has_asp_glu_residue_with_total_occupancy_below_one="false",  # type: ignore[arg-type]
            uses_per_atom_b_factors=1,  # type: ignore[arg-type]
            bnet=1.5,
        )

        result = check_bnet_reference_eligibility(context)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            tuple(issue.reason for issue in result.issues),
            (
                BnetEligibilityReason.INVALID_OCCUPANCY_FLAG,
                BnetEligibilityReason.INVALID_B_FACTOR_MODEL_FLAG,
            ),
        )

    def test_bnet_can_be_optional(self) -> None:
        context = make_eligible_context()
        context_without_bnet = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=context.r_free,
            temperature_k=context.temperature_k,
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=None,
        )

        result = check_bnet_reference_eligibility(
            context_without_bnet,
            require_bnet=False,
        )

        self.assertTrue(result.is_eligible)

    def test_rejects_missing_temperature_as_cannot_verify_temperature(self) -> None:
        context = make_eligible_context()
        context_without_temperature = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=context.r_free,
            temperature_k=None,
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=context.bnet,
        )

        result = check_bnet_reference_eligibility(context_without_temperature)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            result.primary_reason,
            BnetEligibilityReason.MISSING_TEMPERATURE,
        )
        self.assertEqual(
            result.primary_reason.value,
            "cannot_verify_temperature",
        )

    def test_accepts_multiple_verified_cryo_temperatures(self) -> None:
        context = make_eligible_context()
        context_with_multiple_temperatures = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=context.r_free,
            temperature_k=(100.0, 110.0),
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=context.bnet,
        )

        result = check_bnet_reference_eligibility(
            context_with_multiple_temperatures
        )

        self.assertTrue(result.is_eligible)

    def test_rejects_if_any_verified_temperature_is_out_of_range(self) -> None:
        context = make_eligible_context()
        context_with_hot_temperature = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=context.r_free,
            temperature_k=(100.0, 130.0),
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=context.bnet,
        )

        result = check_bnet_reference_eligibility(context_with_hot_temperature)

        self.assertFalse(result.is_eligible)
        self.assertEqual(
            result.primary_reason,
            BnetEligibilityReason.TEMPERATURE_OUTSIDE_CRYO_RANGE,
        )

    def test_rejects_rfree_equal_to_strict_limit(self) -> None:
        context = make_eligible_context()
        context_at_rfree_limit = BnetEligibilityContext(
            resolution_angstrom=context.resolution_angstrom,
            r_free=0.4,
            temperature_k=context.temperature_k,
            asp_glu_carboxyl_oxygen_count=context.asp_glu_carboxyl_oxygen_count,
            has_asp_glu_residue_with_total_occupancy_below_one=(
                context.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            uses_per_atom_b_factors=context.uses_per_atom_b_factors,
            bnet=context.bnet,
        )

        result = check_bnet_reference_eligibility(context_at_rfree_limit)

        self.assertFalse(result.is_eligible)
        self.assertEqual(result.primary_reason, BnetEligibilityReason.RFREE_TOO_HIGH)
