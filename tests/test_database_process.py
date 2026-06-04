from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

from database.checks import PdbRedoStructureCheckError, PdbRedoStructureChecks
from database.discover import PdbRedoCandidate
from database.eligibility import BnetEligibilityReason
from database.metadata import PdbRedoMetadata, PdbRedoMetadataError
from database.process import (
    PdbRedoProcessStage,
    PdbRedoRejectReason,
    _calculate_rabdam_bnet,
    process_pdb_redo_candidate,
)
from input.resolver import StructureFileFormat, StructureSourceType
from rabdam.workflow import BDamageWorkflowError


def make_candidate() -> PdbRedoCandidate:
    entry_dir = Path("/tmp/pdb-redo/1abc")
    return PdbRedoCandidate(
        pdb_id="1abc",
        entry_dir=entry_dir,
        final_cif_path=entry_dir / "1abc_final.cif",
        data_json_path=entry_dir / "data.json",
    )


def make_metadata(
    *,
    resolution_angstrom: float | None = 1.5,
    r_free: float | None = 0.25,
    temperature_k: float | None = 100.0,
    b_factor_refinement_type: str | None = None,
) -> PdbRedoMetadata:
    return PdbRedoMetadata(
        pdb_id="1abc",
        resolution_angstrom=resolution_angstrom,
        r_work=0.2,
        r_free=r_free,
        temperature_k=temperature_k,
        wilson_b=12.5,
        b_factor_restraint_weight=0.8,
        b_factor_refinement_type=b_factor_refinement_type,
        resolution_source="data_json:resolution",
        r_work_source="data_json:rwork",
        r_free_source="data_json:rfree",
        temperature_source="data_json:temperature",
        growth_temperature_k=293.0,
        growth_temperature_values_k=(293.0,),
        growth_temperature_source="data_json:growth_temperature",
        growth_temperature_sources=("data_json:growth_temperature",),
        wilson_b_source="data_json:wilson_b",
        b_factor_restraint_weight_source="data_json:b_factor_restraint_weight",
        b_factor_refinement_type_source=(
            "data_json:properties.BREFTYPE"
            if b_factor_refinement_type is not None
            else None
        ),
        data_json_path=Path("/tmp/pdb-redo/1abc/data.json"),
        final_cif_path=Path("/tmp/pdb-redo/1abc/1abc_final.cif"),
        warnings=("metadata warning",),
    )


def make_checks(
    *,
    has_protein: bool = True,
    has_nucleic_acid: bool = False,
    experimental_methods: tuple[str, ...] = ("X-RAY DIFFRACTION",),
    asp_glu_carboxyl_oxygen_count: int = 24,
    has_low_occupancy: bool = False,
    has_nonflat_protein_b_factors: bool = True,
    model_count: int = 1,
) -> PdbRedoStructureChecks:
    low_occupancy_keys = (
        ("model=1;chain=A;residue=ASP;seqid=1",)
        if has_low_occupancy
        else ()
    )
    return PdbRedoStructureChecks(
        pdb_id="1abc",
        has_protein=has_protein,
        has_nucleic_acid=has_nucleic_acid,
        experimental_methods=experimental_methods,
        asp_glu_carboxyl_oxygen_count=asp_glu_carboxyl_oxygen_count,
        asp_glu_residue_count=12,
        asp_glu_residue_keys_with_occupancy_below_one=low_occupancy_keys,
        has_nonflat_protein_b_factors=has_nonflat_protein_b_factors,
        atom_count=100,
        non_hydrogen_atom_count=90,
        protein_atom_count=80,
        model_count=model_count,
        final_cif_path=Path("/tmp/pdb-redo/1abc/1abc_final.cif"),
        warnings=("structure warning",),
    )


def make_workflow_result(*, selected_atom_count: int = 80, window_size: int = 11):
    return SimpleNamespace(
        prepared_structure=SimpleNamespace(
            report=SimpleNamespace(selected_atom_count=selected_atom_count),
        ),
        bdamage_score_result=sentinel.bdamage_score_result,
        window_size=window_size,
    )


def make_bnet_result(
    *,
    bnet: float | None = 1.2,
    site_count: int = 24,
):
    return SimpleNamespace(
        bnet=bnet,
        site_count=site_count,
        median_bdamage=1.1,
        left_area=0.4,
        right_area=0.6,
    )


class PdbRedoProcessTests(unittest.TestCase):
    def test_metadata_error_rejects_candidate(self) -> None:
        candidate = make_candidate()
        error = PdbRedoMetadataError("bad metadata")

        with patch(
            "database.process.read_pdb_redo_metadata",
            side_effect=error,
        ):
            result = process_pdb_redo_candidate(
                candidate,
                include_traceback=True,
            )

        self.assertTrue(result.is_rejected)
        assert result.rejected is not None
        self.assertEqual(result.rejected.stage, PdbRedoProcessStage.METADATA)
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.METADATA_ERROR.value,
        )
        self.assertEqual(result.rejected.exception_type, "PdbRedoMetadataError")
        self.assertIsNotNone(result.rejected.traceback_text)

    def test_structure_check_error_rejects_with_metadata_context(self) -> None:
        candidate = make_candidate()
        metadata = make_metadata()
        error = PdbRedoStructureCheckError("bad structure")

        with (
            patch("database.process.read_pdb_redo_metadata", return_value=metadata),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                side_effect=error,
            ),
        ):
            result = process_pdb_redo_candidate(candidate)

        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.STRUCTURE_CHECKS,
        )
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.STRUCTURE_CHECK_ERROR.value,
        )
        self.assertEqual(result.rejected.metadata_warnings, metadata.warnings)

    def test_domain_filter_rejects_multiple_models_before_rabdam(self) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(model_count=2),
            ),
            patch("database.process._calculate_rabdam_bnet") as calculate_mock,
        ):
            result = process_pdb_redo_candidate(candidate)

        calculate_mock.assert_not_called()
        assert result.rejected is not None
        self.assertEqual(result.rejected.stage, PdbRedoProcessStage.DOMAIN_FILTER)
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.MULTIPLE_MODELS.value,
        )

    def test_prefilter_eligibility_rejects_before_rabdam(self) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(resolution_angstrom=None),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(),
            ),
            patch("database.process._calculate_rabdam_bnet") as calculate_mock,
        ):
            result = process_pdb_redo_candidate(candidate)

        calculate_mock.assert_not_called()
        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.PREFILTER_ELIGIBILITY,
        )
        self.assertEqual(
            result.rejected.reason,
            BnetEligibilityReason.MISSING_RESOLUTION.value,
        )

    def test_remote_temperature_fetch_is_skipped_for_domain_rejection(
        self,
    ) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(temperature_k=None),
            ) as metadata_mock,
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(has_protein=False),
            ),
            patch("database.process._calculate_rabdam_bnet") as calculate_mock,
        ):
            result = process_pdb_redo_candidate(
                candidate,
                fetch_rcsb_temperature=True,
            )

        calculate_mock.assert_not_called()
        metadata_mock.assert_called_once()
        self.assertFalse(metadata_mock.call_args.kwargs["fetch_rcsb_temperature"])
        assert result.rejected is not None
        self.assertEqual(result.rejected.stage, PdbRedoProcessStage.DOMAIN_FILTER)
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.NO_PROTEIN.value,
        )

    def test_remote_temperature_fetch_runs_when_temperature_is_only_blocker(
        self,
    ) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                side_effect=(
                    make_metadata(temperature_k=None),
                    make_metadata(temperature_k=100.0),
                ),
            ) as metadata_mock,
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(),
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result()),
            ) as calculate_mock,
        ):
            result = process_pdb_redo_candidate(
                candidate,
                fetch_rcsb_temperature=True,
            )

        calculate_mock.assert_called_once()
        self.assertEqual(metadata_mock.call_count, 2)
        self.assertFalse(
            metadata_mock.call_args_list[0].kwargs["fetch_rcsb_temperature"]
        )
        self.assertTrue(
            metadata_mock.call_args_list[1].kwargs["fetch_rcsb_temperature"]
        )
        self.assertTrue(result.is_accepted)

    def test_remote_temperature_fetch_is_skipped_when_other_prefilter_fails(
        self,
    ) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(temperature_k=None),
            ) as metadata_mock,
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(asp_glu_carboxyl_oxygen_count=10),
            ),
            patch("database.process._calculate_rabdam_bnet") as calculate_mock,
        ):
            result = process_pdb_redo_candidate(
                candidate,
                fetch_rcsb_temperature=True,
            )

        calculate_mock.assert_not_called()
        metadata_mock.assert_called_once()
        self.assertFalse(metadata_mock.call_args.kwargs["fetch_rcsb_temperature"])
        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.PREFILTER_ELIGIBILITY,
        )
        self.assertEqual(
            result.rejected.reason,
            BnetEligibilityReason.MISSING_TEMPERATURE.value,
        )

    def test_reference_ineligible_candidate_can_still_record_raw_bnet(
        self,
    ) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(temperature_k=None),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(),
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result(bnet=1.7)),
            ) as calculate_mock,
        ):
            result = process_pdb_redo_candidate(
                candidate,
                attempt_bnet_for_reference_ineligible=True,
            )

        calculate_mock.assert_called_once()
        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.FINAL_ELIGIBILITY,
        )
        self.assertEqual(
            result.rejected.reason,
            BnetEligibilityReason.MISSING_TEMPERATURE.value,
        )
        self.assertEqual(result.rejected.bnet, 1.7)

    def test_breftype_over_rejects_even_when_structural_fallback_passes(
        self,
    ) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(b_factor_refinement_type="OVER"),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(has_nonflat_protein_b_factors=True),
            ),
            patch("database.process._calculate_rabdam_bnet") as calculate_mock,
        ):
            result = process_pdb_redo_candidate(candidate)

        calculate_mock.assert_not_called()
        assert result.rejected is not None
        self.assertEqual(
            result.rejected.reason,
            BnetEligibilityReason.NOT_PER_ATOM_B_FACTOR_MODEL.value,
        )
        self.assertFalse(result.rejected.uses_per_atom_b_factors)
        self.assertEqual(result.rejected.b_factor_refinement_type, "OVER")
        self.assertEqual(
            result.rejected.b_factor_model_source,
            "data_json:properties.BREFTYPE",
        )

    def test_breftype_isot_accepts_when_structural_fallback_fails(self) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(b_factor_refinement_type="ISOT"),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(has_nonflat_protein_b_factors=False),
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result()),
            ),
        ):
            result = process_pdb_redo_candidate(candidate)

        self.assertTrue(result.is_accepted)
        assert result.accepted is not None
        self.assertFalse(result.accepted.has_nonflat_protein_b_factors)
        self.assertTrue(result.accepted.uses_per_atom_b_factors)
        self.assertEqual(result.accepted.b_factor_refinement_type, "ISOT")
        self.assertEqual(
            result.accepted.b_factor_model_source,
            "data_json:properties.BREFTYPE",
        )

    def test_default_allows_nucleic_acid_when_protein_is_present(self) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(has_nucleic_acid=True),
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result()),
            ),
        ):
            result = process_pdb_redo_candidate(candidate)

        self.assertTrue(result.is_accepted)
        assert result.accepted is not None
        self.assertTrue(result.accepted.has_nucleic_acid)

    def test_rabdam_error_rejects_candidate(self) -> None:
        candidate = make_candidate()
        error = BDamageWorkflowError("workflow failed")

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(),
            ),
            patch("database.process._calculate_rabdam_bnet", side_effect=error),
        ):
            result = process_pdb_redo_candidate(candidate)

        assert result.rejected is not None
        self.assertEqual(result.rejected.stage, PdbRedoProcessStage.RABDAM)
        self.assertEqual(
            result.rejected.reason,
            PdbRedoRejectReason.RABDAM_ERROR.value,
        )

    def test_final_eligibility_rejects_after_bnet_calculation(self) -> None:
        candidate = make_candidate()

        with (
            patch(
                "database.process.read_pdb_redo_metadata",
                return_value=make_metadata(),
            ),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=make_checks(),
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result(bnet=None)),
            ),
        ):
            result = process_pdb_redo_candidate(candidate)

        assert result.rejected is not None
        self.assertEqual(
            result.rejected.stage,
            PdbRedoProcessStage.FINAL_ELIGIBILITY,
        )
        self.assertEqual(
            result.rejected.reason,
            BnetEligibilityReason.MISSING_BNET.value,
        )

    def test_accepts_candidate_and_builds_output_row(self) -> None:
        candidate = make_candidate()
        metadata = make_metadata()
        checks = make_checks()

        with (
            patch("database.process.read_pdb_redo_metadata", return_value=metadata),
            patch(
                "database.process.read_pdb_redo_structure_checks",
                return_value=checks,
            ),
            patch(
                "database.process._calculate_rabdam_bnet",
                return_value=(make_workflow_result(), make_bnet_result()),
            ),
        ):
            result = process_pdb_redo_candidate(candidate)

        self.assertTrue(result.is_accepted)
        assert result.accepted is not None
        self.assertEqual(result.accepted.pdb_id, "1abc")
        self.assertEqual(result.accepted.resolution_angstrom, 1.5)
        self.assertEqual(result.accepted.bnet, 1.2)
        self.assertEqual(result.accepted.bnet_site_count, 24)
        self.assertEqual(result.accepted.growth_temperature_k, 293.0)
        self.assertEqual(result.accepted.growth_temperature_values_k, (293.0,))
        self.assertEqual(result.accepted.selected_atom_count, 80)
        self.assertEqual(result.accepted.bdamage_window_size, 11)
        self.assertTrue(result.accepted.has_nonflat_protein_b_factors)
        self.assertTrue(result.accepted.uses_per_atom_b_factors)
        self.assertEqual(
            result.accepted.b_factor_model_source,
            "structure_backbone_b_factor_check",
        )
        self.assertEqual(result.accepted.metadata_warnings, metadata.warnings)
        self.assertEqual(result.accepted.structure_check_warnings, checks.warnings)

    def test_calculate_rabdam_bnet_wires_reader_workflow_and_bnet(self) -> None:
        candidate = make_candidate()
        workflow_result = make_workflow_result()

        with (
            patch(
                "database.process.read_structure",
                return_value=sentinel.structure_data,
            ) as read_structure_mock,
            patch(
                "database.process.calculate_bdamage_for_structure_data",
                return_value=workflow_result,
            ) as workflow_mock,
            patch(
                "database.process.calculate_protein_bnet",
                return_value=sentinel.bnet_result,
            ) as bnet_mock,
        ):
            result = _calculate_rabdam_bnet(
                candidate,
                workflow_options=sentinel.workflow_options,
                preparation_options=sentinel.preparation_options,
            )

        self.assertEqual(result, (workflow_result, sentinel.bnet_result))
        resolved_input = read_structure_mock.call_args.args[0]
        self.assertEqual(resolved_input.original_input, str(candidate.final_cif_path))
        self.assertEqual(resolved_input.source_type, StructureSourceType.LOCAL_FILE)
        self.assertEqual(resolved_input.file_format, StructureFileFormat.MMCIF)
        self.assertEqual(resolved_input.local_path, candidate.final_cif_path)
        self.assertEqual(resolved_input.structure_id, candidate.pdb_id)
        workflow_mock.assert_called_once_with(
            sentinel.structure_data,
            workflow_options=sentinel.workflow_options,
            preparation_options=sentinel.preparation_options,
        )
        bnet_mock.assert_called_once_with(
            prepared_structure=workflow_result.prepared_structure,
            bdamage_score_result=workflow_result.bdamage_score_result,
        )


if __name__ == "__main__":
    unittest.main()
