"""Write simple Bnet reference-database CSV output.

Rejected entries are optional and written to a separate simple CSV for build
diagnostics. The accepted CSV is the file consumed by ``bnet.reference`` and
``bnet.percentile``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from bnet.reference import load_bnet_reference_database

from .metadata import TemperatureCacheEntry
from .process import (
    AcceptedBnetReferenceRow,
    PdbRedoProcessResult,
    PdbRedoProcessStage,
    RejectedBnetReferenceRow,
)


ACCEPTED_FIELDNAMES = (
    "",
    "PDB code",
    "Resolution (A)",
    "Bnet",
)

ACCEPTED_DETAILS_FIELDNAMES = (
    "PDB code",
    "Resolution (A)",
    "Bnet",
    "Rwork",
    "Rfree",
    "temperature_K",
    "wilson_b",
    "b_factor_restraint_weight",
    "bnet_site_count",
    "Asp/Glu carboxyl oxygen count",
    "asp_glu_residue_count",
    "median_bdamage",
    "left_area",
    "right_area",
    "atom_count",
    "non_hydrogen_atom_count",
    "protein_atom_count",
    "selected_atom_count",
    "bdamage_window_size",
    "has_protein",
    "has_nucleic_acid",
    "is_xray",
    "has_nonflat_protein_b_factors",
    "uses_per_atom_b_factors",
    "b_factor_model_source",
    "b_factor_refinement_type",
    "b_factor_refinement_type_source",
    "has_asp_glu_residue_with_total_occupancy_below_one",
    "experimental_methods",
    "metadata_warnings",
    "structure_check_warnings",
    "resolution_source",
    "r_work_source",
    "r_free_source",
    "temperature_source",
    "temperature_values_K",
    "temperature_sources",
    "growth_temperature_K",
    "growth_temperature_values_K",
    "growth_temperature_source",
    "growth_temperature_sources",
    "temperature_cache_status",
    "temperature_cache_message",
    "wilson_b_source",
    "b_factor_restraint_weight_source",
    "final_cif_path",
    "data_json_path",
)

REJECTED_FIELDNAMES = (
    "PDB code",
    "stage",
    "reason",
    "message",
    "Resolution (A)",
    "Rfree",
    "temperature_K",
    "Asp/Glu carboxyl oxygen count",
    "Bnet",
    "uses_per_atom_b_factors",
    "b_factor_model_source",
    "b_factor_refinement_type",
    "b_factor_refinement_type_source",
    "final_cif_path",
    "data_json_path",
    "exception_type",
    "traceback_text",
    "metadata_warnings",
    "structure_check_warnings",
)

ALL_SCORES_FIELDNAMES = (
    "PDB code",
    "status",
    "stage",
    "reason",
    "message",
    "is_reference_eligible",
    "Resolution (A)",
    "Rwork",
    "Rfree",
    "temperature_K",
    "temperature_values_K",
    "growth_temperature_K",
    "growth_temperature_values_K",
    "Bnet",
    "bnet_site_count",
    "Asp/Glu carboxyl oxygen count",
    "asp_glu_residue_count",
    "median_bdamage",
    "left_area",
    "right_area",
    "atom_count",
    "non_hydrogen_atom_count",
    "protein_atom_count",
    "selected_atom_count",
    "bdamage_window_size",
    "has_protein",
    "has_nucleic_acid",
    "is_xray",
    "has_nonflat_protein_b_factors",
    "uses_per_atom_b_factors",
    "b_factor_model_source",
    "b_factor_refinement_type",
    "b_factor_refinement_type_source",
    "has_asp_glu_residue_with_total_occupancy_below_one",
    "experimental_methods",
    "metadata_warnings",
    "structure_check_warnings",
    "resolution_source",
    "r_work_source",
    "r_free_source",
    "temperature_source",
    "temperature_sources",
    "growth_temperature_source",
    "growth_temperature_sources",
    "temperature_cache_status",
    "temperature_cache_message",
    "wilson_b_source",
    "b_factor_restraint_weight_source",
    "final_cif_path",
    "data_json_path",
    "exception_type",
    "traceback_text",
)

TEMPERATURE_CACHE_FIELDNAMES = (
    "PDB code",
    "status",
    "collection_temperature_values_K",
    "source",
    "message",
)


@dataclass(frozen=True, slots=True)
class BnetDatabaseOutputPaths:
    """Output paths for a simple Bnet database build."""

    accepted_csv_path: Path
    accepted_details_csv_path: Path | None = None
    rejected_csv_path: Path | None = None
    all_scores_csv_path: Path | None = None


class BnetDatabaseOutputError(ValueError):
    """Raised when Bnet database output cannot be written."""


class BnetDatabaseCsvWriter:
    """Append accepted and rejected process results to simple CSV files."""

    def __init__(
        self,
        accepted_csv_path: str | Path,
        *,
        accepted_details_csv_path: str | Path | None = None,
        rejected_csv_path: str | Path | None = None,
        all_scores_csv_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> None:
        self.accepted_csv_path = Path(accepted_csv_path).expanduser()
        self.accepted_details_csv_path = (
            Path(accepted_details_csv_path).expanduser()
            if accepted_details_csv_path is not None
            else None
        )
        self.rejected_csv_path = (
            Path(rejected_csv_path).expanduser()
            if rejected_csv_path is not None
            else None
        )
        self.all_scores_csv_path = (
            Path(all_scores_csv_path).expanduser()
            if all_scores_csv_path is not None
            else None
        )

        self.accepted_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.accepted_details_csv_path is not None:
            self.accepted_details_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.rejected_csv_path is not None:
            self.rejected_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.all_scores_csv_path is not None:
            self.all_scores_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self.accepted_csv_path.unlink(missing_ok=True)
            if self.accepted_details_csv_path is not None:
                self.accepted_details_csv_path.unlink(missing_ok=True)
            if self.rejected_csv_path is not None:
                self.rejected_csv_path.unlink(missing_ok=True)
            if self.all_scores_csv_path is not None:
                self.all_scores_csv_path.unlink(missing_ok=True)

        _ensure_header(self.accepted_csv_path, ACCEPTED_FIELDNAMES)
        if self.accepted_details_csv_path is not None:
            _ensure_header(
                self.accepted_details_csv_path,
                ACCEPTED_DETAILS_FIELDNAMES,
            )
        if self.rejected_csv_path is not None:
            _ensure_header(self.rejected_csv_path, REJECTED_FIELDNAMES)
        if self.all_scores_csv_path is not None:
            _ensure_header(self.all_scores_csv_path, ALL_SCORES_FIELDNAMES)

        self._accepted_count = _existing_data_row_count(self.accepted_csv_path)
        accepted_details_count = (
            _existing_data_row_count(self.accepted_details_csv_path)
            if self.accepted_details_csv_path is not None
            else self._accepted_count
        )
        if accepted_details_count != self._accepted_count:
            raise BnetDatabaseOutputError(
                "Accepted CSV and accepted-details CSV contain different "
                "numbers of data rows."
            )

        self._rejected_count = (
            _existing_data_row_count(self.rejected_csv_path)
            if self.rejected_csv_path is not None
            else 0
        )
        self._all_scores_count = (
            _existing_data_row_count(self.all_scores_csv_path)
            if self.all_scores_csv_path is not None
            else 0
        )

    @property
    def accepted_count(self) -> int:
        """Return number of accepted rows written or already present."""

        return self._accepted_count

    @property
    def rejected_count(self) -> int:
        """Return number of rejected rows written or already present."""

        return self._rejected_count

    @property
    def all_scores_count(self) -> int:
        """Return number of all-scores rows written or already present."""

        return self._all_scores_count

    @property
    def paths(self) -> BnetDatabaseOutputPaths:
        """Return output paths."""

        return BnetDatabaseOutputPaths(
            accepted_csv_path=self.accepted_csv_path,
            accepted_details_csv_path=self.accepted_details_csv_path,
            rejected_csv_path=self.rejected_csv_path,
            all_scores_csv_path=self.all_scores_csv_path,
        )

    def write_result(self, result: PdbRedoProcessResult) -> None:
        """Write one accepted or rejected process result."""

        if result.accepted is not None:
            self.write_accepted(result.accepted)
            self.write_all_scores_result(result)
            return

        if result.rejected is not None:
            self.write_rejected(result.rejected)
            self.write_all_scores_result(result)
            return

        raise BnetDatabaseOutputError(
            "PdbRedoProcessResult contains neither accepted nor rejected row."
        )

    def write_accepted(self, row: AcceptedBnetReferenceRow) -> None:
        """Append one accepted Bnet database row in processing order."""

        output_row: dict[str, object] = {
            "": self._accepted_count,
            "PDB code": row.pdb_id.upper(),
            "Resolution (A)": _format_float(row.resolution_angstrom),
            "Bnet": _format_float(row.bnet),
        }

        _append_dict_row(self.accepted_csv_path, ACCEPTED_FIELDNAMES, output_row)
        if self.accepted_details_csv_path is not None:
            _append_dict_row(
                self.accepted_details_csv_path,
                ACCEPTED_DETAILS_FIELDNAMES,
                _accepted_details_dict(row),
            )
        self._accepted_count += 1

    def write_rejected(self, row: RejectedBnetReferenceRow) -> None:
        """Append one rejected build-log row, if rejected output is enabled."""

        if self.rejected_csv_path is None:
            return

        output_row = {
            "PDB code": row.pdb_id.upper(),
            "stage": row.stage.value,
            "reason": row.reason,
            "message": row.message,
            "Resolution (A)": _format_optional_float(row.resolution_angstrom),
            "Rfree": _format_optional_float(row.r_free),
            "temperature_K": _format_optional_float(row.temperature_k),
            "Asp/Glu carboxyl oxygen count": (
                "" if row.asp_glu_carboxyl_oxygen_count is None
                else str(row.asp_glu_carboxyl_oxygen_count)
            ),
            "Bnet": _format_optional_float(row.bnet),
            "uses_per_atom_b_factors": _format_optional_bool(
                row.uses_per_atom_b_factors
            ),
            "b_factor_model_source": row.b_factor_model_source or "",
            "b_factor_refinement_type": row.b_factor_refinement_type or "",
            "b_factor_refinement_type_source": (
                row.b_factor_refinement_type_source or ""
            ),
            "final_cif_path": str(row.final_cif_path),
            "data_json_path": (
                "" if row.data_json_path is None else str(row.data_json_path)
            ),
            "exception_type": row.exception_type or "",
            "traceback_text": row.traceback_text or "",
            "metadata_warnings": _format_text_tuple(row.metadata_warnings),
            "structure_check_warnings": _format_text_tuple(
                row.structure_check_warnings
            ),
        }

        _append_dict_row(self.rejected_csv_path, REJECTED_FIELDNAMES, output_row)
        self._rejected_count += 1

    def write_all_scores_result(self, result: PdbRedoProcessResult) -> None:
        """Append one broad diagnostics row, if all-scores output is enabled."""

        if self.all_scores_csv_path is None:
            return

        if result.accepted is not None:
            row = _all_scores_accepted_dict(result.accepted)
        elif result.rejected is not None:
            row = _all_scores_rejected_dict(result.rejected)
        else:
            raise BnetDatabaseOutputError(
                "PdbRedoProcessResult contains neither accepted nor rejected row."
            )

        _append_dict_row(self.all_scores_csv_path, ALL_SCORES_FIELDNAMES, row)
        self._all_scores_count += 1


def write_simple_bnet_database_csv(
    rows: Sequence[AcceptedBnetReferenceRow],
    path: str | Path,
    *,
    sort_by_bnet_descending: bool = True,
) -> None:
    """Write accepted rows to a simple Bnet reference CSV.

    This is useful for writing a completed database all at once. By default,
    rows are sorted by descending Bnet. For long builds, prefer
    ``BnetDatabaseCsvWriter`` so rows can be appended in processing order as
    each candidate finishes.
    """

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows = list(rows)
    if sort_by_bnet_descending:
        output_rows.sort(key=lambda row: row.bnet, reverse=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCEPTED_FIELDNAMES)
        writer.writeheader()

        for index, row in enumerate(output_rows):
            writer.writerow(
                {
                    "": index,
                    "PDB code": row.pdb_id.upper(),
                    "Resolution (A)": _format_float(row.resolution_angstrom),
                    "Bnet": _format_float(row.bnet),
                }
            )


def write_sorted_reference_csv_from_accepted_csv(
    source_path: str | Path,
    output_path: str | Path,
) -> int:
    """Write a final Bnet reference CSV sorted by descending Bnet.

    Returns the number of reference entries written.
    """

    source_csv_path = Path(source_path).expanduser()
    output_csv_path = Path(output_path).expanduser()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if source_csv_path.is_file() and source_csv_path.stat().st_size > 0:
        with source_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                rows.append(dict(row))

    rows.sort(key=lambda row: float(row["Bnet"]), reverse=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACCEPTED_FIELDNAMES)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "": index,
                    "PDB code": row["PDB code"].strip().upper(),
                    "Resolution (A)": _format_float(float(row["Resolution (A)"])),
                    "Bnet": _format_float(float(row["Bnet"])),
                }
            )

    if rows:
        load_bnet_reference_database(output_csv_path, database_id=output_csv_path.stem)

    return len(rows)


def load_temperature_cache(
    path: str | Path | None,
) -> dict[str, TemperatureCacheEntry]:
    """Load compact RCSB temperature-cache entries keyed by lower-case PDB ID."""

    if path is None:
        return {}

    cache_path = Path(path).expanduser()
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return {}

    with cache_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["PDB code"].strip().casefold(): TemperatureCacheEntry(
                pdb_id=row["PDB code"],
                temperature_values_k=_parse_float_tuple(
                    row.get("collection_temperature_values_K", "")
                ),
                source=row.get("source") or None,
                status=row.get("status") or "missing",
                message=row.get("message") or None,
            )
            for row in reader
            if row.get("PDB code", "").strip()
        }


class TemperatureCacheCsvWriter:
    """Append compact temperature lookup results from the parent process."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        existing_cache: Mapping[str, TemperatureCacheEntry] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._seen_pdb_ids = set(existing_cache or {})

        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.path.unlink(missing_ok=True)
            self._seen_pdb_ids.clear()

        _ensure_header(self.path, TEMPERATURE_CACHE_FIELDNAMES)

    def write_result(self, result: PdbRedoProcessResult) -> None:
        entry = temperature_cache_entry_from_result(result)
        if entry is not None:
            self.write_entry(entry)

    def write_entry(self, entry: TemperatureCacheEntry) -> None:
        if self.path is None:
            return

        pdb_id = entry.pdb_id.casefold()
        if pdb_id in self._seen_pdb_ids:
            return

        _append_dict_row(
            self.path,
            TEMPERATURE_CACHE_FIELDNAMES,
            {
                "PDB code": entry.pdb_id.upper(),
                "status": entry.status,
                "collection_temperature_values_K": _format_float_tuple(
                    entry.temperature_values_k
                ),
                "source": entry.source or "",
                "message": entry.message or "",
            },
        )
        self._seen_pdb_ids.add(pdb_id)


def temperature_cache_entry_from_result(
    result: PdbRedoProcessResult,
) -> TemperatureCacheEntry | None:
    row = result.accepted if result.accepted is not None else result.rejected
    if row is None or row.temperature_cache_status is None:
        return None

    return TemperatureCacheEntry(
        pdb_id=row.pdb_id,
        temperature_values_k=row.temperature_values_k,
        source=row.temperature_source,
        status=row.temperature_cache_status,
        message=row.temperature_cache_message,
    )


def _ensure_header(path: Path, fieldnames: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration:
                header = ()

        if header != fieldnames:
            raise BnetDatabaseOutputError(
                f"Existing CSV header does not match expected schema: {path}"
            )
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _accepted_details_dict(row: AcceptedBnetReferenceRow) -> dict[str, object]:
    return {
        "PDB code": row.pdb_id.upper(),
        "Resolution (A)": _format_float(row.resolution_angstrom),
        "Bnet": _format_float(row.bnet),
        "Rwork": _format_optional_float(row.r_work),
        "Rfree": _format_optional_float(row.r_free),
        "temperature_K": _format_optional_float(row.temperature_k),
        "wilson_b": _format_optional_float(row.wilson_b),
        "b_factor_restraint_weight": _format_optional_float(
            row.b_factor_restraint_weight
        ),
        "bnet_site_count": row.bnet_site_count,
        "Asp/Glu carboxyl oxygen count": row.asp_glu_carboxyl_oxygen_count,
        "asp_glu_residue_count": row.asp_glu_residue_count,
        "median_bdamage": _format_float(row.median_bdamage),
        "left_area": _format_float(row.left_area),
        "right_area": _format_float(row.right_area),
        "atom_count": row.atom_count,
        "non_hydrogen_atom_count": row.non_hydrogen_atom_count,
        "protein_atom_count": row.protein_atom_count,
        "selected_atom_count": row.selected_atom_count,
        "bdamage_window_size": row.bdamage_window_size,
        "has_protein": _format_bool(row.has_protein),
        "has_nucleic_acid": _format_bool(row.has_nucleic_acid),
        "is_xray": _format_bool(row.is_xray),
        "has_nonflat_protein_b_factors": _format_bool(
            row.has_nonflat_protein_b_factors
        ),
        "uses_per_atom_b_factors": _format_bool(row.uses_per_atom_b_factors),
        "b_factor_model_source": row.b_factor_model_source or "",
        "b_factor_refinement_type": row.b_factor_refinement_type or "",
        "b_factor_refinement_type_source": (
            row.b_factor_refinement_type_source or ""
        ),
        "has_asp_glu_residue_with_total_occupancy_below_one": _format_bool(
            row.has_asp_glu_residue_with_total_occupancy_below_one
        ),
        "experimental_methods": _format_text_tuple(row.experimental_methods),
        "metadata_warnings": _format_text_tuple(row.metadata_warnings),
        "structure_check_warnings": _format_text_tuple(
            row.structure_check_warnings
        ),
        "resolution_source": row.resolution_source or "",
        "r_work_source": row.r_work_source or "",
        "r_free_source": row.r_free_source or "",
        "temperature_source": row.temperature_source or "",
        "temperature_values_K": _format_float_tuple(row.temperature_values_k),
        "temperature_sources": _format_text_tuple(row.temperature_sources),
        "growth_temperature_K": _format_optional_float(row.growth_temperature_k),
        "growth_temperature_values_K": _format_float_tuple(
            row.growth_temperature_values_k
        ),
        "growth_temperature_source": row.growth_temperature_source or "",
        "growth_temperature_sources": _format_text_tuple(
            row.growth_temperature_sources
        ),
        "temperature_cache_status": row.temperature_cache_status or "",
        "temperature_cache_message": row.temperature_cache_message or "",
        "wilson_b_source": row.wilson_b_source or "",
        "b_factor_restraint_weight_source": (
            row.b_factor_restraint_weight_source or ""
        ),
        "final_cif_path": "" if row.final_cif_path is None else str(row.final_cif_path),
        "data_json_path": "" if row.data_json_path is None else str(row.data_json_path),
    }


def _all_scores_accepted_dict(row: AcceptedBnetReferenceRow) -> dict[str, object]:
    output_row = _blank_all_scores_row()
    output_row.update(
        {
            "PDB code": row.pdb_id.upper(),
            "status": "accepted",
            "stage": PdbRedoProcessStage.ACCEPTED.value,
            "is_reference_eligible": "true",
            "Resolution (A)": _format_float(row.resolution_angstrom),
            "Rwork": _format_optional_float(row.r_work),
            "Rfree": _format_optional_float(row.r_free),
            "temperature_K": _format_optional_float(row.temperature_k),
            "temperature_values_K": _format_float_tuple(row.temperature_values_k),
            "growth_temperature_K": _format_optional_float(
                row.growth_temperature_k
            ),
            "growth_temperature_values_K": _format_float_tuple(
                row.growth_temperature_values_k
            ),
            "Bnet": _format_float(row.bnet),
            "bnet_site_count": row.bnet_site_count,
            "Asp/Glu carboxyl oxygen count": row.asp_glu_carboxyl_oxygen_count,
            "asp_glu_residue_count": row.asp_glu_residue_count,
            "median_bdamage": _format_float(row.median_bdamage),
            "left_area": _format_float(row.left_area),
            "right_area": _format_float(row.right_area),
            "atom_count": row.atom_count,
            "non_hydrogen_atom_count": row.non_hydrogen_atom_count,
            "protein_atom_count": row.protein_atom_count,
            "selected_atom_count": row.selected_atom_count,
            "bdamage_window_size": row.bdamage_window_size,
            "has_protein": _format_bool(row.has_protein),
            "has_nucleic_acid": _format_bool(row.has_nucleic_acid),
            "is_xray": _format_bool(row.is_xray),
            "has_nonflat_protein_b_factors": _format_bool(
                row.has_nonflat_protein_b_factors
            ),
            "uses_per_atom_b_factors": _format_bool(row.uses_per_atom_b_factors),
            "b_factor_model_source": row.b_factor_model_source or "",
            "b_factor_refinement_type": row.b_factor_refinement_type or "",
            "b_factor_refinement_type_source": (
                row.b_factor_refinement_type_source or ""
            ),
            "has_asp_glu_residue_with_total_occupancy_below_one": _format_bool(
                row.has_asp_glu_residue_with_total_occupancy_below_one
            ),
            "experimental_methods": _format_text_tuple(row.experimental_methods),
            "metadata_warnings": _format_text_tuple(row.metadata_warnings),
            "structure_check_warnings": _format_text_tuple(
                row.structure_check_warnings
            ),
            "resolution_source": row.resolution_source or "",
            "r_work_source": row.r_work_source or "",
            "r_free_source": row.r_free_source or "",
            "temperature_source": row.temperature_source or "",
            "temperature_sources": _format_text_tuple(row.temperature_sources),
            "growth_temperature_source": row.growth_temperature_source or "",
            "growth_temperature_sources": _format_text_tuple(
                row.growth_temperature_sources
            ),
            "temperature_cache_status": row.temperature_cache_status or "",
            "temperature_cache_message": row.temperature_cache_message or "",
            "wilson_b_source": row.wilson_b_source or "",
            "b_factor_restraint_weight_source": (
                row.b_factor_restraint_weight_source or ""
            ),
            "final_cif_path": (
                "" if row.final_cif_path is None else str(row.final_cif_path)
            ),
            "data_json_path": (
                "" if row.data_json_path is None else str(row.data_json_path)
            ),
        }
    )
    return output_row


def _all_scores_rejected_dict(row: RejectedBnetReferenceRow) -> dict[str, object]:
    output_row = _blank_all_scores_row()
    output_row.update(
        {
            "PDB code": row.pdb_id.upper(),
            "status": "rejected",
            "stage": row.stage.value,
            "reason": row.reason,
            "message": row.message,
            "is_reference_eligible": "false",
            "Resolution (A)": _format_optional_float(row.resolution_angstrom),
            "Rwork": _format_optional_float(row.r_work),
            "Rfree": _format_optional_float(row.r_free),
            "temperature_K": _format_optional_float(row.temperature_k),
            "temperature_values_K": _format_float_tuple(row.temperature_values_k),
            "growth_temperature_K": _format_optional_float(
                row.growth_temperature_k
            ),
            "growth_temperature_values_K": _format_float_tuple(
                row.growth_temperature_values_k
            ),
            "Bnet": _format_optional_float(row.bnet),
            "uses_per_atom_b_factors": _format_optional_bool(
                row.uses_per_atom_b_factors
            ),
            "b_factor_model_source": row.b_factor_model_source or "",
            "b_factor_refinement_type": row.b_factor_refinement_type or "",
            "b_factor_refinement_type_source": (
                row.b_factor_refinement_type_source or ""
            ),
            "Asp/Glu carboxyl oxygen count": (
                "" if row.asp_glu_carboxyl_oxygen_count is None
                else str(row.asp_glu_carboxyl_oxygen_count)
            ),
            "metadata_warnings": _format_text_tuple(row.metadata_warnings),
            "structure_check_warnings": _format_text_tuple(
                row.structure_check_warnings
            ),
            "resolution_source": row.resolution_source or "",
            "r_work_source": row.r_work_source or "",
            "r_free_source": row.r_free_source or "",
            "temperature_source": row.temperature_source or "",
            "temperature_sources": _format_text_tuple(row.temperature_sources),
            "growth_temperature_source": row.growth_temperature_source or "",
            "growth_temperature_sources": _format_text_tuple(
                row.growth_temperature_sources
            ),
            "temperature_cache_status": row.temperature_cache_status or "",
            "temperature_cache_message": row.temperature_cache_message or "",
            "wilson_b_source": row.wilson_b_source or "",
            "b_factor_restraint_weight_source": (
                row.b_factor_restraint_weight_source or ""
            ),
            "final_cif_path": str(row.final_cif_path),
            "data_json_path": (
                "" if row.data_json_path is None else str(row.data_json_path)
            ),
            "exception_type": row.exception_type or "",
            "traceback_text": row.traceback_text or "",
        }
    )
    return output_row


def _blank_all_scores_row() -> dict[str, object]:
    return {fieldname: "" for fieldname in ALL_SCORES_FIELDNAMES}


def _append_dict_row(
    path: Path,
    fieldnames: tuple[str, ...],
    row: Mapping[str, object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _existing_data_row_count(path: Path | None) -> int:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return 0

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0

        return sum(1 for _ in reader)


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""

    return _format_float(value)


def _format_float_tuple(values: tuple[float, ...]) -> str:
    if not values:
        return ""

    return json.dumps([float(_format_float(value)) for value in values])


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    text = value.strip()
    if not text:
        return ()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(";") if part.strip()]

    if not isinstance(parsed, list):
        return ()

    values: list[float] = []
    for item in parsed:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue

    return tuple(values)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""

    return _format_bool(value)


def _format_text_tuple(values: tuple[str, ...]) -> str:
    if not values:
        return ""

    return json.dumps(list(values), separators=(",", ":"))


__all__ = [
    "ACCEPTED_FIELDNAMES",
    "ACCEPTED_DETAILS_FIELDNAMES",
    "ALL_SCORES_FIELDNAMES",
    "REJECTED_FIELDNAMES",
    "TEMPERATURE_CACHE_FIELDNAMES",
    "BnetDatabaseCsvWriter",
    "BnetDatabaseOutputError",
    "BnetDatabaseOutputPaths",
    "TemperatureCacheCsvWriter",
    "load_temperature_cache",
    "temperature_cache_entry_from_result",
    "write_sorted_reference_csv_from_accepted_csv",
    "write_simple_bnet_database_csv",
]
