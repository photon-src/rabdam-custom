"""Bnet reference database loading and validation.

The Bnet-percentile calculation is intentionally separated from the code that
builds the reference database. This module only knows how to load, validate,
and expose a table of reference Bnet values.

Reference CSV Format
--------------------
The loader ignores an optional leading index column. A reference CSV must contain
one recognizable PDB identifier column, one resolution column, and one Bnet column.

The loader also accepts some equivalent normalized column names.

Extra columns are allowed and ignored by this module. Files must contain exactly 
one recognizable column for each required value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, SupportsFloat, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray


DEFAULT_REFERENCE_SCHEMA_VERSION = "1.0"
DEFAULT_REFERENCE_METRIC_KIND = "protein_cryo_asp_glu_bnet"

_PDB_ID_COLUMN_ALIASES = frozenset(
    {
        "pdbid",
        "pdbcode",
        "pdb",
        "code",
        "entryid",
        "entry",
    }
)
_RESOLUTION_COLUMN_ALIASES = frozenset(
    {
        "resolutionangstrom",
        "resolutionangstroms",
        "resolutiona",
        "resolution",
        "resolutionang",
    }
)
_BNET_COLUMN_ALIASES = frozenset(
    {
        "bnet",
        "bnetvalue",
        "bnetscore",
    }
)

_Floatable = str | bytes | bytearray | SupportsFloat | SupportsIndex


class BnetReferenceError(ValueError):
    """Raised when a Bnet reference database cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class BnetReferenceEntry:
    """One row in a Bnet reference database."""

    pdb_id: str
    resolution_angstrom: float
    bnet: float

    def __post_init__(self) -> None:
        pdb_id = str(self.pdb_id).strip()
        if not pdb_id:
            raise BnetReferenceError("pdb_id must not be empty.")

        resolution_angstrom = _finite_float(
            self.resolution_angstrom,
            name="resolution_angstrom",
        )
        if resolution_angstrom <= 0.0:
            raise BnetReferenceError(
                "resolution_angstrom must be greater than zero, "
                f"got {resolution_angstrom!r}."
            )

        bnet = _finite_float(self.bnet, name="bnet")
        if bnet < 0.0:
            raise BnetReferenceError(
                f"bnet must be greater than or equal to zero, got {bnet!r}."
            )

        object.__setattr__(self, "pdb_id", pdb_id)
        object.__setattr__(self, "resolution_angstrom", resolution_angstrom)
        object.__setattr__(self, "bnet", bnet)


@dataclass(frozen=True, slots=True)
class BnetReferenceMetadata:
    """Provenance for a Bnet reference database.

    The percentile algorithm does not depend on most of these values, but they
    make it clear which database was used for a reported percentile.
    """

    database_id: str = "unknown_bnet_reference"
    schema_version: str = DEFAULT_REFERENCE_SCHEMA_VERSION
    metric_kind: str = DEFAULT_REFERENCE_METRIC_KIND
    source: str | None = None
    pdb_redo_snapshot: str | None = None
    eligibility_policy_id: str | None = None
    source_path: str | None = None
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        database_id = str(self.database_id).strip()
        if not database_id:
            raise BnetReferenceError("metadata.database_id must not be empty.")

        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise BnetReferenceError("metadata.schema_version must not be empty.")

        metric_kind = str(self.metric_kind).strip()
        if not metric_kind:
            raise BnetReferenceError("metadata.metric_kind must not be empty.")

        object.__setattr__(self, "database_id", database_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "metric_kind", metric_kind)


@dataclass(frozen=True, slots=True)
class BnetReferenceDatabase:
    """Validated Bnet reference database.

    The public ``entries`` tuple preserves the CSV order. Read-only NumPy arrays
    are cached for fast percentile calculations.
    """

    entries: Sequence[BnetReferenceEntry]
    metadata: BnetReferenceMetadata = field(default_factory=BnetReferenceMetadata)
    _pdb_ids: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _resolution_values: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _bnet_values: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries:
            raise BnetReferenceError("Bnet reference database must contain entries.")

        seen_pdb_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, BnetReferenceEntry):
                raise BnetReferenceError(
                    "entries must contain only BnetReferenceEntry objects."
                )

            duplicate_key = entry.pdb_id.casefold()
            if duplicate_key in seen_pdb_ids:
                raise BnetReferenceError(
                    f"Duplicate pdb_id in Bnet reference database: {entry.pdb_id!r}."
                )
            seen_pdb_ids.add(duplicate_key)

        pdb_ids = tuple(entry.pdb_id for entry in entries)
        resolution_values = np.asarray(
            [entry.resolution_angstrom for entry in entries],
            dtype=float,
        )
        bnet_values = np.asarray([entry.bnet for entry in entries], dtype=float)

        resolution_values.setflags(write=False)
        bnet_values.setflags(write=False)

        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_pdb_ids", pdb_ids)
        object.__setattr__(self, "_resolution_values", resolution_values)
        object.__setattr__(self, "_bnet_values", bnet_values)

    def __len__(self) -> int:
        """Return the number of reference entries."""

        return self.entry_count

    @property
    def entry_count(self) -> int:
        """Return the number of reference entries."""

        return len(self.entries)

    @property
    def pdb_ids(self) -> tuple[str, ...]:
        """Return PDB identifiers in reference-table order."""

        return self._pdb_ids

    @property
    def resolution_values(self) -> NDArray[np.float64]:
        """Return read-only resolution values in reference-table order."""

        return self._resolution_values

    @property
    def bnet_values(self) -> NDArray[np.float64]:
        """Return read-only Bnet values in reference-table order."""

        return self._bnet_values


def load_bnet_reference_database(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    database_id: str | None = None,
) -> BnetReferenceDatabase:
    """Load a Bnet reference database from CSV.

    Parameters
    ----------
    path
        CSV file containing at least PDB ID, resolution, and Bnet columns.
    manifest_path
        Optional JSON manifest. If omitted, ``<csv-stem>.manifest.json`` is
        loaded when present.
    database_id
        Optional fallback database identifier when no manifest is present.

    Returns
    -------
    BnetReferenceDatabase
        Validated reference database ready for percentile calculation.
    """

    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise BnetReferenceError(f"Bnet reference CSV does not exist: {csv_path}.")

    entries = _read_reference_entries_csv(csv_path)
    resolved_manifest_path = _resolve_manifest_path(csv_path, manifest_path)
    manifest = _read_manifest_json(resolved_manifest_path)

    metadata = _metadata_from_manifest(
        manifest,
        database_id=database_id,
        source_path=csv_path,
        manifest_path=resolved_manifest_path,
    )

    return BnetReferenceDatabase(entries=entries, metadata=metadata)


def write_bnet_reference_database(
    database: BnetReferenceDatabase,
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> None:
    """Write the minimal swappable Bnet reference database CSV and manifest.

    The database-builder pipeline may write richer outputs itself. This helper
    is intended for simple exports and tests.
    """

    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pdb_id", "resolution_angstrom", "bnet"),
        )
        writer.writeheader()
        for entry in database.entries:
            writer.writerow(
                {
                    "pdb_id": entry.pdb_id,
                    "resolution_angstrom": f"{entry.resolution_angstrom:.10g}",
                    "bnet": f"{entry.bnet:.12g}",
                }
            )

    if manifest_path is None:
        manifest_path = csv_path.with_suffix(".manifest.json")

    manifest_output_path = Path(manifest_path).expanduser()
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = asdict(database.metadata)
    manifest["entry_count"] = database.entry_count

    with manifest_output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_reference_entries_csv(path: Path) -> tuple[BnetReferenceEntry, ...]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BnetReferenceError(f"Bnet reference CSV has no header: {path}.")

        pdb_id_column = _find_column(
            reader.fieldnames,
            aliases=_PDB_ID_COLUMN_ALIASES,
            required_name="pdb_id",
        )
        resolution_column = _find_column(
            reader.fieldnames,
            aliases=_RESOLUTION_COLUMN_ALIASES,
            required_name="resolution_angstrom",
        )
        bnet_column = _find_column(
            reader.fieldnames,
            aliases=_BNET_COLUMN_ALIASES,
            required_name="bnet",
        )

        entries: list[BnetReferenceEntry] = []
        for row_number, row in enumerate(reader, start=2):
            if _row_is_empty(row):
                continue

            try:
                entries.append(
                    BnetReferenceEntry(
                        pdb_id=_required_text(
                            row.get(pdb_id_column),
                            column=pdb_id_column,
                            row_number=row_number,
                        ),
                        resolution_angstrom=_required_float(
                            row.get(resolution_column),
                            column=resolution_column,
                            row_number=row_number,
                        ),
                        bnet=_required_float(
                            row.get(bnet_column),
                            column=bnet_column,
                            row_number=row_number,
                        ),
                    )
                )
            except BnetReferenceError as error:
                raise BnetReferenceError(
                    f"Invalid Bnet reference row {row_number}: {error}"
                ) from error

    return tuple(entries)


def _resolve_manifest_path(
    csv_path: Path,
    manifest_path: str | Path | None,
) -> Path | None:
    if manifest_path is not None:
        resolved = Path(manifest_path).expanduser()
        if not resolved.is_file():
            raise BnetReferenceError(
                f"Bnet reference manifest does not exist: {resolved}."
            )
        return resolved

    candidate = csv_path.with_suffix(".manifest.json")
    if candidate.is_file():
        return candidate

    return None


def _read_manifest_json(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        try:
            value = json.load(handle)
        except json.JSONDecodeError as error:
            raise BnetReferenceError(
                f"Could not parse Bnet manifest JSON: {path}."
            ) from error

    if not isinstance(value, Mapping):
        raise BnetReferenceError(f"Bnet manifest must contain a JSON object: {path}.")

    return cast(Mapping[str, Any], value)


def _metadata_from_manifest(
    manifest: Mapping[str, Any],
    *,
    database_id: str | None,
    source_path: Path,
    manifest_path: Path | None,
) -> BnetReferenceMetadata:
    return BnetReferenceMetadata(
        database_id=str(manifest.get("database_id") or database_id or source_path.stem),
        schema_version=str(
            manifest.get("schema_version") or DEFAULT_REFERENCE_SCHEMA_VERSION
        ),
        metric_kind=str(manifest.get("metric_kind") or DEFAULT_REFERENCE_METRIC_KIND),
        source=_optional_string(manifest.get("source")),
        pdb_redo_snapshot=_optional_string(manifest.get("pdb_redo_snapshot")),
        eligibility_policy_id=_optional_string(manifest.get("eligibility_policy_id")),
        source_path=str(source_path),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


def _find_column(
    fieldnames: Sequence[str | None],
    *,
    aliases: frozenset[str],
    required_name: str,
) -> str:
    matching_columns: list[str] = []
    for fieldname in fieldnames:
        if fieldname is None:
            continue
        normalized = _normalize_column_name(fieldname)
        if normalized in aliases:
            matching_columns.append(fieldname)

    if len(matching_columns) == 1:
        return matching_columns[0]

    if len(matching_columns) > 1:
        matches = ", ".join(repr(name) for name in matching_columns)
        raise BnetReferenceError(
            f"Bnet reference CSV has ambiguous {required_name!r} columns: "
            f"{matches}. Use exactly one."
        )

    available = ", ".join(repr(name) for name in fieldnames if name is not None)
    raise BnetReferenceError(
        f"Bnet reference CSV is missing required {required_name!r} column. "
        f"Available columns: {available}."
    )


def _normalize_column_name(value: str) -> str:
    normalized = value.casefold().replace("å", "a")
    return "".join(character for character in normalized if character.isalnum())


def _required_text(value: object, *, column: str, row_number: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise BnetReferenceError(f"column {column!r} is empty at row {row_number}.")
    return text


def _required_float(value: object, *, column: str, row_number: int) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        raise BnetReferenceError(f"column {column!r} is empty at row {row_number}.")

    try:
        return float(text)
    except ValueError as error:
        raise BnetReferenceError(
            f"column {column!r} contains a non-numeric value at row {row_number}: "
            f"{text!r}."
        ) from error


def _row_is_empty(row: Mapping[str, Any]) -> bool:
    return all(value is None or str(value).strip() == "" for value in row.values())


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise BnetReferenceError(f"{name} must be numeric, got {value!r}.")

    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        raise BnetReferenceError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(cast(_Floatable, value))
    except (TypeError, ValueError) as error:
        raise BnetReferenceError(f"{name} must be numeric, got {value!r}.") from error

    if not math.isfinite(number):
        raise BnetReferenceError(f"{name} must be finite, got {number!r}.")

    return number


def _optional_string(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


__all__ = [
    "BnetReferenceDatabase",
    "BnetReferenceEntry",
    "BnetReferenceError",
    "BnetReferenceMetadata",
    "DEFAULT_REFERENCE_METRIC_KIND",
    "DEFAULT_REFERENCE_SCHEMA_VERSION",
    "load_bnet_reference_database",
    "write_bnet_reference_database",
]
