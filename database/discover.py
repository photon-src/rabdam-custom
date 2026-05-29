"""Discover local PDB-REDO entries for Bnet reference-database construction.

This module performs filesystem discovery only.

Expected PDB-REDO entry files
-----------------------------
For a PDB ID ``1abc``, the discovery code looks for:

    1abc_final.cif
    data.json

The PDB-REDO mirror may be either flat, for example::

    pdb-redo/
      1abc/
        1abc_final.cif
        data.json

or nested. Nested layouts are supported by recursively searching for
``*_final.cif`` files.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


_FINAL_CIF_SUFFIX = "_final.cif"
_DATA_JSON_NAME = "data.json"


class PdbRedoDiscoveryError(ValueError):
    """Raised when local PDB-REDO discovery cannot be performed."""


class PdbRedoDiscoverySkipReason(str, Enum):
    """Machine-readable reasons a possible PDB-REDO entry was skipped."""

    MISSING_DATA_JSON = "missing_data_json"
    INVALID_PDB_ID = "invalid_pdb_id"
    DUPLICATE_PDB_ID = "duplicate_pdb_id"


@dataclass(frozen=True, slots=True)
class PdbRedoCandidate:
    """Candidate PDB-REDO entry discovered in a local mirror."""

    pdb_id: str
    entry_dir: Path
    final_cif_path: Path
    data_json_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PdbRedoDiscoverySkip:
    """A discovered path that could not be used as a candidate."""

    pdb_id: str | None
    path: Path
    reason: PdbRedoDiscoverySkipReason
    message: str


@dataclass(frozen=True, slots=True)
class PdbRedoDiscoveryResult:
    """Result of discovering PDB-REDO candidates."""

    candidates: tuple[PdbRedoCandidate, ...]
    skipped: tuple[PdbRedoDiscoverySkip, ...]

    @property
    def candidate_count(self) -> int:
        """Return the number of usable candidates."""

        return len(self.candidates)

    @property
    def skipped_count(self) -> int:
        """Return the number of skipped possible candidates."""

        return len(self.skipped)

    @property
    def pdb_ids(self) -> tuple[str, ...]:
        """Return discovered PDB IDs in candidate order."""

        return tuple(candidate.pdb_id for candidate in self.candidates)


def discover_pdb_redo_candidates(
    root: str | Path,
    *,
    require_data_json: bool = True,
    recursive: bool = True,
) -> PdbRedoDiscoveryResult:
    """Discover candidate PDB-REDO entries in a local mirror.

    Parameters
    ----------
    root
        Root directory of a local PDB-REDO mirror.
    require_data_json
        If true, candidates must have both ``<pdb_id>_final.cif`` and
        ``data.json``. If false, candidates with a final mmCIF are returned even
        when ``data.json`` is absent.
    recursive
        If true, recursively search the mirror for ``*_final.cif`` files. If
        false, only immediate child directories of ``root`` are inspected.

    Returns
    -------
    PdbRedoDiscoveryResult
        Usable candidates plus skipped possible candidates.
    """

    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise PdbRedoDiscoveryError(
            f"PDB-REDO mirror root does not exist or is not a directory: {root_path}"
        )

    possible_candidates = (
        _iter_final_cif_paths_recursive(root_path)
        if recursive
        else _iter_final_cif_paths_flat(root_path)
    )

    return _build_discovery_result(
        sorted(possible_candidates, key=_path_sort_key),
        require_data_json=require_data_json,
    )


def iter_pdb_redo_candidates(
    root: str | Path,
    *,
    require_data_json: bool = True,
    recursive: bool = True,
) -> Iterator[PdbRedoCandidate]:
    """Yield usable PDB-REDO candidates from a local mirror.

    This is a convenience wrapper around :func:`discover_pdb_redo_candidates`
    for callers that only need accepted candidates.
    """

    result = discover_pdb_redo_candidates(
        root,
        require_data_json=require_data_json,
        recursive=recursive,
    )
    yield from result.candidates


def _build_discovery_result(
    final_cif_paths: Iterable[Path],
    *,
    require_data_json: bool,
) -> PdbRedoDiscoveryResult:
    candidates: list[PdbRedoCandidate] = []
    skipped: list[PdbRedoDiscoverySkip] = []
    seen_pdb_ids: dict[str, Path] = {}

    for final_cif_path in final_cif_paths:
        pdb_id = _pdb_id_from_final_cif_path(final_cif_path)

        if pdb_id is None:
            skipped.append(
                PdbRedoDiscoverySkip(
                    pdb_id=None,
                    path=final_cif_path,
                    reason=PdbRedoDiscoverySkipReason.INVALID_PDB_ID,
                    message=(
                        "Could not derive a valid four-character PDB ID from "
                        f"final mmCIF path: {final_cif_path}"
                    ),
                )
            )
            continue

        duplicate_of = seen_pdb_ids.get(pdb_id)
        if duplicate_of is not None:
            skipped.append(
                PdbRedoDiscoverySkip(
                    pdb_id=pdb_id,
                    path=final_cif_path,
                    reason=PdbRedoDiscoverySkipReason.DUPLICATE_PDB_ID,
                    message=(
                        f"Duplicate PDB-REDO final mmCIF for {pdb_id!r}; "
                        f"already using {duplicate_of}"
                    ),
                )
            )
            continue

        entry_dir = final_cif_path.parent
        data_json_path = entry_dir / _DATA_JSON_NAME

        if require_data_json and not data_json_path.is_file():
            skipped.append(
                PdbRedoDiscoverySkip(
                    pdb_id=pdb_id,
                    path=final_cif_path,
                    reason=PdbRedoDiscoverySkipReason.MISSING_DATA_JSON,
                    message=f"Missing data.json for PDB-REDO entry {pdb_id!r}.",
                )
            )
            continue

        seen_pdb_ids[pdb_id] = final_cif_path

        candidates.append(
            PdbRedoCandidate(
                pdb_id=pdb_id,
                entry_dir=entry_dir,
                final_cif_path=final_cif_path,
                data_json_path=data_json_path if data_json_path.is_file() else None,
            )
        )

    return PdbRedoDiscoveryResult(
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.pdb_id)),
        skipped=tuple(sorted(skipped, key=_skip_sort_key)),
    )


def _iter_final_cif_paths_recursive(root: Path) -> Iterator[Path]:
    yield from root.rglob(f"*{_FINAL_CIF_SUFFIX}")


def _iter_final_cif_paths_flat(root: Path) -> Iterator[Path]:
    for entry_dir in sorted(root.iterdir(), key=_path_sort_key):
        if not entry_dir.is_dir():
            continue

        pdb_id = entry_dir.name.lower()
        final_cif_path = entry_dir / f"{pdb_id}{_FINAL_CIF_SUFFIX}"

        if final_cif_path.is_file():
            yield final_cif_path


def _pdb_id_from_final_cif_path(path: Path) -> str | None:
    name = path.name.lower()

    if not name.endswith(_FINAL_CIF_SUFFIX):
        return None

    pdb_id = name[: -len(_FINAL_CIF_SUFFIX)]
    if not _looks_like_pdb_id(pdb_id):
        return None

    return pdb_id


def _looks_like_pdb_id(value: str) -> bool:
    return len(value) == 4 and value[0].isdigit() and value.isalnum()


def _skip_sort_key(skip: PdbRedoDiscoverySkip) -> tuple[str, str, str]:
    return (
        skip.pdb_id or "",
        skip.reason.value,
        str(skip.path),
    )


def _path_sort_key(path: Path) -> str:
    return str(path)


__all__ = [
    "PdbRedoCandidate",
    "PdbRedoDiscoveryError",
    "PdbRedoDiscoveryResult",
    "PdbRedoDiscoverySkip",
    "PdbRedoDiscoverySkipReason",
    "discover_pdb_redo_candidates",
    "iter_pdb_redo_candidates",
]
