"""Read metadata for PDB-REDO Bnet reference-database candidates.

This module extracts cheap metadata before the RABDAM calculation.
It is intentionally conservative: missing values are returned as ``None`` and
can be handled later by the generic Bnet eligibility checks.

The main entry point is:

    read_pdb_redo_metadata(candidate)

Metadata is read from ``data.json`` when available, with mmCIF fallback for
common crystallographic fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, cast

import gemmi

from .discover import PdbRedoCandidate


_MISSING_VALUES = frozenset({"", ".", "?", "null", "none", "nan", "na", "n/a"})


class PdbRedoMetadataError(ValueError):
    """Raised when PDB-REDO metadata cannot be read."""


@dataclass(frozen=True, slots=True)
class PdbRedoMetadata:
    """Metadata extracted for one PDB-REDO candidate."""

    pdb_id: str
    resolution_angstrom: float | None = None
    r_work: float | None = None
    r_free: float | None = None
    temperature_k: float | None = None
    wilson_b: float | None = None
    b_factor_restraint_weight: float | None = None
    resolution_source: str | None = None
    r_work_source: str | None = None
    r_free_source: str | None = None
    temperature_source: str | None = None
    wilson_b_source: str | None = None
    b_factor_restraint_weight_source: str | None = None
    data_json_path: Path | None = None
    final_cif_path: Path | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class _RawMetadataValue:
    value: object
    source: str


@dataclass(frozen=True, slots=True)
class _FloatMetadataValue:
    value: float | None
    source: str | None = None


def read_pdb_redo_metadata(
    candidate: PdbRedoCandidate,
    *,
    use_mmcif_fallback: bool = True,
) -> PdbRedoMetadata:
    """Read metadata for a discovered PDB-REDO candidate.

    Parameters
    ----------
    candidate
        Candidate discovered by ``discover_pdb_redo_candidates``.
    use_mmcif_fallback
        If true, common crystallographic fields missing from ``data.json`` are
        extracted from ``<pdb_id>_final.cif`` when possible.

    Returns
    -------
    PdbRedoMetadata
        Extracted metadata. Individual values may be ``None`` if unavailable.
    """

    warnings: list[str] = []
    data: Mapping[str, Any] = {}

    if candidate.data_json_path is not None:
        data = _read_json_object(candidate.data_json_path)
    else:
        warnings.append(
            "data.json is missing; metadata will rely on mmCIF fallback."
        )

    cif_values: Mapping[str, str] = {}
    if use_mmcif_fallback:
        try:
            cif_values = _read_mmcif_scalar_values(candidate.final_cif_path)
        except (OSError, RuntimeError, ValueError) as error:
            warnings.append(f"Could not read mmCIF fallback metadata: {error}")

    resolution = _first_float(
        _json_values_by_alias(data, _RESOLUTION_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _RESOLUTION_CIF_TAGS),
    )
    r_work = _first_float(
        _json_values_by_alias(data, _R_WORK_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _R_WORK_CIF_TAGS),
    )
    r_free = _first_float(
        _json_values_by_alias(data, _R_FREE_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _R_FREE_CIF_TAGS),
    )
    temperature = _first_float(
        _json_values_by_alias(data, _TEMPERATURE_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _TEMPERATURE_CIF_TAGS),
    )
    wilson_b = _first_float(
        _json_values_by_alias(data, _WILSON_B_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _WILSON_B_CIF_TAGS),
    )
    b_factor_restraint_weight = _first_float(
        _json_values_by_alias(data, _B_FACTOR_RESTRAINT_JSON_ALIASES),
        _cif_values_by_tag(cif_values, _B_FACTOR_RESTRAINT_CIF_TAGS),
    )

    return PdbRedoMetadata(
        pdb_id=candidate.pdb_id,
        resolution_angstrom=resolution.value,
        r_work=r_work.value,
        r_free=r_free.value,
        temperature_k=temperature.value,
        wilson_b=wilson_b.value,
        b_factor_restraint_weight=b_factor_restraint_weight.value,
        resolution_source=resolution.source,
        r_work_source=r_work.source,
        r_free_source=r_free.source,
        temperature_source=temperature.source,
        wilson_b_source=wilson_b.source,
        b_factor_restraint_weight_source=b_factor_restraint_weight.source,
        data_json_path=candidate.data_json_path,
        final_cif_path=candidate.final_cif_path,
        warnings=tuple(warnings),
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as error:
        raise PdbRedoMetadataError(
            f"Could not read PDB-REDO data.json: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise PdbRedoMetadataError(
            f"Could not parse PDB-REDO data.json: {path}"
        ) from error

    if not isinstance(value, Mapping):
        raise PdbRedoMetadataError(
            f"PDB-REDO data.json must contain a JSON object: {path}"
        )

    return cast(Mapping[str, Any], value)


def _read_mmcif_scalar_values(path: Path) -> dict[str, str]:
    """Read common mmCIF scalar tag-value pairs using Gemmi."""

    document = gemmi.cif.read_file(str(path))
    if len(document) == 0:
        return {}

    values: dict[str, str] = {}
    block = document[0]

    for tag in _ALL_CIF_TAGS:
        value = block.find_value(tag)
        if value is not None:
            values[tag.casefold()] = _strip_cif_quotes(value)

    return values


def _json_values_by_alias(
    data: Mapping[str, Any],
    aliases: frozenset[str],
) -> tuple[_RawMetadataValue, ...]:
    matches: list[_RawMetadataValue] = []

    for key_path, value in _walk_json_values(data):
        normalized_key_path = tuple(_normalize_name(part) for part in key_path)
        normalized_leaf_key = normalized_key_path[-1] if normalized_key_path else ""
        joined_source_path = ".".join(str(part) for part in key_path)
        source = f"data_json:{joined_source_path}"

        if normalized_leaf_key in aliases:
            matches.append(_RawMetadataValue(value=value, source=source))
            continue

        joined_path = ".".join(normalized_key_path)
        condensed_path = "".join(normalized_key_path)
        if joined_path in aliases or condensed_path in aliases:
            matches.append(_RawMetadataValue(value=value, source=source))

    return tuple(matches)


def _cif_values_by_tag(
    cif_values: Mapping[str, str],
    tags: frozenset[str],
) -> tuple[_RawMetadataValue, ...]:
    return tuple(
        _RawMetadataValue(value=value, source=f"mmcif:{tag.casefold()}")
        for tag, value in cif_values.items()
        if tag.casefold() in tags
    )


def _walk_json_values(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_json_values(child, path=(*path, str(key)))
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json_values(child, path=(*path, str(index)))
        return

    yield path, value


def _first_float(*groups: Iterable[_RawMetadataValue]) -> _FloatMetadataValue:
    for group in groups:
        for raw_value in group:
            number = _optional_float(raw_value.value)
            if number is not None:
                return _FloatMetadataValue(value=number, source=raw_value.source)
    return _FloatMetadataValue(value=None)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        text = _strip_cif_quotes(value.strip())
        if text.casefold() in _MISSING_VALUES:
            return None
    else:
        text = str(value)

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _strip_cif_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


_RESOLUTION_JSON_ALIASES = frozenset(
    {
        "resolution",
        "resolutionangstrom",
        "resolutionangstroms",
        "dresolutionhigh",
        "highresolution",
        "refinelsdreshigh",
    }
)

_R_WORK_JSON_ALIASES = frozenset(
    {
        "rwork",
        "rworkfactor",
        "rfactorwork",
        "rvaluework",
        "refinelsrfactorrwork",
    }
)

_R_FREE_JSON_ALIASES = frozenset(
    {
        "rfree",
        "rfreefactor",
        "rfactorfree",
        "rvaluefree",
        "refinelsrfactorrfree",
    }
)

_TEMPERATURE_JSON_ALIASES = frozenset(
    {
        "temperature",
        "temperaturek",
        "collectiontemperature",
        "collectiontemperaturek",
        "ambienttemp",
        "diffrnambienttemp",
    }
)

_WILSON_B_JSON_ALIASES = frozenset(
    {
        "wilsonb",
        "wilsonbfactor",
        "wilsonbestimate",
        "bisowilsonestimate",
        "reflnsbisowilsonestimate",
    }
)

_B_FACTOR_RESTRAINT_JSON_ALIASES = frozenset(
    {
        "bfactorrestraintweight",
        "brestraintweight",
        "brestweight",
        "adprestraintweight",
        "uvalrestraintweight",
    }
)


_RESOLUTION_CIF_TAGS = frozenset(
    {
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
        "_em_3d_reconstruction.resolution",
    }
)

_R_WORK_CIF_TAGS = frozenset(
    {
        "_refine.ls_r_factor_r_work",
        "_refine.ls_r_factor_obs",
        "_refine.ls_r_factor_all",
    }
)

_R_FREE_CIF_TAGS = frozenset(
    {
        "_refine.ls_r_factor_r_free",
    }
)

_TEMPERATURE_CIF_TAGS = frozenset(
    {
        "_diffrn.ambient_temp",
        "_exptl_crystal_grow.temp",
    }
)

_WILSON_B_CIF_TAGS = frozenset(
    {
        "_reflns.b_iso_wilson_estimate",
    }
)

_B_FACTOR_RESTRAINT_CIF_TAGS = frozenset(
    {
        "_refine.pdbx_adp_restraints_weight",
    }
)

_ALL_CIF_TAGS = frozenset(
    tag
    for tag_group in (
        _RESOLUTION_CIF_TAGS,
        _R_WORK_CIF_TAGS,
        _R_FREE_CIF_TAGS,
        _TEMPERATURE_CIF_TAGS,
        _WILSON_B_CIF_TAGS,
        _B_FACTOR_RESTRAINT_CIF_TAGS,
    )
    for tag in tag_group
)


__all__ = [
    "PdbRedoMetadata",
    "PdbRedoMetadataError",
    "read_pdb_redo_metadata",
]
