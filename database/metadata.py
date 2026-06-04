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
import gzip
import json
import math
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import gemmi

from input.rcsb import RCSB_MMCIF_DOWNLOAD_URL

from .discover import PdbRedoCandidate


_MISSING_VALUES = frozenset({"", ".", "?", "null", "none", "nan", "na", "n/a"})


class PdbRedoMetadataError(ValueError):
    """Raised when PDB-REDO metadata cannot be read."""


@dataclass(frozen=True, slots=True)
class TemperatureCacheEntry:
    """One compact cached RCSB collection-temperature lookup."""

    pdb_id: str
    temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    source: str | None = None
    status: str = "missing"
    message: str | None = None

    def __post_init__(self) -> None:
        pdb_id = str(self.pdb_id).strip().casefold()
        if not pdb_id:
            raise PdbRedoMetadataError("temperature cache pdb_id must not be empty.")

        values = tuple(
            value
            for value in (
                _optional_float(raw_value)
                for raw_value in self.temperature_values_k
            )
            if value is not None
        )
        status = str(self.status).strip() or ("found" if values else "missing")

        object.__setattr__(self, "pdb_id", pdb_id)
        object.__setattr__(self, "temperature_values_k", values)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class PdbRedoMetadata:
    """Metadata extracted for one PDB-REDO candidate."""

    pdb_id: str
    resolution_angstrom: float | None = None
    r_work: float | None = None
    r_free: float | None = None
    temperature_k: float | None = None
    temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    growth_temperature_k: float | None = None
    growth_temperature_values_k: tuple[float, ...] = field(default_factory=tuple)
    wilson_b: float | None = None
    b_factor_restraint_weight: float | None = None
    b_factor_refinement_type: str | None = None
    resolution_source: str | None = None
    r_work_source: str | None = None
    r_free_source: str | None = None
    temperature_source: str | None = None
    temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    growth_temperature_source: str | None = None
    growth_temperature_sources: tuple[str, ...] = field(default_factory=tuple)
    wilson_b_source: str | None = None
    b_factor_restraint_weight_source: str | None = None
    b_factor_refinement_type_source: str | None = None
    temperature_cache_status: str | None = None
    temperature_cache_message: str | None = None
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


@dataclass(frozen=True, slots=True)
class _FloatMetadataValues:
    values: tuple[float, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TextMetadataValue:
    value: str | None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class _RecoveredTemperatureValues:
    values: _FloatMetadataValues
    cache_status: str | None = None
    cache_message: str | None = None


def read_pdb_redo_metadata(
    candidate: PdbRedoCandidate,
    *,
    use_mmcif_fallback: bool = True,
    recover_missing_temperature: bool = True,
    temperature_cache: Mapping[str, TemperatureCacheEntry] | None = None,
    fetch_rcsb_temperature: bool = False,
) -> PdbRedoMetadata:
    """Read metadata for a discovered PDB-REDO candidate.

    Parameters
    ----------
    candidate
        Candidate discovered by ``discover_pdb_redo_candidates``.
    use_mmcif_fallback
        If true, common crystallographic fields missing from ``data.json`` are
        extracted from ``<pdb_id>_final.cif`` when possible.
    recover_missing_temperature
        If true, collection temperature missing from ``data.json`` and the
        final mmCIF is looked up in local PDB-REDO companion CIFs, then a
        compact cache, then optionally canonical RCSB mmCIF metadata.
    temperature_cache
        Optional compact temperature cache keyed by PDB ID.
    fetch_rcsb_temperature
        If true, download canonical RCSB mmCIF only when local/cache metadata
        cannot verify collection temperature.

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
    final_cif_collection_temperature_values: tuple[_RawMetadataValue, ...] = ()
    final_cif_growth_temperature_values: tuple[_RawMetadataValue, ...] = ()
    if use_mmcif_fallback:
        try:
            cif_document = _read_cif_document(candidate.final_cif_path)
            cif_values = _read_mmcif_scalar_values_from_document(cif_document)
            final_cif_collection_temperature_values = (
                _cif_values_by_tag_from_document(
                    cif_document,
                    _COLLECTION_TEMPERATURE_CIF_TAGS,
                    source_prefix="mmcif",
                )
            )
            final_cif_growth_temperature_values = _cif_values_by_tag_from_document(
                cif_document,
                _GROWTH_TEMPERATURE_CIF_TAGS,
                source_prefix="mmcif",
            )
        except (OSError, RuntimeError, ValueError) as error:
            warnings.append(f"Could not read mmCIF fallback metadata: {error}")

    resolution = _first_float(
        _json_values_by_path_or_alias(
            data,
            _RESOLUTION_JSON_PATHS,
            _RESOLUTION_JSON_ALIASES,
        ),
        _cif_values_by_tag(cif_values, _RESOLUTION_CIF_TAGS),
    )
    r_work = _first_float(
        _json_values_by_path_or_alias(
            data,
            _R_WORK_JSON_PATHS,
            _R_WORK_JSON_ALIASES,
        ),
        _cif_values_by_tag(cif_values, _R_WORK_CIF_TAGS),
    )
    r_free = _first_float(
        _json_values_by_path_or_alias(
            data,
            _R_FREE_JSON_PATHS,
            _R_FREE_JSON_ALIASES,
        ),
        _cif_values_by_tag(cif_values, _R_FREE_CIF_TAGS),
    )
    temperature_values = _first_floats(
        _json_values_by_path_or_alias(
            data,
            _COLLECTION_TEMPERATURE_JSON_PATHS,
            _COLLECTION_TEMPERATURE_JSON_ALIASES,
        ),
        final_cif_collection_temperature_values,
    )
    growth_temperature_values = _first_floats(
        _json_values_by_path_or_alias(
            data,
            _GROWTH_TEMPERATURE_JSON_PATHS,
            _GROWTH_TEMPERATURE_JSON_ALIASES,
        ),
        final_cif_growth_temperature_values,
    )
    temperature_cache_status: str | None = None
    temperature_cache_message: str | None = None
    if (
        recover_missing_temperature
        and not temperature_values.values
        and use_mmcif_fallback
    ):
        recovered_temperature = _recover_temperature_values(
            candidate,
            temperature_cache=temperature_cache,
            fetch_rcsb_temperature=fetch_rcsb_temperature,
            warnings=warnings,
        )
        temperature_values = recovered_temperature.values
        temperature_cache_status = recovered_temperature.cache_status
        temperature_cache_message = recovered_temperature.cache_message

    wilson_b = _first_float(
        _json_values_by_path_or_alias(
            data,
            _WILSON_B_JSON_PATHS,
            _WILSON_B_JSON_ALIASES,
        ),
        _cif_values_by_tag(cif_values, _WILSON_B_CIF_TAGS),
    )
    b_factor_restraint_weight = _first_float(
        _json_values_by_path_or_alias(
            data,
            _B_FACTOR_RESTRAINT_JSON_PATHS,
            _B_FACTOR_RESTRAINT_JSON_ALIASES,
        ),
        _cif_values_by_tag(cif_values, _B_FACTOR_RESTRAINT_CIF_TAGS),
    )
    b_factor_refinement_type = _first_text(
        _json_values_by_path_or_alias(
            data,
            _B_FACTOR_REFINEMENT_TYPE_JSON_PATHS,
            _B_FACTOR_REFINEMENT_TYPE_JSON_ALIASES,
        )
    )

    return PdbRedoMetadata(
        pdb_id=candidate.pdb_id,
        resolution_angstrom=resolution.value,
        r_work=r_work.value,
        r_free=r_free.value,
        temperature_k=(
            temperature_values.values[0] if temperature_values.values else None
        ),
        temperature_values_k=temperature_values.values,
        growth_temperature_k=(
            growth_temperature_values.values[0]
            if growth_temperature_values.values
            else None
        ),
        growth_temperature_values_k=growth_temperature_values.values,
        wilson_b=wilson_b.value,
        b_factor_restraint_weight=b_factor_restraint_weight.value,
        b_factor_refinement_type=b_factor_refinement_type.value,
        resolution_source=resolution.source,
        r_work_source=r_work.source,
        r_free_source=r_free.source,
        temperature_source=(
            temperature_values.sources[0] if temperature_values.sources else None
        ),
        temperature_sources=temperature_values.sources,
        growth_temperature_source=(
            growth_temperature_values.sources[0]
            if growth_temperature_values.sources
            else None
        ),
        growth_temperature_sources=growth_temperature_values.sources,
        wilson_b_source=wilson_b.source,
        b_factor_restraint_weight_source=b_factor_restraint_weight.source,
        b_factor_refinement_type_source=b_factor_refinement_type.source,
        temperature_cache_status=temperature_cache_status,
        temperature_cache_message=temperature_cache_message,
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

    document = _read_cif_document(path)
    return _read_mmcif_scalar_values_from_document(document)


def _read_cif_document(path: Path) -> gemmi.cif.Document:
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return gemmi.cif.read_string(handle.read())

    return gemmi.cif.read_file(str(path))


def _read_mmcif_scalar_values_from_document(
    document: gemmi.cif.Document,
) -> dict[str, str]:
    if len(document) == 0:
        return {}

    values: dict[str, str] = {}
    block = document[0]

    for tag in _ALL_CIF_TAGS:
        value = block.find_value(tag)
        if value is not None:
            values[tag.casefold()] = _strip_cif_quotes(value)

    return values


def _cif_values_by_tag_from_document(
    document: gemmi.cif.Document,
    tags: frozenset[str],
    *,
    source_prefix: str,
) -> tuple[_RawMetadataValue, ...]:
    if len(document) == 0:
        return ()

    values: list[_RawMetadataValue] = []
    block = document[0]

    for tag in tags:
        scalar_value = block.find_value(tag)
        if scalar_value is not None:
            values.append(
                _RawMetadataValue(
                    value=_strip_cif_quotes(scalar_value),
                    source=f"{source_prefix}:{tag.casefold()}",
                )
            )

        loop = block.find_loop(tag)
        if loop is None:
            continue

        for loop_value in loop:
            values.append(
                _RawMetadataValue(
                    value=_strip_cif_quotes(loop_value),
                    source=f"{source_prefix}:{tag.casefold()}",
                )
            )

    return tuple(values)


def _recover_temperature_values(
    candidate: PdbRedoCandidate,
    *,
    temperature_cache: Mapping[str, TemperatureCacheEntry] | None,
    fetch_rcsb_temperature: bool,
    warnings: list[str],
) -> _RecoveredTemperatureValues:
    for cif_path in _iter_temperature_companion_cif_paths(candidate):
        try:
            document = _read_cif_document(cif_path)
        except (OSError, RuntimeError, ValueError) as error:
            warnings.append(
                "Could not read local temperature fallback metadata from "
                f"{cif_path}: {error}"
            )
            continue

        values = _first_floats(
            _cif_values_by_tag_from_document(
                document,
                _COLLECTION_TEMPERATURE_CIF_TAGS,
                source_prefix=f"mmcif:{cif_path.name}",
            )
        )
        if values.values:
            return _RecoveredTemperatureValues(values=values)

    cache_entry = (
        temperature_cache.get(candidate.pdb_id.casefold())
        if temperature_cache is not None
        else None
    )
    if cache_entry is not None:
        if cache_entry.temperature_values_k:
            return _RecoveredTemperatureValues(
                values=_FloatMetadataValues(
                    values=cache_entry.temperature_values_k,
                    sources=(cache_entry.source or "temperature_cache",),
                ),
                cache_status=cache_entry.status,
                cache_message=cache_entry.message,
            )

        warnings.append(
            "Temperature was not recovered from compact cache: "
            f"{cache_entry.message or cache_entry.status}."
        )
        return _RecoveredTemperatureValues(
            values=_FloatMetadataValues(),
            cache_status=cache_entry.status,
            cache_message=cache_entry.message,
        )

    if not fetch_rcsb_temperature:
        return _RecoveredTemperatureValues(values=_FloatMetadataValues())

    try:
        cif_text = _download_rcsb_mmcif_text(candidate.pdb_id)
        document = gemmi.cif.read_string(cif_text)
        values = _first_floats(
            _cif_values_by_tag_from_document(
                document,
                _COLLECTION_TEMPERATURE_CIF_TAGS,
                source_prefix="rcsb_mmcif",
            )
        )
    except (OSError, RuntimeError, ValueError, PdbRedoMetadataError) as error:
        message = str(error)
        warnings.append(f"Could not recover RCSB mmCIF temperature: {message}")
        return _RecoveredTemperatureValues(
            values=_FloatMetadataValues(),
            cache_status="error",
            cache_message=message,
        )

    if values.values:
        return _RecoveredTemperatureValues(
            values=values,
            cache_status="found",
            cache_message=None,
        )

    message = "RCSB mmCIF did not contain recognized temperature metadata."
    warnings.append(message)
    return _RecoveredTemperatureValues(
        values=values,
        cache_status="missing",
        cache_message=message,
    )


def _iter_temperature_companion_cif_paths(
    candidate: PdbRedoCandidate,
) -> tuple[Path, ...]:
    pdb_id = candidate.pdb_id.lower()
    possible_paths = (
        candidate.entry_dir / f"{pdb_id}_0cyc.cif.gz",
        candidate.entry_dir / f"{pdb_id}_besttls.cif.gz",
        candidate.entry_dir / f"{pdb_id}_0cyc.cif",
        candidate.entry_dir / f"{pdb_id}_besttls.cif",
    )

    return tuple(path for path in possible_paths if path.is_file())


def _download_rcsb_mmcif_text(pdb_id: str) -> str:
    structure_id = pdb_id.strip().lower()
    url = RCSB_MMCIF_DOWNLOAD_URL.format(structure_id=structure_id)

    try:
        with urlopen(url, timeout=30) as response:
            cif_bytes = response.read()
    except HTTPError as error:
        raise PdbRedoMetadataError(
            "RCSB temperature metadata download failed for "
            f"{pdb_id.upper()!r} with HTTP status {error.code}."
        ) from error
    except URLError as error:
        raise PdbRedoMetadataError(
            f"Could not connect to RCSB for {pdb_id.upper()!r}: {error.reason}"
        ) from error

    if not cif_bytes:
        raise PdbRedoMetadataError(
            "RCSB temperature metadata download for "
            f"{pdb_id.upper()!r} returned an empty file."
        )

    return cif_bytes.decode("utf-8")


def _json_values_by_path_or_alias(
    data: Mapping[str, Any],
    paths: frozenset[tuple[str, ...]],
    aliases: frozenset[str],
) -> tuple[_RawMetadataValue, ...]:
    path_matches = _json_values_by_path(data, paths)
    if path_matches:
        return path_matches
    return _json_values_by_alias(data, aliases)


def _json_values_by_path(
    data: Mapping[str, Any],
    paths: frozenset[tuple[str, ...]],
) -> tuple[_RawMetadataValue, ...]:
    normalized_paths = frozenset(_normalize_path(path) for path in paths)
    matches: list[_RawMetadataValue] = []

    for key_path, value in _walk_json_values(data):
        normalized_key_path = _normalize_path(key_path)
        if normalized_key_path in normalized_paths:
            joined_source_path = ".".join(str(part) for part in key_path)
            matches.append(
                _RawMetadataValue(
                    value=value,
                    source=f"data_json:{joined_source_path}",
                )
            )

    return tuple(matches)


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


def _first_floats(*groups: Iterable[_RawMetadataValue]) -> _FloatMetadataValues:
    for group in groups:
        values: list[float] = []
        sources: list[str] = []
        seen: set[tuple[float, str]] = set()

        for raw_value in group:
            number = _optional_float(raw_value.value)
            if number is None:
                continue

            key = (number, raw_value.source)
            if key in seen:
                continue

            seen.add(key)
            values.append(number)
            sources.append(raw_value.source)

        if values:
            return _FloatMetadataValues(
                values=tuple(values),
                sources=tuple(sources),
            )

    return _FloatMetadataValues()


def _first_text(*groups: Iterable[_RawMetadataValue]) -> _TextMetadataValue:
    for group in groups:
        for raw_value in group:
            text = _optional_text(raw_value.value)
            if text is not None:
                return _TextMetadataValue(value=text, source=raw_value.source)

    return _TextMetadataValue(value=None)


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


def _optional_text(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None

    text = _strip_cif_quotes(str(value).strip())
    if text.casefold() in _MISSING_VALUES:
        return None

    return text


def _strip_cif_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _normalize_path(path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_name(part) for part in path)


_RESOLUTION_JSON_PATHS = frozenset(
    {
        ("refine", "ls_d_res_high"),
        ("reflns", "d_resolution_high"),
    }
)

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

_R_WORK_JSON_PATHS = frozenset(
    {
        ("refine", "ls_r_factor_r_work"),
        ("refine", "ls_r_factor_obs"),
        ("refine", "ls_r_factor_all"),
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

_R_FREE_JSON_PATHS = frozenset(
    {
        ("refine", "ls_r_factor_r_free"),
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

_COLLECTION_TEMPERATURE_JSON_PATHS = frozenset(
    {
        ("diffrn", "ambient_temp"),
    }
)

_COLLECTION_TEMPERATURE_JSON_ALIASES = frozenset(
    {
        "collectiontemperature",
        "collectiontemperaturek",
        "ambienttemp",
        "diffrnambienttemp",
    }
)

_GROWTH_TEMPERATURE_JSON_PATHS = frozenset(
    {
        ("exptl_crystal_grow", "temp"),
    }
)

_GROWTH_TEMPERATURE_JSON_ALIASES = frozenset(
    {
        "growthtemperature",
        "growthtemperaturek",
        "crystalgrowtemperature",
        "crystalgrowtemperaturek",
        "crystalgrowthtemperature",
        "crystalgrowthtemperaturek",
        "exptlcrystalgrowtemp",
    }
)

_WILSON_B_JSON_PATHS = frozenset(
    {
        ("reflns", "b_iso_wilson_estimate"),
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

_B_FACTOR_RESTRAINT_JSON_PATHS = frozenset(
    {
        ("refine", "pdbx_adp_restraints_weight"),
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

_B_FACTOR_REFINEMENT_TYPE_JSON_PATHS = frozenset(
    {
        ("properties", "breftype"),
    }
)

_B_FACTOR_REFINEMENT_TYPE_JSON_ALIASES = frozenset(
    {
        "breftype",
        "bfactorrefinementtype",
        "brefinementtype",
        "adprefinementtype",
    }
)


_RESOLUTION_CIF_TAGS = frozenset(
    {
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
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

_COLLECTION_TEMPERATURE_CIF_TAGS = frozenset(
    {
        "_diffrn.ambient_temp",
    }
)

_GROWTH_TEMPERATURE_CIF_TAGS = frozenset(
    {
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
        _COLLECTION_TEMPERATURE_CIF_TAGS,
        _GROWTH_TEMPERATURE_CIF_TAGS,
        _WILSON_B_CIF_TAGS,
        _B_FACTOR_RESTRAINT_CIF_TAGS,
    )
    for tag in tag_group
)


__all__ = [
    "PdbRedoMetadata",
    "PdbRedoMetadataError",
    "TemperatureCacheEntry",
    "read_pdb_redo_metadata",
]
