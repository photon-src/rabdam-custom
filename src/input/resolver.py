"""
Resolve user-supplied structure inputs for RABDAM.

This module takes raw user input such as:

    "model.cif"
    "model.pdb"
    "2blx"

and decides whether each input is a local structure file or an RCSB/PDB ID.

Resolution order:
1. If the input exists as a local file, treat it as a local file
2. Otherwise, try a case-insensitive local filename match
3. Otherwise, if it looks like a classic PDB ID, treat it as an RCSB/PDB ID
4. Otherwise, raise a clear error
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class InputResolutionError(ValueError):
    """Raised when RABDAM cannot understand or resolve a structure input."""


class InputBatchResolutionError(InputResolutionError):
    """Raised when one or more structure inputs fail during batch resolution."""

    def __init__(self, errors: list[tuple[str, InputResolutionError]]):
        self.errors = errors
        summary = "\n".join(
            f"- {raw_input!r}: {error}"
            for raw_input, error in errors
        )
        super().__init__(
            "One or more structure inputs could not be resolved:\n" + summary
        )


class StructureSourceType(str, Enum):
    """Where the structure input comes from."""

    LOCAL_FILE = "local_file"
    RCSB_ID = "rcsb_id"


class StructureFileFormat(str, Enum):
    """Accepted structure file formats."""

    MMCIF = "mmcif"
    PDB = "pdb"


@dataclass(frozen=True, slots=True)
class ResolvedStructureInput:
    """
    A normalized description of one structure input.

    For a local file, local_path will be set.
    """

    original_input: str
    source_type: StructureSourceType
    file_format: StructureFileFormat
    local_path: Path | None = None
    structure_id: str | None = None

    @property
    def needs_download(self) -> bool:
        """Return True if this input still needs to be downloaded."""
        return self.source_type == StructureSourceType.RCSB_ID

    @property
    def is_local_file(self) -> bool:
        """Return True if this input points to an existing local file."""
        return self.source_type == StructureSourceType.LOCAL_FILE

def resolve_structure_input(raw_input: str) -> ResolvedStructureInput:
    """
    Resolve one user-supplied structure input.

    Resolution order:
    1. If the input exists as a local path, treat it as a local file.
    2. If not local, try a case-insensitive local filename match.
    3. If not local, check whether it looks like a valid PDB ID.
    4. Otherwise, raise a clear error.
    """

    cleaned_input = _clean_raw_input(raw_input)
    possible_path = _expand_path(cleaned_input)

    if possible_path.exists():
        return _resolve_local_file(
            cleaned_input,
            original_input=raw_input,
        )

    case_insensitive_path = _find_case_insensitive_local_path(possible_path)
    if case_insensitive_path is not None:
        return _resolve_local_file(
            str(case_insensitive_path),
            original_input=raw_input,
        )

    if is_valid_classic_pdb_id(cleaned_input):
        return _resolve_rcsb_id(
            cleaned_input,
            original_input=raw_input,
        )

    raise InputResolutionError(
        f"Could not resolve structure input {raw_input!r}.\n"
        "Expected an existing local .cif, .mmcif, or .pdb file "
        "or a classic 4-character PDB ID such as 2blx."
    )


def resolve_many_structure_inputs(
    raw_inputs: Sequence[str],
) -> list[ResolvedStructureInput]:
    """
    Resolve multiple structure inputs in a batch run.

    If any inputs fail, all inputs are attempted before raising a batch error.
    """

    if not raw_inputs:
        raise InputResolutionError("No structure inputs were provided.")

    resolved_inputs: list[ResolvedStructureInput] = []
    errors: list[tuple[str, InputResolutionError]] = []

    for raw_input in raw_inputs:
        try:
            resolved_inputs.append(resolve_structure_input(raw_input))
        except InputResolutionError as error:
            errors.append((raw_input, error))

    if errors:
        raise InputBatchResolutionError(errors)

    return resolved_inputs

def is_valid_classic_pdb_id(value: str) -> bool:
    """
    Return True if value looks like a classic 4-character PDB ID.

    Classic PDB IDs are four characters long and usually begin with a digit.
    """

    return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", value.strip()))

def detect_structure_format_from_path(path: Path) -> StructureFileFormat:
    """
    Detect structure file format from a local file path.

    Supported:
        .cif
        .mmcif
        .pdb
    """

    suffixes = [suffix.lower() for suffix in path.suffixes]

    if not suffixes:
        raise InputResolutionError(
            f"Could not detect structure format for {path!s}: file has no extension."
        )

    final_suffix = suffixes[-1]

    if final_suffix in {".cif", ".mmcif"}:
        return StructureFileFormat.MMCIF

    if final_suffix == ".pdb":
        return StructureFileFormat.PDB

    raise InputResolutionError(
        f"Unsupported structure file extension for {path!s}.\n"
        "Supported extensions are: .cif, .mmcif, and .pdb"
    )

def _resolve_local_file(
    path_text: str,
    *,
    original_input: str,
) -> ResolvedStructureInput:
    """Resolve and validate a local structure file."""

    path = _expand_path(path_text)

    if not path.exists():
        raise InputResolutionError(f"Local structure file does not exist: {path!s}")

    if not path.is_file():
        raise InputResolutionError(
            f"Expected a structure file, but got a directory or non-file path: {path!s}"
        )

    file_format = detect_structure_format_from_path(path)

    return ResolvedStructureInput(
        original_input=original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=file_format,
        local_path=path.resolve(),
        structure_id=None,
    )

def _resolve_rcsb_id(
    structure_id: str,
    *,
    original_input: str,
) -> ResolvedStructureInput:
    """
    Resolve an RCSB/PDB accession ID.

    mmCIF only.
    """

    cleaned_id = structure_id.strip()

    if not is_valid_classic_pdb_id(cleaned_id):
        raise InputResolutionError(
            f"Invalid classic PDB ID: {structure_id!r}.\n"
            "Expected a 4-character ID such as 2blx, 1abc, or 7xyz."
        )

    return ResolvedStructureInput(
        original_input=original_input,
        source_type=StructureSourceType.RCSB_ID,
        file_format=StructureFileFormat.MMCIF,
        local_path=None,
        structure_id=cleaned_id.upper(),
    )


def _clean_raw_input(raw_input: str) -> str:
    """Clean and validate the raw user input string."""

    if not isinstance(raw_input, str):
        raise InputResolutionError(
            f"Structure input must be a string, not {type(raw_input).__name__}."
        )

    cleaned = raw_input.strip()

    if not cleaned:
        raise InputResolutionError("Structure input cannot be empty.")

    return cleaned


def _expand_path(path_text: str) -> Path:
    """
    Expand user and environment variables in a path.
    """

    expanded = os.path.expandvars(path_text)
    return Path(expanded).expanduser()


def _find_case_insensitive_local_path(path: Path) -> Path | None:
    """
    Return a local path whose filename matches case-insensitively.

    Exact paths are handled before this function is called. This fallback only
    compares the final path component inside an existing parent directory, so it
    avoids guessing across unrelated directory names.
    """

    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        return None

    requested_name = path.name.lower()
    matches = [
        candidate
        for candidate in parent.iterdir()
        if candidate.name.lower() == requested_name
    ]

    if not matches:
        return None

    if len(matches) > 1:
        matching_names = ", ".join(sorted(candidate.name for candidate in matches))
        raise InputResolutionError(
            f"Local structure input {path!s} is ambiguous because multiple files "
            f"match case-insensitively: {matching_names}."
        )

    return matches[0]
