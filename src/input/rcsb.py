"""
Download RCSB/PDB structure files for RABDAM.

This module takes a resolved RCSB/PDB ID input and downloads the corresponding
mmCIF file to a local downloads directory.
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from input.resolver import (
    InputResolutionError,
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)

RCSB_MMCIF_DOWNLOAD_URL = "https://files.rcsb.org/download/{structure_id}.cif"
DEFAULT_RCSB_DOWNLOAD_DIR = Path("downloads") / "rcsb"


class RcsbDownloadError(InputResolutionError):
    """Raised when RABDAM cannot download a structure from RCSB."""


class RcsbBatchDownloadError(RcsbDownloadError):
    """Raised when one or more structures fail during a batch download."""

    def __init__(
        self,
        errors: list[tuple[ResolvedStructureInput, RcsbDownloadError]],
    ):
        self.errors = errors
        summary = "\n".join(
            f"- {resolved_input.original_input!r}: {error}"
            for resolved_input, error in errors
        )
        super().__init__(
            "One or more RCSB/PDB inputs could not be downloaded:\n" + summary
        )


def download_rcsb_mmcif(
    resolved_input: ResolvedStructureInput,
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> ResolvedStructureInput:
    """
    Download an RCSB/PDB structure as an mmCIF file.

    Parameters
    ----------
    resolved_input:
        A ResolvedStructureInput with source_type == RCSB_ID.

    download_dir:
        Directory where downloaded mmCIF files should be saved.

    overwrite:
        If False, reuse an existing downloaded file.
        If True, download again even if the file already exists.

    Returns
    -------
    ResolvedStructureInput
        A new resolved input pointing to the downloaded local mmCIF file.
    """

    if resolved_input.source_type != StructureSourceType.RCSB_ID:
        raise RcsbDownloadError(
            "download_rcsb_mmcif expected an RCSB_ID input, "
            f"but got {resolved_input.source_type.value!r}."
        )

    if resolved_input.structure_id is None:
        raise RcsbDownloadError("RCSB input is missing a structure ID.")

    structure_id = resolved_input.structure_id.upper()
    output_dir = Path(download_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{structure_id}.cif"

    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return _as_downloaded_local_input(
            resolved_input=resolved_input,
            local_path=output_path,
        )

    url = RCSB_MMCIF_DOWNLOAD_URL.format(structure_id=structure_id.lower())

    try:
        with urlopen(url, timeout=30) as response:
            cif_bytes = response.read()
    except HTTPError as error:
        if error.code == 404:
            raise RcsbDownloadError(
                f"Could not find RCSB/PDB entry {structure_id!r}."
            ) from error

        raise RcsbDownloadError(
            f"RCSB download failed for {structure_id!r} with HTTP status {error.code}."
        ) from error

    except URLError as error:
        raise RcsbDownloadError(
            f"Could not connect to RCSB to download {structure_id!r}: {error.reason}"
        ) from error

    if not cif_bytes:
        raise RcsbDownloadError(
            f"RCSB download for {structure_id!r} returned an empty file."
        )

    output_path.write_bytes(cif_bytes)

    return _as_downloaded_local_input(
        resolved_input=resolved_input,
        local_path=output_path,
    )


def ensure_local_structure_file(
    resolved_input: ResolvedStructureInput,
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> ResolvedStructureInput:
    """
    Ensure a resolved structure input points to a local file.

    Local file inputs are returned unchanged.
    RCSB/PDB ID inputs are downloaded as mmCIF files.
    """

    if resolved_input.is_local_file:
        return resolved_input

    if resolved_input.needs_download:
        return download_rcsb_mmcif(
            resolved_input,
            download_dir=download_dir,
            overwrite=overwrite,
        )

    raise RcsbDownloadError(
        f"Cannot handle structure input source type: {resolved_input.source_type.value!r}"
    )


def ensure_local_structure_files(
    resolved_inputs: list[ResolvedStructureInput],
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> list[ResolvedStructureInput]:
    """
    Ensure multiple resolved structure inputs point to local files.

    Local file inputs are returned unchanged.
    RCSB/PDB ID inputs are downloaded as mmCIF files.
    If any inputs fail, all inputs are attempted before raising a batch error.
    """

    local_inputs: list[ResolvedStructureInput] = []
    errors: list[tuple[ResolvedStructureInput, RcsbDownloadError]] = []

    for resolved_input in resolved_inputs:
        try:
            local_inputs.append(
                ensure_local_structure_file(
                    resolved_input,
                    download_dir=download_dir,
                    overwrite=overwrite,
                )
            )
        except RcsbDownloadError as error:
            errors.append((resolved_input, error))

    if errors:
        raise RcsbBatchDownloadError(errors)

    return local_inputs


def _as_downloaded_local_input(
    *,
    resolved_input: ResolvedStructureInput,
    local_path: Path,
) -> ResolvedStructureInput:
    """
    Convert a downloaded RCSB input into a local-file resolved input.

    The original RCSB ID is preserved in structure_id.
    """

    return ResolvedStructureInput(
        original_input=resolved_input.original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=local_path.resolve(),
        structure_id=resolved_input.structure_id,
    )
