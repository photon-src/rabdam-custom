# Bnet Reference Database Build

Most users do not need this pipeline. RABDAM custom already includes an active
Bnet reference database at `database/reference/database.csv`, built from a
PDB-REDO rsync snapshot 8.20 downloaded in 2026-06.

Use `build_bnet_database` only when you need to rebuild or replace that
reference from a local PDB-REDO mirror.

## Input

Download the PDB-REDO databank mirror with `rsync` before running the builder.
The recommended local mirror path is `downloads/pdb-redo`:

```bash
mkdir -p downloads/pdb-redo
rsync -av --exclude=attic rsync://rsync.pdb-redo.eu/pdb-redo/ downloads/pdb-redo/
```

The builder expects that mirror to contain final mmCIF files and, by default,
companion `data.json` files. A typical entry looks like:

```text
downloads/pdb-redo/
  1abc/
    1abc_final.cif
    data.json
```

Recursive discovery is enabled by default, so nested mirrors are supported.

## Run

Small trial run:

```bash
./build_bnet_database downloads/pdb-redo \
  --output-dir database/output-trial \
  --max-candidates 100 \
```

Full build:

```bash
./build_bnet_database downloads/pdb-redo --output-dir database/output
```

Add `--diagnostics` when you want the larger accepted-details and all-scores
CSVs.

## Outputs

Standard outputs are written under `--output-dir`:

- `database.accepted.csv`: accepted reference rows.
- `database.rejected.csv`: rejected entries and reasons.
- `database.csv`: final sorted reference CSV.
- `database.manifest.json`: provenance and build settings for `database.csv`.
- `rcsb_temperature_cache.csv`: compact collection-temperature cache.

With `--diagnostics`, the builder also writes:

- `database.accepted_details.csv`: detailed metadata for accepted rows.
- `database.all_scores.csv`: broad diagnostics for every processed candidate.

The active default reference for normal `rabdam` runs is
`database/reference/database.csv`. Replace it only after checking the new
`database.csv` and manifest.

## Eligibility Policy

The default build accepts entries suitable for the protein cryo Asp/Glu Bnet
reference:

- X-ray crystallography.
- Single model.
- Contains protein.
- Resolution <= 3.5 A.
- Rfree < 0.4.
- Verified collection temperature from 80 to 120 K.
- At least 20 Asp/Glu side-chain carboxyl oxygen atoms.
- No Asp/Glu side chain with total occupancy below one.
- Per-atom B-factor model.

Nucleic-acid polymers are allowed by default for Bnet builds.
Use `--reject-nucleic-acid` to reject them.

## Resume And Metadata

Builds resume by default: existing accepted, rejected, and all-scores rows are
used to skip already processed PDB IDs. Use `--overwrite` to start fresh, or
`--no-resume` to attempt entries even if they appear in existing outputs.

When local metadata cannot verify collection temperature, the builder fetches
canonical RCSB mmCIF metadata by default. Use `--no-fetch-rcsb-temperatures` for
fully local/offline builds.

Run `./build_bnet_database --help` for the full option list.
