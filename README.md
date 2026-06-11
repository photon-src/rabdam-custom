# RABDAM custom

RABDAM custom calculates per-atom BDamage scores, raw protein Bnet, and
Bnet-percentile for protein crystal structures. It is a from-scratch
implementation of the RABDAM 2 workflow created by the Garman Group at 
the University of Oxford

Use this near the end of refinement, or on deposited models, where per-atom
B-factors are meaningful.

## What It Calculates

BDamage is a per-atom score that compares an atom's B-factor with atoms in a
similar local packing-density environment. By default, packing density is the
number of non-hydrogen crystal-environment atoms within 7.0 A.

Bnet summarizes likely protein radiation damage from the BDamage values of
Asp/Glu side-chain carboxyl oxygen atoms. Bnet-percentile compares the raw Bnet
against the included resolution-matched reference database, making damage scores
easier to compare between structures solved at different resolutions.

## Install

RABDAM custom requires Python 3.10 or newer and the dependencies listed in
`pyproject.toml`: `gemmi`, `numpy`, and `scipy`.

From the repository root:

```bash
./rabdam --help
```

## Run

Local structure file:

```bash
./rabdam model.cif
```

RCSB/PDB ID:

```bash
./rabdam 1LYZ
```

Batch run:

```bash
./rabdam 1LYZ 2BLP model.cif
```

If the structure file does not contain resolution metadata, provide it for
Bnet-percentile:

```bash
./rabdam model.pdb --bnet-resolution-angstrom 1.8
```

Accepted inputs are `.cif`, `.mmcif`, `.pdb`, and classic 4-character PDB IDs.
Pass one input for a single run or multiple inputs for a batch run.

## Output

Single and batch runs write CSVs in `output/` by default, using the PDB ID or
input filename stem: for example `output/1LYZ_BDamage.csv` or
`output/model_BDamage.csv`. Use `--output-dir` to choose another output
directory; use `--output-csv` only when running one structure.

Each CSV contains one row per selected atom. The main result columns are:

- `PD`: packing-density neighbour count.
- `AVRG_BF`: local average B-factor for atoms with similar packing density.
- `BDAM`: per-atom BDamage score.

The terminal summary reports the raw protein Bnet, Bnet-percentile when
available, the reference database used, and the local resolution range used for
percentile ranking.

## Selection Defaults

By default RABDAM custom removes hydrogens, resolves alternate atom sites by
highest occupancy, scores protein atoms, and requires a protein selection.

Useful selection options:

```bash
./rabdam model.cif --include-hetatm
./rabdam model.cif --include-nucleic-acid
./rabdam model.cif --remove-component HOH
./rabdam model.cif --add-component LIG
./rabdam model.cif --remove-atom-serial 123 --add-atom-serial 456
```

Changing the selected atoms changes the BDamage distribution and the Bnet
median. For comparable Bnet-percentile values, prefer the default protein
selection unless you have a specific reason to change it.

## Differences From RABDAM 2

RABDAM custom is a calculation-compatible rewrite of RABDAM 2, not a
feature-for-feature clone. Its purpose is to keep the scientific calculation
stable while making the program easier to install, faster to run, easier to
test, and better suited to scripted use.

The core calculation is preserved. Under normal default use, `PD`, `AVRG_BF`,
`BDAM`, and raw protein Bnet are intended to match RABDAM 2. The rewrite keeps
the same scientific model: crystallographic symmetry expansion, neighbouring
unit-cell translation, strict 7.0 A packing-density counts, BDamage sliding
windows, Asp/Glu protein Bnet sites, and resolution-matched Bnet-percentile
comparison.

The main expected numerical difference is Bnet-percentile. Percentile is not an
intrinsic property of the input structure; it depends on the reference
population. RABDAM 2 used the original 2020 reference CSV, while RABDAM custom
uses a newer, larger bundled reference database at
`database/reference/database.csv`. Raw Bnet can therefore match while
Bnet-percentile changes.

The rewrite improves practical use in several ways:

- Faster calculations: packing-density neighbour searches use `cKDTree`,
  translation and trimming are fused by default to avoid materializing the full
  3x3x3 crystal block, and BDamage window averages use cumulative sums instead
  of pandas rolling means.
- Larger-structure speedups: in large-structure runs, these changes can make
  RABDAM custom complete more than 80 times faster than RABDAM 2.
- Lighter installation: the runtime stack is Gemmi, NumPy, and SciPy. It does
  not require cctbx, CCP4/PDBCUR, pandas, matplotlib, or requests.
- Clearer structure: the code is split into small tested modules for input,
  structure preparation, crystal expansion, packing density, BDamage, Bnet, and
  reference-database handling.
- Cleaner automation: the CLI uses
  `rabdam STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]`, supports simple
  batch runs, and writes compact per-atom CSVs plus terminal summaries.
- Maintained, modular reference data: the included reference database can be
  regenerated with the separate PDB-REDO build pipeline, and the active
  reference CSV is easy to swap for another compatible database.

These improvements intentionally remove some RABDAM 2 application features.
RABDAM custom does not support the legacy text input file, `-f`, `-i`, `-r`
dataframe restart modes, interactive overwrite prompts, HTML reports, plots,
BDamage-colored coordinate files, or restartable dataframe outputs.

## Limits And Assumptions

The CLI expects a single-model crystallographic structure with unit-cell and
space-group metadata. BDamage and Bnet rely on per-atom B-factors; structures
with grouped or otherwise non-comparable B-factors should be interpreted with
care. Bnet and Bnet-percentile are intended for cryo-temperature protein crystal
structures.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Citations

RABDAM: Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
J Appl Cryst 51, 552-559. https://doi.org/10.1107/S1600576718002509

BDamage: Gerstel M, Deane CM & Garman EF (2015).
J Synchrotron Radiat 22, 201-212. https://doi.org/10.1107/S1600577515002131

Bnet and Bnet-percentile: Shelley KL & Garman EF (2022).
Nat Commun 13, 1314. https://doi.org/10.1038/s41467-022-28934-0
