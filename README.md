# RABDAM

RABDAM calculates per-atom BDamage scores from local crystal packing density.

## Command Line

Run from the repository root without installing:

```bash
./rabdam example.cif --output-csv rabdam_BDamage.csv
```

The launcher keeps RABDAM isolated from other programs named `rabdam` on your
shell path. You can also call the module directly:

```bash
PYTHONPATH=src python3 -m rabdam example.cif --output-csv rabdam_BDamage.csv
```

Run a local structure file:

```bash
./rabdam example.cif --output-csv rabdam_BDamage.csv
```

Run an RCSB/PDB entry by ID:

```bash
./rabdam 1LYZ --cache-dir .rabdam_cache/rcsb
```

Useful options:

```bash
./rabdam example.cif \
  --packing-density-threshold 7.0 \
  --window-size-fraction 0.02 \
  --minimum-window-size 11 \
  --translation-range 1
```

The CLI accepts `.cif`, `.mmcif`, and `.pdb` files, or classic 4-character PDB
IDs. It writes a RABDAM-style CSV containing packing density, local average
B-factor, and BDamage values for selected atoms.
