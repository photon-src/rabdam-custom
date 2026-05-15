# RABDAM 3 Altloc Handling Note

## Purpose

For the first implementation pass, RABDAM 3 should replicate RABDAM 2 behavior as closely as possible before introducing behavioral improvements.

## Current decision

Residue-coherent alternate conformer selection will be deferred.

RABDAM 3 should initially use a RABDAM-2-compatible altloc rule:

- Treat alternate locations independently at the atom-site level.
- For each atom site, retain the alternate location with the highest occupancy.
- Remove the other alternate locations.
- Keep atoms with no alternate-location label as normal.

This means that different atoms within the same residue may select different altloc labels if their highest-occupancy conformers differ.

## Reason for deferring residue-coherent selection

A residue-coherent strategy is probably more chemically principled, because it avoids constructing hybrid residues from atom positions belonging to different conformers. For example, it would avoid keeping CB from altloc A while keeping CG from altloc B in the same residue.

However, RABDAM 2 appears to have resolved altlocs on a per-atom-site basis. Since the immediate goal is to reproduce RABDAM 2 output, changing this now could introduce differences in atom selection, packing-density calculation, and BDamage values before the baseline implementation has been validated.

## Future improvement

After RABDAM 3 can reproduce RABDAM 2 results, add a new residue-coherent altloc strategy.

Suggested future behavior:

- Group atoms by residue.
- Identify all altloc labels present in that residue.
- Score each altloc label, for example by mean occupancy, then atom count, then input order.
- Select one altloc label for the residue.
- Keep atoms with blank altloc labels.
- Keep atoms whose altloc label matches the selected residue conformer.
- Discard atoms belonging to other altloc labels.

## Suggested configuration design

Use an explicit strategy option:

```text
altloc_strategy = "atom_site"   # RABDAM 2-compatible behavior
altloc_strategy = "residue"     # future residue-coherent behavior
```

For the initial RABDAM 3 release or validation phase, default to:

```text
altloc_strategy = "atom_site"
```

Once RABDAM 2 parity has been established, consider changing the default to:

```text
altloc_strategy = "residue"
```

while keeping `"atom_site"` available as a legacy/reproducibility option.

## Implementation note

Document this as an intentional compatibility choice, not as the final preferred behavior. The long-term goal should be to preserve RABDAM 2 reproducibility while offering a more chemically coherent altloc-selection mode in RABDAM 3.
