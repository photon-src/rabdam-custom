# Symmetry Coordinate Wrapping Design Note

## Context

During crystallographic symmetry expansion, RABDAM starts from asymmetric-unit atoms and applies the input structure's space-group operations to generate the explicit full-unit-cell atom cloud.

These symmetry operations are applied in fractional coordinate space. For example:

```text
-x+1/2, -y, z+1/2
```

Applied to an atom at:

```text
(0.10, 0.20, 0.30)
```

this gives:

```text
(0.40, -0.20, 0.80)
```

The generated `y` coordinate is outside the usual `[0, 1)` fractional unit-cell range.

## Current RABDAM-3 Compatibility Choice

For the current RABDAM-3 BDamage implementation, symmetry-generated fractional coordinates are **not wrapped** into `[0, 1)`.

So:

```text
(0.40, -0.20, 0.80)
```

is kept as-is, rather than being converted to:

```text
(0.40, 0.80, 0.80)
```

This is intended to preserve compatibility with RABDAM 2, which used CCTBX `expand_to_p1()` without forcing symmetry-generated sites into the positive unit-cell range.

The goal is to keep the same geometric pathway before the later translated-cell and neighbour-box trimming stages, so that RABDAM 3 can produce BDamage values as close as possible to RABDAM 2.

## Why This Matters

The unwrapped and wrapped coordinates are crystallographically equivalent in an infinite periodic crystal because they differ by a whole unit-cell translation.

However, RABDAM performs calculations using a finite generated environment:

```text
asymmetric unit
→ symmetry-expanded full unit cell
→ translated 3×3×3 neighbouring cell block
→ trimmed neighbour box
→ packing-density counts
→ BDamage
```

In this finite construction, wrapping earlier can move an atom to the opposite side of the central unit cell before translation and trimming. That can change which generated image appears in the trimmed neighbour environment and may slightly change packing-density counts.

## Cleaner Future Design Option

Ignoring RABDAM-2 compatibility, a cleaner modern implementation could use wrapped symmetry-expanded coordinates plus an explicit periodic neighbour search.

That future design would look like:

```text
1. Apply space-group operations in fractional coordinates.
2. Wrap generated fractional coordinates into [0, 1).
3. Store the full-unit-cell atom list in a clean central-cell representation.
4. During neighbour search, explicitly handle periodic images using integer cell offsets.
```

This separates two concepts more cleanly:

```text
symmetry expansion = contents of one unit cell
periodic translation = neighbouring unit-cell images
```

Potential benefits of wrapping in a future implementation:

- cleaner and more intuitive unit-cell representation;
- easier visualization and debugging;
- clearer separation between symmetry and periodic translation;
- better compatibility with standard periodic neighbour-search methods.

Potential drawbacks:

- may produce small differences from RABDAM 2;
- requires carefully validating neighbour-search and trimming behavior;
- may require a more explicit periodic-image model to avoid changing packing-density counts.

## Current Recommendation

For the RABDAM-3 BDamage compatibility baseline:

```text
Do not wrap symmetry-generated fractional coordinates.
```

For a future redesigned packing-density engine:

```text
Consider wrapping symmetry-expanded coordinates into [0, 1) and implementing an explicit periodic neighbour-search model.
```

This should only be changed after comparing BDamage outputs against RABDAM 2 on a representative benchmark set.

## Future Unit-Cell Validation Note

`unit_cell_from_metadata()` currently validates that all unit-cell parameters are present, that lengths are positive, and that each angle is individually between 0 and 180 degrees.

It does not currently validate that the three angles form a non-degenerate 3D cell with positive volume. A stricter future validation could check the standard unit-cell volume factor:

```text
1 - cos(alpha)^2 - cos(beta)^2 - cos(gamma)^2
+ 2 cos(alpha) cos(beta) cos(gamma)
```

and reject cells where this value is non-finite or less than/equal to zero.

RABDAM 2 did not appear to perform this as an explicit up-front validation step. Its symmetry expansion delegated crystal-symmetry handling to CCTBX `expand_to_p1()`, while its later unit-cell translation computed the same volume factor directly and would fail with a raw math error for impossible angle combinations.

For RABDAM 3, adding this check would be a robustness and error-message improvement for malformed/artificial metadata, not a RABDAM-2 compatibility requirement. It should be considered low priority unless validation is being tightened generally.
