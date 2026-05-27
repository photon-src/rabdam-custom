"""
Validation checks for prepared RABDAM structures.
"""

from structure.models import (
    PreparedAtom,
    StructurePreparationError,
    StructurePreparationOptions,
)


def check_prepared_structure(
    *,
    cleaned_atoms: tuple[PreparedAtom, ...],
    selected_atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> None:
    """
    Validate the prepared structure before later RABDAM stages run.
    """

    if not cleaned_atoms:
        raise StructurePreparationError(
            "No atoms remain after structure preparation."
        )

    if not selected_atoms:
        raise StructurePreparationError(
            "No atoms were selected for BDamage calculation."
        )

    if options.require_protein_selection:
        if not any(atom.is_protein for atom in selected_atoms):
            raise StructurePreparationError(
                "No protein atoms were selected for BDamage calculation."
            )
