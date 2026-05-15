"""
BDamage atom selection for RABDAM structure preparation.
"""

from structure.models import PreparedAtom, StructurePreparationOptions


def select_bdamage_atoms(
    atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> tuple[PreparedAtom, ...]:
    """
    Select atoms that should receive BDamage values.

    Default behavior selects standard protein atoms from the cleaned atom list.
    """

    selected_atoms: list[PreparedAtom] = []
    selected_keys: set[int] = set()

    remove_component_names = normalize_component_name_set(options.remove_component_names)
    add_component_names = normalize_component_name_set(options.add_component_names)

    for atom in atoms:
        if should_select_bdamage_atom(
            atom,
            options,
            remove_component_names,
            add_component_names,
        ):
            append_selected_atom(atom, selected_atoms, selected_keys)

    return tuple(selected_atoms)


def should_select_bdamage_atom(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
    remove_component_names: frozenset[str],
    add_component_names: frozenset[str],
) -> bool:
    """
    Return True if an atom should be selected for BDamage calculation.

    Selection precedence is:
        atom serial remove > atom serial add > component remove >
        component add > default selection.
    """

    atom_serial = atom.record.atom_serial
    component_name = atom.record.residue_name.strip().upper()

    if atom_serial is not None and atom_serial in options.remove_atom_serials:
        return False

    if atom_serial is not None and atom_serial in options.add_atom_serials:
        return True

    if component_name in remove_component_names:
        return False

    if component_name in add_component_names:
        return True

    return is_default_bdamage_selection(atom, options)


def is_default_bdamage_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
) -> bool:
    """
    Return True if an atom should be selected by default for BDamage.
    """

    if atom.is_protein and (
        not atom.is_hetatm or options.include_protein_like_hetatm_in_selection
    ):
        return True

    if options.include_hetatm_in_selection and atom.is_hetatm and not atom.is_solvent:
        return True

    if options.include_nucleic_acid_in_selection and atom.is_nucleic_acid:
        return True

    return False


def append_selected_atom(
    atom: PreparedAtom,
    selected_atoms: list[PreparedAtom],
    selected_keys: set[int],
) -> None:
    """
    Append an atom to the selected list if it is not already present.
    """

    key = prepared_atom_key(atom)

    if key in selected_keys:
        return

    selected_atoms.append(atom)
    selected_keys.add(key)


def normalize_component_name_set(component_names: frozenset[str]) -> frozenset[str]:
    """
    Normalize component names to uppercase.
    """

    return frozenset(component_name.strip().upper() for component_name in component_names)


def prepared_atom_key(atom: PreparedAtom) -> int:
    """
    Return a stable key representing a prepared atom.
    """

    return atom.record.source_atom_index
