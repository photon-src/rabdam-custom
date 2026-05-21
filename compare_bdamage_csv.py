import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final BDamage values from RABDAM 2 and RABDAM 3 CSV files."
    )
    parser.add_argument("rabdam2_csv", help="RABDAM 2 *_BDamage.csv file.")
    parser.add_argument("rabdam3_csv", help="RABDAM 3 rabdam3_BDamage.csv file.")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of largest absolute differences to print.",
    )
    return parser.parse_args()


def read_bdamage_csv(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text().splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("REC,ATMNUM,ATMNAME")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not find BDamage CSV header in {path}")

    rows = csv.DictReader(lines[header_index:])
    by_atom_number: dict[str, dict[str, str]] = {}
    for row in rows:
        atom_number = row["ATMNUM"].strip()
        if atom_number in by_atom_number:
            raise ValueError(
                f"Duplicate ATMNUM {atom_number!r} in {path}; comparison needs a unique atom serial."
            )
        by_atom_number[atom_number] = row

    return by_atom_number


def as_float(row: dict[str, str], column: str) -> float:
    return float(row[column])


def main() -> None:
    args = parse_args()
    rabdam2_rows = read_bdamage_csv(Path(args.rabdam2_csv))
    rabdam3_rows = read_bdamage_csv(Path(args.rabdam3_csv))

    rabdam2_atoms = set(rabdam2_rows)
    rabdam3_atoms = set(rabdam3_rows)
    common_atoms = sorted(rabdam2_atoms & rabdam3_atoms, key=int)
    only_rabdam2 = sorted(rabdam2_atoms - rabdam3_atoms, key=int)
    only_rabdam3 = sorted(rabdam3_atoms - rabdam2_atoms, key=int)

    differences = []
    pd_mismatches = 0
    for atom_number in common_atoms:
        row2 = rabdam2_rows[atom_number]
        row3 = rabdam3_rows[atom_number]
        bdamage2 = as_float(row2, "BDAM")
        bdamage3 = as_float(row3, "BDAM")
        difference = bdamage3 - bdamage2
        differences.append(
            {
                "ATMNUM": atom_number,
                "BDAM_RABDAM2": bdamage2,
                "BDAM_RABDAM3": bdamage3,
                "DIFF": difference,
                "ABS_DIFF": abs(difference),
                "PD_RABDAM2": as_float(row2, "PD"),
                "PD_RABDAM3": as_float(row3, "PD"),
            }
        )
        if as_float(row2, "PD") != as_float(row3, "PD"):
            pd_mismatches += 1

    if not differences:
        raise ValueError("No atom serials overlap between the two CSV files.")

    abs_differences = [row["ABS_DIFF"] for row in differences]
    signed_differences = [row["DIFF"] for row in differences]
    mean_abs_difference = sum(abs_differences) / len(abs_differences)
    rms_difference = math.sqrt(
        sum(difference * difference for difference in signed_differences)
        / len(signed_differences)
    )
    largest = sorted(differences, key=lambda row: row["ABS_DIFF"], reverse=True)[
        : args.top
    ]

    print(f"RABDAM 2 atoms: {len(rabdam2_rows)}")
    print(f"RABDAM 3 atoms: {len(rabdam3_rows)}")
    print(f"Matched atoms: {len(common_atoms)}")
    print(f"Only in RABDAM 2: {len(only_rabdam2)}")
    print(f"Only in RABDAM 3: {len(only_rabdam3)}")
    print(f"Packing-density mismatches among matched atoms: {pd_mismatches}")
    print(f"Max absolute BDamage difference: {max(abs_differences):.12g}")
    print(f"Mean absolute BDamage difference: {mean_abs_difference:.12g}")
    print(f"RMS BDamage difference: {rms_difference:.12g}")

    print("\nLargest absolute BDamage differences:")
    print("ATMNUM,BDAM_RABDAM2,BDAM_RABDAM3,DIFF,PD_RABDAM2,PD_RABDAM3")
    for row in largest:
        print(
            f"{row['ATMNUM']},"
            f"{row['BDAM_RABDAM2']:.12g},"
            f"{row['BDAM_RABDAM3']:.12g},"
            f"{row['DIFF']:.12g},"
            f"{row['PD_RABDAM2']:.12g},"
            f"{row['PD_RABDAM3']:.12g}"
        )


if __name__ == "__main__":
    main()
