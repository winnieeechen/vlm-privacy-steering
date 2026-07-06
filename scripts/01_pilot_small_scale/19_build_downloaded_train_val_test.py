#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict
from pathlib import Path

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()

IN_CSV = ROOT / "data" / "processed" / "geoprivacy_downloaded_subset.csv"
OUT_DIR = ROOT / "data" / "processed"

OUT_FILES = {
    "train": OUT_DIR / "geoprivacy_downloaded_train.csv",
    "val": OUT_DIR / "geoprivacy_downloaded_val.csv",
    "test": OUT_DIR / "geoprivacy_downloaded_test.csv",
}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with open(IN_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("Total downloaded rows:", len(rows))
    print("Overall label counts:", Counter(r["true_label"] for r in rows))
    print("Overall split counts:", Counter(r["split"] for r in rows))

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    fieldnames = list(rows[0].keys())

    print("\nSplit details:")
    for split in ["train", "val", "test"]:
        split_rows = by_split[split]
        out_path = OUT_FILES[split]

        write_csv(out_path, split_rows, fieldnames)

        print(f"\n{split}:")
        print(" rows:", len(split_rows))
        print(" label counts:", Counter(r["true_label"] for r in split_rows))
        print(" saved:", out_path)


if __name__ == "__main__":
    main()
