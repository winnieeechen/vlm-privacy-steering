#!/usr/bin/env python3
import csv
from collections import Counter
from pathlib import Path

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
IN_CSV = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"
OUT_CSV = ROOT / "data" / "processed" / "geoprivacy_downloaded_subset.csv"

rows = []

with open(IN_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        if row["image_exists"] == "True":
            rows.append(row)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT_CSV}")
print(f"Downloaded samples: {len(rows)}")
print("Label counts:", Counter(r["true_label"] for r in rows))
print("Split counts:", Counter(r["split"] for r in rows))
print("Source counts:", Counter(r["image_source"] for r in rows))

print("\nFirst 5:")
for r in rows[:5]:
    print(
        r["full_id"],
        "label=" + r["true_label"],
        "split=" + r["split"],
        "path=" + r["image_path"],
    )
