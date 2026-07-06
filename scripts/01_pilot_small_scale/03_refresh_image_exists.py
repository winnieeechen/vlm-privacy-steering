#!/usr/bin/env python3
import csv
from pathlib import Path

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
CSV_PATH = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"
TMP_PATH = ROOT / "data" / "processed" / "geoprivacy_q7_labels.tmp.csv"

rows = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        image_path = Path(row["image_path"])
        row["image_exists"] = str(image_path.exists())
        rows.append(row)

with open(TMP_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

TMP_PATH.replace(CSV_PATH)

n_exists = sum(r["image_exists"] == "True" for r in rows)
print(f"Updated: {CSV_PATH}")
print(f"Images found: {n_exists}/{len(rows)}")

print("\nDownloaded samples:")
shown = 0
for r in rows:
    if r["image_exists"] == "True":
        print(
            r["full_id"],
            "label=" + r["true_label"],
            "split=" + r["split"],
            "source=" + r["image_source"],
            "path=" + r["image_path"],
        )
        shown += 1
        if shown >= 10:
            break
