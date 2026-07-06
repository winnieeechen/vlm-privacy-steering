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
CSV_PATH = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("Loaded rows:", len(rows))

print("\nLabel counts:")
print(Counter(r["true_label"] for r in rows))

print("\nSplit counts:")
print(Counter(r["split"] for r in rows))

print("\nPrivacy-sensitive counts:")
print(Counter(r["privacy_sensitive"] for r in rows))

print("\nFirst example:")
r = rows[0]
for k in ["full_id", "true_label", "privacy_sensitive", "split", "image_path", "image_exists"]:
    print(k + ":", r[k])
