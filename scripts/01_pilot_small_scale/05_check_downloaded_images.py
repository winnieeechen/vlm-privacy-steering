#!/usr/bin/env python3
import csv
from pathlib import Path
from PIL import Image

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
CSV_PATH = ROOT / "data" / "processed" / "geoprivacy_downloaded_subset.csv"

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Rows:", len(rows))

ok = 0
bad = 0

for r in rows:
    path = Path(r["image_path"])
    try:
        img = Image.open(path)
        img.verify()
        ok += 1
    except Exception as e:
        bad += 1
        print("BAD:", path, e)

print("Readable images:", ok)
print("Bad images:", bad)

print("\nFirst 5 image sizes:")
for r in rows[:5]:
    path = Path(r["image_path"])
    img = Image.open(path)
    print(path.name, img.size, "label=" + r["true_label"])
