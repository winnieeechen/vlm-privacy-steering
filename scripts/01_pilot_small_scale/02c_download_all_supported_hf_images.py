#!/usr/bin/env python3
import csv
from io import BytesIO
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from PIL import Image


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
LABEL_CSV = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"
OUT_DIR = ROOT / "data" / "images"

SUPPORTED_HF_SOURCES = {
    "Flickr-yfcc_openai_train": "train",
    "Flickr-yfcc_openai_valid": "validation",
}


def save_image(out_path: Path, img_obj):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img_obj, Image.Image):
        img_obj.convert("RGB").save(out_path)
        return

    if isinstance(img_obj, bytes):
        Image.open(BytesIO(img_obj)).convert("RGB").save(out_path)
        return

    if isinstance(img_obj, dict) and "bytes" in img_obj:
        Image.open(BytesIO(img_obj["bytes"])).convert("RGB").save(out_path)
        return

    raise TypeError(f"Unsupported image type: {type(img_obj)}")


def main():
    rows = []
    with open(LABEL_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image_source"] in SUPPORTED_HF_SOURCES:
                rows.append(r)

    print("Supported HF rows:", len(rows))
    print("Label counts:", Counter(r["true_label"] for r in rows))

    by_split = {"train": {}, "validation": {}}
    for r in rows:
        hf_split = SUPPORTED_HF_SOURCES[r["image_source"]]
        by_split[hf_split][r["numeric_id"]] = r

    total_targets = sum(len(v) for v in by_split.values())
    already = sum(Path(r["image_path"]).exists() for r in rows)

    print("Already downloaded:", already)
    print("Total targets:", total_targets)

    ds = load_dataset(
        "dalle-mini/YFCC100M_OpenAI_subset",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    skipped_existing = 0
    seen_targets = set()

    for split in ["train", "validation"]:
        id_to_row = by_split[split]
        remaining_ids = set(id_to_row.keys())

        print(f"\nScanning split: {split}")
        print("Targets in this split:", len(remaining_ids))

        scanned = 0

        for item in ds[split]:
            scanned += 1

            if scanned % 10000 == 0:
                print(
                    f"  scanned={scanned}, saved={saved}, "
                    f"skipped_existing={skipped_existing}, "
                    f"found_targets={len(seen_targets)}/{total_targets}",
                    flush=True,
                )

            photoid = str(item.get("photoid", ""))

            if photoid not in remaining_ids:
                continue

            row = id_to_row[photoid]
            out_path = Path(row["image_path"])

            seen_targets.add(photoid)
            remaining_ids.remove(photoid)

            if out_path.exists():
                skipped_existing += 1
            else:
                img_obj = item.get("img")
                if img_obj is None:
                    print("  missing image object:", photoid)
                    continue

                save_image(out_path, img_obj)
                saved += 1
                print(f"  saved {photoid} label={row['true_label']} split={split}")

            if not remaining_ids:
                print(f"All targets found for split {split}.")
                break

        print(f"Finished split {split}. Remaining not found:", len(remaining_ids))

    print("\nDone.")
    print("Newly saved:", saved)
    print("Skipped existing:", skipped_existing)
    print("Found targets:", len(seen_targets), "/", total_targets)


if __name__ == "__main__":
    main()
