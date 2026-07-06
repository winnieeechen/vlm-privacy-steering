#!/usr/bin/env python3
import argparse
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


def load_rows():
    rows = []
    with open(LABEL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["image_source"] in SUPPORTED_HF_SOURCES:
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-b", type=int, default=10)
    parser.add_argument("--target-c", type=int, default=10)
    parser.add_argument("--scan-limit", type=int, default=300000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = load_rows()

    # Map split -> numeric_id -> metadata row
    by_split = {"train": {}, "validation": {}}
    for r in rows:
        split = SUPPORTED_HF_SOURCES[r["image_source"]]
        by_split[split][r["numeric_id"]] = r

    print("Supported HF rows:", len(rows))
    print("Label counts in supported HF rows:", Counter(r["true_label"] for r in rows))
    print("Output dir:", OUT_DIR)

    existing_counts = Counter()
    for r in rows:
        path = Path(r["image_path"])
        if path.exists():
            existing_counts[r["true_label"]] += 1

    print("Existing downloaded label counts:", existing_counts)

    targets = {
        "B": args.target_b,
        "C": args.target_c,
    }

    needed = {}
    for label, target in targets.items():
        already = existing_counts[label]
        needed[label] = max(0, target - already)

    print("Need to download:", needed)

    if all(v == 0 for v in needed.values()):
        print("Nothing to download.")
        return

    ds = load_dataset(
        "dalle-mini/YFCC100M_OpenAI_subset",
        streaming=True,
        trust_remote_code=True,
    )

    saved_counts = Counter()

    for split in ["train", "validation"]:
        if all(saved_counts[label] >= needed[label] for label in needed):
            break

        print(f"\nScanning HF split: {split}")
        scanned = 0

        id_to_row = by_split[split]

        for item in ds[split]:
            scanned += 1

            if scanned % 10000 == 0:
                print(
                    f"  scanned {scanned}, saved B={saved_counts['B']}/{needed['B']}, "
                    f"C={saved_counts['C']}/{needed['C']}",
                    flush=True,
                )

            if scanned >= args.scan_limit:
                print(f"  reached scan limit {args.scan_limit}; stop {split}.")
                break

            photoid = str(item.get("photoid", ""))
            if photoid not in id_to_row:
                continue

            row = id_to_row[photoid]
            label = row["true_label"]

            if label not in needed:
                continue

            if saved_counts[label] >= needed[label]:
                continue

            out_path = Path(row["image_path"])

            if args.skip_existing and out_path.exists():
                saved_counts[label] += 1
                continue

            img_obj = item.get("img")
            if img_obj is None:
                continue

            save_image(out_path, img_obj)
            saved_counts[label] += 1
            print(f"  saved {photoid} label={label} split={split}")

            if all(saved_counts[label] >= needed[label] for label in needed):
                break

    print("\nSaved counts this run:", saved_counts)
    print("Done.")


if __name__ == "__main__":
    main()
