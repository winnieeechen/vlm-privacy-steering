#!/usr/bin/env python3
"""
Download images needed by our processed GeoPrivacy label table.

This script is written for this project. It reads:
  data/processed/geoprivacy_q7_labels.csv

and saves images to:
  data/images/

Currently supported sources:
  - Flickr-yfcc_openai_train
  - Flickr-yfcc_openai_valid

Other sources will be added later.
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
DEFAULT_LABEL_CSV = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "images"


SUPPORTED_HF_SOURCES = {
    "Flickr-yfcc_openai_train": "train",
    "Flickr-yfcc_openai_valid": "validation",
}


def load_needed_rows(label_csv: Path, sources: list[str] | None, max_images: int | None):
    rows = []

    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["image_source"]
            if sources is not None and source not in sources:
                continue
            if source not in SUPPORTED_HF_SOURCES:
                continue
            rows.append(row)
            if max_images is not None and len(rows) >= max_images:
                break

    return rows


def save_image(out_path: Path, img_obj):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img_obj, bytes):
        with open(out_path, "wb") as f:
            f.write(img_obj)
        return

    if isinstance(img_obj, dict) and "bytes" in img_obj:
        with open(out_path, "wb") as f:
            f.write(img_obj["bytes"])
        return

    if isinstance(img_obj, list):
        with open(out_path, "wb") as f:
            f.write(bytes(img_obj))
        return

    if isinstance(img_obj, Image.Image):
        img_obj.convert("RGB").save(out_path)
        return

    # Sometimes datasets returns an encoded image-like object.
    try:
        Image.open(BytesIO(img_obj)).convert("RGB").save(out_path)
        return
    except Exception as e:
        raise TypeError(f"Unsupported image object type: {type(img_obj)}") from e


def download_hf_subset(rows, output_dir: Path, skip_existing: bool, scan_limit: int | None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets", file=sys.stderr)
        return 0

    by_split = {}
    for r in rows:
        split = SUPPORTED_HF_SOURCES[r["image_source"]]
        by_split.setdefault(split, []).append(r)

    total_saved = 0

    print("Loading dalle-mini/YFCC100M_OpenAI_subset with streaming=True ...", flush=True)
    dataset = load_dataset(
        "dalle-mini/YFCC100M_OpenAI_subset",
        streaming=True,
        trust_remote_code=True,
    )
    print("Dataset stream ready.", flush=True)

    for split, split_rows in by_split.items():
        needed_ids = {r["numeric_id"] for r in split_rows}
        print(f"[{split}] Need {len(needed_ids)} images.", flush=True)

        saved = 0
        scanned = 0
        futures = []

        def _write(out_path, img_obj):
            save_image(out_path, img_obj)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for item in dataset[split]:
                scanned += 1

                if scanned % 10000 == 0:
                    print(f"  scanned {scanned}, saved {saved}/{len(needed_ids)}", flush=True)

                if scan_limit is not None and scanned >= scan_limit:
                    print(f"  reached scan limit {scan_limit}; stop scanning this split.", flush=True)
                    break

                photoid = str(item.get("photoid", ""))
                if photoid not in needed_ids:
                    continue

                out_path = output_dir / f"{photoid}.jpg"

                if skip_existing and out_path.exists():
                    saved += 1
                else:
                    img_obj = item.get("img")
                    if img_obj is not None:
                        futures.append(executor.submit(_write, out_path, img_obj))
                        saved += 1

                if saved >= len(needed_ids):
                    break

            for future in as_completed(futures):
                future.result()

        print(f"[{split}] Saved {saved}/{len(needed_ids)} images.", flush=True)
        total_saved += saved

    return total_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["Flickr-yfcc_openai_train", "Flickr-yfcc_openai_valid"],
        help="Image sources to download.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=None,
                        help="Maximum number of HF dataset items to scan per split.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = load_needed_rows(args.label_csv, args.sources, args.max_images)

    print(f"Label CSV: {args.label_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Rows to download: {len(rows)}")

    if len(rows) == 0:
        print("No supported rows found.")
        return

    total_saved = download_hf_subset(rows, args.output_dir, args.skip_existing, args.scan_limit)
    print(f"Done. Total saved/found: {total_saved}")


if __name__ == "__main__":
    main()
