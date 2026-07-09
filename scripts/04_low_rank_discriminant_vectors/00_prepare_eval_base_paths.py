#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()

DEFAULT_INPUTS = {
    "val": ROOT / "outputs" / "02_formal_full1200" / "00_base" / "val" / "base_qwen_vl_val_238.csv",
    "test": ROOT / "outputs" / "02_formal_full1200" / "00_base" / "test" / "base_qwen_vl_test_243.csv",
}

DEFAULT_OUTPUTS = {
    "val": ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "00_base" / "val" / "base_qwen_vl_val_238.csv",
    "test": ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "00_base" / "test" / "base_qwen_vl_test_243.csv",
}


def normalize_image_path(row):
    image_name = row.get("image_name") or f"{row['image_stem']}.jpg"
    candidates = [
        ROOT / "data" / "02_full1200" / "images" / image_name,
        ROOT / "data" / "images_full1200" / image_name,
        Path(row["image_path"]),
    ]
    for candidate in candidates:
        if candidate.exists():
            rr = dict(row)
            rr["image_path"] = str(candidate)
            rr["image_exists"] = "True"
            rr["image_readable"] = "True"
            return rr
    raise FileNotFoundError(f"Cannot find image for {row.get('full_id')}: {image_name}")


def normalize_csv(input_csv, output_csv):
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    normalized = [normalize_image_path(row) for row in rows]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)

    print("Input:", input_csv)
    print("Output:", output_csv)
    print("Rows:", len(normalized))
    print("First image_path:", normalized[0]["image_path"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        normalize_csv(DEFAULT_INPUTS[split], DEFAULT_OUTPUTS[split])
        print()


if __name__ == "__main__":
    main()
