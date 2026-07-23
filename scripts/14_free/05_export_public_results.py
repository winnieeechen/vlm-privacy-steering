#!/usr/bin/env python3
"""Export portable, complete method-14 CSVs for version control."""

import argparse
import csv
from pathlib import Path

import method14_common as common


ROOT = common.ROOT
DEFAULT_FILES = {
    "val_results.csv": (
        "outputs/14_free/val/"
        "14_free_val_a1.0-1.5_c3.0-4.0.csv"
    ),
    "test_results.csv": (
        "outputs/14_free/test/"
        "14_free_test_full_a1.0-1.5_c3.0-4.0.csv"
    ),
    "validation_confidence_sweep.csv": (
        "outputs/14_free/val_confidence_sweep/summary.csv"
    ),
}


def portable(value):
    prefix = str(ROOT) + "/"
    return value.replace(prefix, "") if isinstance(value, str) else value


def export_csv(source, destination):
    rows = common.read_rows(source)
    if not rows:
        raise RuntimeError(f"Cannot export empty CSV: {source}")
    cleaned = [
        {key: portable(value) for key, value in row.items() if key != "alpha_c"}
        for row in rows
    ]
    common.write_rows(destination, cleaned)
    print(f"Exported {len(cleaned)} rows: {destination}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/14_free")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = common.project_path(args.output_dir)
    for filename, source in DEFAULT_FILES.items():
        export_csv(source, output_dir / filename)


if __name__ == "__main__":
    main()
