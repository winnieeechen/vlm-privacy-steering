#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=32)
    args = parser.parse_args()

    delegate = ROOT / "scripts" / "00_common" / "sweep_conditional_cached.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        "outputs/02_formal_full1200/00_base/val/base_qwen_vl_val_238.csv",
        "--steered-csv",
        (
            "outputs/06_cats_pca_behavior_vectors/02_over/val/"
            f"steered_qwen_vl_val_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--score-csv",
        "outputs/02_formal_full1200/02_over/val/condition_scores_val_layer32.csv",
        "--output-csv",
        (
            "outputs/06_cats_pca_behavior_vectors/02_over/val/"
            f"conditional_sweep_cats_pca_val_layer{args.layer}_alpha{args.alpha}.csv"
        ),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
