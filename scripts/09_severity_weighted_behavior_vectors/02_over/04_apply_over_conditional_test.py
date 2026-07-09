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
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    delegate = ROOT / "scripts" / "04_H2_hybrid_conditional_pca" / "00_common" / "apply_hybrid_conditional_test.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv",
        "--steered-csv",
        (
            "outputs/09_severity_weighted_behavior_vectors/02_over/test/"
            f"steered_qwen_vl_test_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--score-csv",
        "outputs/02_formal_full1200/02_over/test/condition_scores_test_layer32.csv",
        "--output-csv",
        (
            "outputs/09_severity_weighted_behavior_vectors/02_over/test/"
            f"conditional_severity_weighted_test_layer{args.layer}_alpha{args.alpha}_thr{args.threshold}.csv"
        ),
        "--threshold",
        str(args.threshold),
        "--score-key",
        "condition_score",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
