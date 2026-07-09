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
    parser.add_argument("--threshold-mode", choices=["quantile", "fixed"], default="quantile")
    parser.add_argument("--min-gate-rate", type=float, default=0.2)
    parser.add_argument("--max-gate-rate", type=float, default=0.8)
    args = parser.parse_args()

    delegate = ROOT / "scripts" / "04_H2_hybrid_conditional_pca" / "00_common" / "sweep_hybrid_conditional_cached.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        "outputs/02_formal_full1200/00_base/val/base_qwen_vl_val_238.csv",
        "--steered-csv",
        (
            "outputs/08_weighted_mean_behavior_vectors/02_over/val/"
            f"steered_qwen_vl_val_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--score-csv",
        "outputs/02_formal_full1200/02_over/val/condition_scores_val_layer32.csv",
        "--output-csv",
        (
            "outputs/08_weighted_mean_behavior_vectors/02_over/val/"
            f"conditional_quantile_sweep_val_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--score-key",
        "condition_score",
        "--threshold-mode",
        args.threshold_mode,
        "--min-gate-rate",
        str(args.min_gate_rate),
        "--max-gate-rate",
        str(args.max_gate_rate),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
