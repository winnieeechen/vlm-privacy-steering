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


def split_n(split):
    return 238 if split == "val" else 243


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha-over", type=float, default=0.5)
    parser.add_argument("--alpha-under", type=float, default=0.5)
    parser.add_argument("--over-threshold", type=float, default=-0.03)
    parser.add_argument("--under-threshold", type=float, default=0.04)
    args = parser.parse_args()

    n = split_n(args.split)
    delegate = ROOT / "scripts" / "04_low_rank_discriminant_vectors" / "05_dual_additive" / "03_run_dual_additive_steering.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--split",
        args.split,
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--over-behavior-vector",
        "outputs/08_weighted_mean_behavior_vectors/02_over/vectors/behavior_vectors_weighted_mean_full1200.pt",
        "--under-behavior-vector",
        "outputs/08_weighted_mean_behavior_vectors/03_under/vectors/under_behavior_vectors_weighted_mean_full1200.pt",
        "--over-score-csv",
        f"outputs/02_formal_full1200/02_over/{args.split}/condition_scores_{args.split}_layer32.csv",
        "--under-score-csv",
        f"outputs/02_formal_full1200/03_under/{args.split}/under_condition_scores_{args.split}_layer32.csv",
        "--output-csv",
        (
            f"outputs/08_weighted_mean_behavior_vectors/05_dual_additive/{args.split}/"
            f"dual_additive_weighted_mean_{args.split}_layer{args.layer}_"
            f"alpha{args.alpha_over}_over{args.over_threshold}_under{args.under_threshold}.csv"
        ),
        "--layer",
        str(args.layer),
        "--alpha-over",
        str(args.alpha_over),
        "--alpha-under",
        str(args.alpha_under),
        "--over-threshold",
        str(args.over_threshold),
        "--under-threshold",
        str(args.under_threshold),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
