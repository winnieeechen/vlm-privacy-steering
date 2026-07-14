#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--alpha-over", type=float, required=True)
    parser.add_argument("--alpha-under", type=float, required=True)
    parser.add_argument("--over-threshold", type=float, required=True)
    parser.add_argument("--under-threshold", type=float, required=True)
    args = parser.parse_args()
    n = 238 if args.split == "val" else 243
    score_csv = (
        f"outputs/02_formal_full1200/05_dual_additive/optimized_layer{args.layer}/"
        f"{args.split}/condition_scores_{args.split}_layer{args.layer}.csv"
    )
    cmd = [
        sys.executable,
        str(
            ROOT
            / "scripts"
            / "04_low_rank_discriminant_vectors"
            / "05_dual_additive"
            / "03_run_dual_additive_steering.py"
        ),
        "--split",
        args.split,
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--over-behavior-vector",
        "outputs/10b_balanced_cats_pc1_vectors/02_over/vectors/behavior_vectors_balanced_cats_pc1_full1200.pt",
        "--under-behavior-vector",
        "outputs/10b_balanced_cats_pc1_vectors/03_under/vectors/under_behavior_vectors_balanced_cats_pc1_full1200.pt",
        "--over-score-csv",
        score_csv,
        "--under-score-csv",
        score_csv,
        "--output-csv",
        (
            f"outputs/10b_balanced_cats_pc1_vectors/05_dual_additive/{args.split}/"
            f"dual_conditional_balanced_cats_pc1_{args.split}_layer{args.layer}_"
            f"over{args.alpha_over}_under{args.alpha_under}_"
            f"overthr{args.over_threshold}_underthr{args.under_threshold}.csv"
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
