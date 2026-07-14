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
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    n = 238 if args.split == "val" else 243
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "00_common" / "run_under_steering_general.py"),
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--under-vector",
        "outputs/10b_balanced_cats_pc1_vectors/03_under/vectors/under_behavior_vectors_balanced_cats_pc1_full1200.pt",
        "--output-csv",
        (
            f"outputs/10b_balanced_cats_pc1_vectors/03_under/{args.split}/"
            f"steered_under_qwen_vl_{args.split}_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--layer",
        str(args.layer),
        "--alpha",
        str(args.alpha),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
