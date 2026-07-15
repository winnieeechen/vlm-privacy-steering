#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def split_n(split):
    return 238 if split == "val" else 243


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--vector-method")
    args = parser.parse_args()

    n = split_n(args.split)
    delegate = ROOT / "scripts" / "00_common" / "run_under_steering_general.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--under-vector",
        "outputs/11_pairwise_boundary_vectors/03_under/vectors/under_behavior_vectors_pairwise_axis.pt",
        "--output-csv",
        (
            f"outputs/11_pairwise_boundary_vectors/03_under/{args.split}/"
            f"steered_under_qwen_vl_{args.split}_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--layer",
        str(args.layer),
        "--alpha",
        str(args.alpha),
    ]
    if args.vector_method:
        cmd.extend(["--vector-method", args.vector_method])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

