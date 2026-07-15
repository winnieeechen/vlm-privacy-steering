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
    args = parser.parse_args()

    n = split_n(args.split)
    delegate = ROOT / "scripts" / "00_common" / "score_condition_signal.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--condition-vector",
        "outputs/11_pairwise_boundary_vectors/02_over/vectors/condition_vectors_pairwise_axis_over.pt",
        "--output-csv",
        (
            f"outputs/11_pairwise_boundary_vectors/02_over/{args.split}/"
            f"condition_scores_{args.split}_layer{args.layer}.csv"
        ),
        "--layer",
        str(args.layer),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

