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
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    n = split_n(args.split)
    delegate = ROOT / "scripts" / "00_common" / "run_under_steering_general.py"
    cmd = [
        sys.executable,
        str(delegate),
        "--base-csv",
        f"outputs/02_formal_full1200/00_base/{args.split}/base_qwen_vl_{args.split}_{n}.csv",
        "--under-vector",
        "outputs/09_severity_weighted_behavior_vectors/03_under/vectors/under_behavior_vectors_severity_weighted_full1200.pt",
        "--output-csv",
        (
            f"outputs/09_severity_weighted_behavior_vectors/03_under/{args.split}/"
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
