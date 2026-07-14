#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    size = {"val": 238, "test": 243}[args.split]
    cmd = [
        sys.executable,
        str(ROOT / "scripts/03_other_vlms/qwen25_vl_7b_instruct/02_over/14_run_behavior_steering.py"),
        "--base-csv", f"outputs/03_other_vlms/qwen25_vl_7b_instruct/00_base/{args.split}/base_qwen25_vl_7b_{args.split}_{size}.csv",
        "--behavior-vector", "outputs/03_other_vlms/qwen25_vl_7b_instruct/05_balanced_cats_transition_vectors/02_over/vectors/behavior_vectors_balanced_cats_train_717.pt",
        "--output-csv", f"outputs/03_other_vlms/qwen25_vl_7b_instruct/05_balanced_cats_transition_vectors/02_over/{args.split}/steered_qwen25_vl_7b_{args.split}_layer{args.layer}_alpha{args.alpha}.csv",
        "--layer", str(args.layer), "--alpha", str(args.alpha),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
