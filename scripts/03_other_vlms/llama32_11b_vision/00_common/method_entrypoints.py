#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parent
MODEL_ROOT = "outputs/03_other_vlms/llama32_11b_vision"
SCRIPT_ROOT = "scripts/03_other_vlms/llama32_11b_vision"
MEAN_ROOT = f"{MODEL_ROOT}/02_mean_behavior_vectors"
SHARED_CACHE = f"{MODEL_ROOT}/00_shared/cache/train_answer_suffix_states_k8.pt"


def run(command):
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)


def extraction_entrypoint(method_dir, mode, side):
    vector_name = (
        "behavior_vectors_train_717.pt"
        if side == "over"
        else "under_behavior_vectors_train_717.pt"
    )
    reference = (
        f"{MEAN_ROOT}/02_over/vectors/behavior_vectors_train_717.pt"
        if side == "over"
        else f"{MEAN_ROOT}/03_under/vectors/under_behavior_vectors_train_717.pt"
    )
    reference_key = "behavior_vector" if side == "over" else "under_behavior_vector"
    command = [
        sys.executable,
        str(COMMON_DIR / "extract_transition_vectors.py"),
        "--mode", mode,
        "--side", side,
        "--base-csv", f"{MODEL_ROOT}/00_base/train/base_llama32_vision_train_717.csv",
        "--cache", SHARED_CACHE,
        "--reference-vector", reference,
        "--reference-vector-key", reference_key,
        "--output", f"{MODEL_ROOT}/{method_dir}/{side_dir(side)}/vectors/{vector_name}",
        *sys.argv[1:],
    ]
    run(command)


def select_pc1_entrypoint(side):
    vector_name = (
        "behavior_vectors_train_717.pt"
        if side == "over"
        else "under_behavior_vectors_train_717.pt"
    )
    command = [
        sys.executable,
        str(COMMON_DIR / "select_vector_method.py"),
        "--source",
        f"{MODEL_ROOT}/10_balanced_cats_transition_vectors/{side_dir(side)}/vectors/{vector_name}",
        "--output",
        f"{MODEL_ROOT}/10b_balanced_cats_pc1_vectors/{side_dir(side)}/vectors/{vector_name}",
        "--side", side,
        *sys.argv[1:],
    ]
    run(command)


def side_dir(side):
    return "02_over" if side == "over" else "03_under"


def steering_entrypoint(method_dir, side):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    args, extra = parser.parse_known_args()
    n = 238 if args.split == "val" else 243
    vector_name = (
        "behavior_vectors_train_717.pt"
        if side == "over"
        else "under_behavior_vectors_train_717.pt"
    )
    output_prefix = "steered" if side == "over" else "under_steered"
    command = [
        sys.executable,
        str(COMMON_DIR / "run_behavior_steering.py"),
        "--side", side,
        "--base-csv",
        f"{MODEL_ROOT}/00_base/{args.split}/base_llama32_vision_{args.split}_{n}.csv",
        "--behavior-vector",
        f"{MODEL_ROOT}/{method_dir}/{side_dir(side)}/vectors/{vector_name}",
        "--output-csv",
        (
            f"{MODEL_ROOT}/{method_dir}/{side_dir(side)}/{args.split}/"
            f"{output_prefix}_llama32_vision_{args.split}_layer{args.layer}_alpha{args.alpha}.csv"
        ),
        "--layer", str(args.layer),
        "--alpha", str(args.alpha),
        *extra,
    ]
    run(command)


def dual_entrypoint(method_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--over-layer", type=int, default=32)
    parser.add_argument("--under-layer", type=int, default=32)
    parser.add_argument("--alpha-over", type=float, default=1.0)
    parser.add_argument("--alpha-under", type=float, default=1.0)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--score-layer", type=int, default=32)
    parser.add_argument("--over-threshold", type=float)
    parser.add_argument("--under-threshold", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    n = 238 if args.split == "val" else 243
    mode = "conditional" if args.conditional else "unconditional"
    command = [
        sys.executable,
        str(COMMON_DIR / "run_dual_steering.py"),
        "--base-csv",
        f"{MODEL_ROOT}/00_base/{args.split}/base_llama32_vision_{args.split}_{n}.csv",
        "--over-behavior-vector",
        f"{MODEL_ROOT}/{method_dir}/02_over/vectors/behavior_vectors_train_717.pt",
        "--under-behavior-vector",
        f"{MODEL_ROOT}/{method_dir}/03_under/vectors/under_behavior_vectors_train_717.pt",
        "--output-csv",
        (
            f"{MODEL_ROOT}/{method_dir}/05_dual_additive/{args.split}/dual_{mode}_"
            f"llama32_vision_{args.split}_overL{args.over_layer}_underL{args.under_layer}_"
            f"overA{args.alpha_over}_underA{args.alpha_under}.csv"
        ),
        "--over-layer", str(args.over_layer),
        "--under-layer", str(args.under_layer),
        "--alpha-over", str(args.alpha_over),
        "--alpha-under", str(args.alpha_under),
    ]
    if args.conditional:
        if args.over_threshold is None or args.under_threshold is None:
            parser.error("--conditional requires both thresholds selected on val")
        command.extend([
            "--conditional",
            "--over-score-csv",
            f"{MEAN_ROOT}/02_over/{args.split}/condition_scores_{args.split}_layer{args.score_layer}.csv",
            "--under-score-csv",
            f"{MEAN_ROOT}/03_under/{args.split}/under_condition_scores_{args.split}_layer{args.score_layer}.csv",
            "--over-threshold", str(args.over_threshold),
            "--under-threshold", str(args.under_threshold),
        ])
    if args.resume:
        command.append("--resume")
    if args.max_rows:
        command.extend(["--max-rows", str(args.max_rows)])
    run(command)
