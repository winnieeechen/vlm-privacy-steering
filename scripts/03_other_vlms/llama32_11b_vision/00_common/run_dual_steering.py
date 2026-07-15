#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from llama32_vision_common import (  # noqa: E402
    language_layers,
    load_model_and_processor,
    make_case,
    make_dynamic_hook,
    project_path,
    read_csv,
    run_generation,
    summarize,
    vector_from_payload,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unconditional or condition-gated dual Llama steering."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--over-behavior-vector", required=True)
    parser.add_argument("--under-behavior-vector", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--over-layer", type=int, required=True)
    parser.add_argument("--under-layer", type=int, required=True)
    parser.add_argument("--alpha-over", type=float, required=True)
    parser.add_argument("--alpha-under", type=float, required=True)
    parser.add_argument("--over-vector-method")
    parser.add_argument("--under-vector-method")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--over-score-csv")
    parser.add_argument("--under-score-csv")
    parser.add_argument("--over-threshold", type=float)
    parser.add_argument("--under-threshold", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def score_value(row, preferred, fallback):
    if preferred in row:
        return float(row[preferred])
    return float(row[fallback])


def validate_conditional_args(args):
    required = {
        "--over-score-csv": args.over_score_csv,
        "--under-score-csv": args.under_score_csv,
        "--over-threshold": args.over_threshold,
        "--under-threshold": args.under_threshold,
    }
    missing = [name for name, value in required.items() if value is None]
    if args.conditional and missing:
        raise ValueError("Conditional dual steering requires " + ", ".join(missing))


def main():
    args = parse_args()
    validate_conditional_args(args)
    rows = read_csv(args.base_csv)
    if args.max_rows:
        rows = rows[:args.max_rows]
    output = project_path(args.output_csv)
    existing = read_csv(output) if args.resume and output.exists() else []
    done_ids = {row["full_id"] for row in existing}

    over_payload = torch.load(
        project_path(args.over_behavior_vector), map_location="cpu"
    )
    under_payload = torch.load(
        project_path(args.under_behavior_vector), map_location="cpu"
    )
    over_vector = vector_from_payload(
        over_payload, "over", args.over_layer, args.over_vector_method
    )
    under_vector = vector_from_payload(
        under_payload, "under", args.under_layer, args.under_vector_method
    )

    over_scores = {}
    under_scores = {}
    if args.conditional:
        over_scores = {row["full_id"]: row for row in read_csv(args.over_score_csv)}
        under_scores = {row["full_id"]: row for row in read_csv(args.under_score_csv)}

    print("Rows:", len(rows))
    print("Conditional:", args.conditional)
    print("Over layer/alpha:", args.over_layer, args.alpha_over)
    print("Under layer/alpha:", args.under_layer, args.alpha_under)
    print("Over vector norm:", float(over_vector.norm()))
    print("Under vector norm:", float(under_vector.norm()))
    print("Output:", output)

    model, processor = load_model_and_processor()
    layers = language_layers(model)
    for layer in (args.over_layer, args.under_layer):
        if not 0 <= layer < len(layers):
            raise ValueError(f"Layer {layer} out of range 0..{len(layers) - 1}")

    handles = []
    if args.over_layer == args.under_layer:
        combined_state = {"vector": torch.zeros_like(over_vector)}
        handles.append(
            layers[args.over_layer].register_forward_hook(
                make_dynamic_hook(combined_state)
            )
        )
        over_state = under_state = None
    else:
        combined_state = None
        over_state = {"vector": torch.zeros_like(over_vector)}
        under_state = {"vector": torch.zeros_like(under_vector)}
        handles.append(
            layers[args.over_layer].register_forward_hook(
                make_dynamic_hook(over_state)
            )
        )
        handles.append(
            layers[args.under_layer].register_forward_hook(
                make_dynamic_hook(under_state)
            )
        )

    results = existing
    gate_counts = Counter()
    try:
        for index, row in enumerate(rows, 1):
            if row["full_id"] in done_ids:
                print(f"[{index}/{len(rows)}] skip {row['full_id']}")
                continue
            if args.conditional:
                over_score = score_value(
                    over_scores[row["full_id"]],
                    "condition_score",
                    "over_condition_score",
                )
                under_score = score_value(
                    under_scores[row["full_id"]],
                    "under_condition_score",
                    "condition_score",
                )
                over_gate = over_score >= args.over_threshold
                under_gate = under_score >= args.under_threshold
            else:
                over_score = ""
                under_score = ""
                over_gate = True
                under_gate = True
            gate_counts[(over_gate, under_gate)] += 1

            over_steer = args.alpha_over * over_vector if over_gate else torch.zeros_like(over_vector)
            under_steer = args.alpha_under * under_vector if under_gate else torch.zeros_like(under_vector)
            if combined_state is not None:
                combined_state["vector"] = over_steer + under_steer
            else:
                over_state["vector"] = over_steer
                under_state["vector"] = under_steer

            answer, prediction = run_generation(model, processor, row)
            case = make_case(row["true_label"], prediction)
            result = dict(row)
            result["dual_additive_answer"] = answer
            result["dual_additive_pred_label"] = prediction
            result["dual_additive_case_type"] = case
            result["over_layer"] = args.over_layer
            result["under_layer"] = args.under_layer
            result["alpha_over"] = args.alpha_over
            result["alpha_under"] = args.alpha_under
            result["over_condition_score"] = over_score
            result["under_condition_score"] = under_score
            result["over_threshold"] = args.over_threshold if args.conditional else ""
            result["under_threshold"] = args.under_threshold if args.conditional else ""
            result["over_gate"] = over_gate
            result["under_gate"] = under_gate
            results.append(result)
            write_csv(output, results)
            print(
                f"[{index}/{len(rows)}] {row['full_id']} true={row['true_label']} "
                f"g_over={int(over_gate)} g_under={int(under_gate)} "
                f"base={row.get('pred_label')}->{prediction} "
                f"{row.get('case_type')}->{case}"
            )
    finally:
        for handle in handles:
            handle.remove()

    print("\nGate counts:", gate_counts)
    print("\nBase summary:")
    summarize(results, "case_type")
    print("\nDual steering summary:")
    summarize(results, "dual_additive_case_type")
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
