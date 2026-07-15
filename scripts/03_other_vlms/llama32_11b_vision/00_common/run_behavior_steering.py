#!/usr/bin/env python3
import argparse
import sys
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
        description="Run unconditional Llama-3.2 Vision behavior steering."
    )
    parser.add_argument("--side", choices=["over", "under"], required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--behavior-vector", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--vector-method")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def column_names(side):
    if side == "over":
        return "steered_answer", "steered_pred_label", "steered_case_type"
    return "under_answer", "under_pred_label", "under_case_type"


def main():
    args = parse_args()
    rows = read_csv(args.base_csv)
    if args.max_rows:
        rows = rows[:args.max_rows]
    output = project_path(args.output_csv)
    existing = read_csv(output) if args.resume and output.exists() else []
    done_ids = {row["full_id"] for row in existing}

    payload = torch.load(project_path(args.behavior_vector), map_location="cpu")
    vector = vector_from_payload(
        payload, args.side, args.layer, args.vector_method
    )
    print("Rows:", len(rows))
    print("Already complete:", len(done_ids))
    print("Side:", args.side)
    print("Layer:", args.layer)
    print("Alpha:", args.alpha)
    print("Vector method:", args.vector_method or payload.get("selected_method", "default"))
    print("Vector norm:", float(vector.norm()))
    print("Output:", output)

    model, processor = load_model_and_processor()
    state = {"vector": args.alpha * vector}
    layers = language_layers(model)
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"Layer {args.layer} out of range 0..{len(layers) - 1}")
    handle = layers[args.layer].register_forward_hook(make_dynamic_hook(state))

    answer_key, pred_key, case_key = column_names(args.side)
    results = existing
    try:
        for index, row in enumerate(rows, 1):
            if row["full_id"] in done_ids:
                print(f"[{index}/{len(rows)}] skip {row['full_id']}")
                continue
            answer, prediction = run_generation(model, processor, row)
            case = make_case(row["true_label"], prediction)
            result = dict(row)
            result[answer_key] = answer
            result[pred_key] = prediction
            result[case_key] = case
            result["steer_side"] = args.side
            result["steer_layer"] = args.layer
            result["steer_alpha"] = args.alpha
            result["vector_method"] = (
                args.vector_method or payload.get("selected_method", "payload_default")
            )
            results.append(result)
            write_csv(output, results)
            print(
                f"[{index}/{len(rows)}] {row['full_id']} true={row['true_label']} "
                f"base={row.get('pred_label')}->{prediction} "
                f"{row.get('case_type')}->{case}"
            )
    finally:
        handle.remove()

    print("\nBase summary:")
    summarize(results, "case_type")
    print(f"\n{args.side.title()} steering summary:")
    summarize(results, case_key)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
