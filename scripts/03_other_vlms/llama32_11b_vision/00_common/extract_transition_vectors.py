#!/usr/bin/env python3
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from llama32_vision_common import (  # noqa: E402
    ANSWER_SUFFIXES,
    LABELS,
    LABEL_TO_INDEX,
    MODEL_NAME,
    RANK,
    get_suffix_layer_means,
    load_model_and_processor,
    project_path,
    read_csv,
)


MODES = {
    "natural_pc1": {
        "method": "counterfactual_answer_token_transition_pca",
        "short_name": "CATS-PCA",
        "aggregation": "sign_aligned_pc1",
        "source": "natural_error_transitions",
    },
    "balanced_mean": {
        "method": "balanced_counterfactual_answer_token_transition_vector",
        "short_name": "Balanced CATS",
        "aggregation": "balanced_mean",
        "source": "all_true_label_counterfactual_transitions",
    },
    "balanced_pc1": {
        "method": "balanced_counterfactual_answer_token_transition_pc1_vector",
        "short_name": "Balanced CATS-PC1",
        "aggregation": "sign_aligned_pc1",
        "source": "all_true_label_counterfactual_transitions",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Llama-3.2 Vision counterfactual answer-token vectors."
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--side", choices=["over", "under"], required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference-vector", required=True)
    parser.add_argument("--reference-vector-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-answer-tokens", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser.parse_args()


def save_cache(path, args, rows, states, complete):
    payload = {
        "method": "llama32_counterfactual_answer_suffix_cache",
        "model_name": args.model_name,
        "source_csv": str(project_path(args.base_csv)),
        "k_answer_tokens": args.k_answer_tokens,
        "answer_suffixes": ANSWER_SUFFIXES,
        "label_order": LABELS,
        "complete": complete,
        "rows": rows,
        "full_ids": [row["full_id"] for row in rows],
        "true_labels": [row["true_label"] for row in rows],
        "pred_labels": [row.get("pred_label", "") for row in rows],
        "case_types": [row.get("case_type", "") for row in rows],
        "suffix_hidden_states": torch.stack(states, dim=0).float(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved {'complete' if complete else 'partial'} cache: {path}")
    return payload


def collect_or_load_cache(args):
    cache_path = project_path(args.cache)
    rows = [
        row for row in read_csv(args.base_csv)
        if row.get("true_label") in RANK
    ]
    if args.max_rows:
        rows = rows[:args.max_rows]
    expected_ids = [row["full_id"] for row in rows]

    states = []
    if cache_path.exists() and not args.refresh_cache:
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("model_name") != args.model_name:
            raise ValueError("Cache model does not match --model-name")
        if cached.get("k_answer_tokens") != args.k_answer_tokens:
            raise ValueError("Cache K does not match --k-answer-tokens")
        cached_ids = cached.get("full_ids", [])
        if expected_ids[:len(cached_ids)] != cached_ids:
            raise ValueError("Cache rows do not match the requested base CSV")
        states = list(cached["suffix_hidden_states"])
        if cached.get("complete") and len(cached_ids) == len(expected_ids):
            print("Loaded complete shared suffix cache:", cache_path)
            return cached
        print(f"Resuming shared suffix cache at {len(states)}/{len(rows)}")

    print("Training rows:", len(rows))
    print("True label counts:", Counter(row["true_label"] for row in rows))
    print("K answer tokens:", args.k_answer_tokens)
    model, processor = load_model_and_processor(args.model_name)
    start = len(states)
    for index, row in enumerate(rows[start:], start + 1):
        states.append(
            get_suffix_layer_means(
                model, processor, row, args.k_answer_tokens
            )
        )
        print(
            f"[{index}/{len(rows)}] {row['full_id']} "
            f"true={row['true_label']} pred={row.get('pred_label', '')}"
        )
        if args.save_every and index % args.save_every == 0:
            save_cache(cache_path, args, rows[:index], states, complete=False)
    return save_cache(cache_path, args, rows, states, complete=True)


def transition_side(true_label, wrong_label):
    if RANK[wrong_label] > RANK[true_label]:
        return "over"
    if RANK[wrong_label] < RANK[true_label]:
        return "under"
    return "correct"


def collect_natural_deltas(cache, side):
    states = cache["suffix_hidden_states"].float()
    deltas = []
    names = []
    for index, row in enumerate(cache["rows"]):
        true_label = row["true_label"]
        pred_label = row.get("pred_label", "")
        if pred_label not in RANK or transition_side(true_label, pred_label) != side:
            continue
        correct_index = LABEL_TO_INDEX[true_label]
        wrong_index = LABEL_TO_INDEX[pred_label]
        deltas.append(states[index, correct_index] - states[index, wrong_index])
        names.append(f"{pred_label}_to_{true_label}")
    if not deltas:
        raise RuntimeError(f"No natural {side} errors found in the training CSV")
    return torch.stack(deltas, dim=0), names


def collect_balanced_deltas(cache, side):
    states = cache["suffix_hidden_states"].float()
    deltas = []
    names = []
    for index, row in enumerate(cache["rows"]):
        true_label = row["true_label"]
        for wrong_label in LABELS:
            if transition_side(true_label, wrong_label) != side:
                continue
            correct_index = LABEL_TO_INDEX[true_label]
            wrong_index = LABEL_TO_INDEX[wrong_label]
            deltas.append(states[index, correct_index] - states[index, wrong_index])
            names.append(f"{wrong_label}_to_{true_label}")
    if not deltas:
        raise RuntimeError(f"No balanced {side} transitions could be constructed")
    return torch.stack(deltas, dim=0), names


def rescale_to_reference(vector, reference):
    unit = vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return unit * reference.norm(dim=1, keepdim=True).clamp_min(1e-8)


def raw_mean(deltas, reference):
    return rescale_to_reference(deltas.mean(dim=0), reference)


def balanced_mean(deltas, names, reference):
    grouped = defaultdict(list)
    for delta, name in zip(deltas, names):
        grouped[name].append(delta)
    transition_means = [
        torch.stack(group, dim=0).mean(dim=0)
        for group in grouped.values()
    ]
    vector = torch.stack(transition_means, dim=0).mean(dim=0)
    return rescale_to_reference(vector, reference)


def sign_aligned_pc1(deltas, reference):
    vectors = []
    for layer in range(deltas.shape[1]):
        values = deltas[:, layer, :].float()
        mean_vector = values.mean(dim=0)
        unit = values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)
        centered = unit - unit.mean(dim=0, keepdim=True)
        if values.shape[0] < 2 or float(centered.norm()) < 1e-8:
            pc1 = mean_vector
        else:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            pc1 = vh[0].float()
            if float(torch.dot(pc1, mean_vector)) < 0:
                pc1 = -pc1
        vectors.append(pc1)
        print(f"PCA layer {layer + 1}/{deltas.shape[1]}")
    return rescale_to_reference(torch.stack(vectors, dim=0), reference)


def load_reference(args):
    payload = torch.load(project_path(args.reference_vector), map_location="cpu")
    return payload[args.reference_vector_key].float()


def main():
    args = parse_args()
    cache = collect_or_load_cache(args)
    reference = load_reference(args)

    if args.mode == "natural_pc1":
        deltas, names = collect_natural_deltas(cache, args.side)
    else:
        deltas, names = collect_balanced_deltas(cache, args.side)

    method_vectors = {
        "raw_mean": raw_mean(deltas, reference),
        "balanced_mean": balanced_mean(deltas, names, reference),
        "sign_aligned_pc1": sign_aligned_pc1(deltas, reference),
    }
    selected_method = MODES[args.mode]["aggregation"]
    selected = method_vectors[selected_method]
    vector_key = "behavior_vector" if args.side == "over" else "under_behavior_vector"
    unit_key = f"{vector_key}_unit"

    result = {
        "method": MODES[args.mode]["method"],
        "method_short_name": MODES[args.mode]["short_name"],
        "model_name": args.model_name,
        "side": args.side,
        "source_distribution": MODES[args.mode]["source"],
        "source_csv": str(project_path(args.base_csv)),
        "suffix_cache": str(project_path(args.cache)),
        "reference_vector": str(project_path(args.reference_vector)),
        "reference_vector_key": args.reference_vector_key,
        "selected_method": selected_method,
        "aggregation": selected_method,
        "transition_definition": (
            "delta = h(true_label answer suffix) - h(wrong_label answer suffix)"
        ),
        "k_answer_tokens": args.k_answer_tokens,
        "train_sample_count": len(cache["rows"]),
        "transition_count": len(names),
        "transition_counts": dict(Counter(names)),
        "num_layers": selected.shape[0],
        "hidden_size": selected.shape[1],
        "method_vectors": method_vectors,
        "method_layer_norms": {
            key: value.norm(dim=1) for key, value in method_vectors.items()
        },
        "layer_norms": selected.norm(dim=1),
        vector_key: selected,
        unit_key: selected / selected.norm(dim=1, keepdim=True).clamp_min(1e-8),
    }
    if args.side == "under":
        result["under_B_behavior_vector"] = result[vector_key]
        result["under_B_behavior_vector_unit"] = result[unit_key]

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print("Saved:", output)
    print("Method:", result["method_short_name"])
    print("Side:", args.side)
    print("Transitions:", result["transition_counts"])
    print("Vector shape:", tuple(selected.shape))


if __name__ == "__main__":
    main()
