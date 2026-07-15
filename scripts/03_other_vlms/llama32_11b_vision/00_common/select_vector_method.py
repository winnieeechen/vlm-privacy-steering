#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = project_root()


def path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--side", choices=["over", "under"], required=True)
    parser.add_argument("--method", default="sign_aligned_pc1")
    parser.add_argument("--name", default="Balanced CATS-PC1")
    args = parser.parse_args()

    source = path(args.source)
    payload = torch.load(source, map_location="cpu")
    vector = payload["method_vectors"][args.method].float()
    vector_key = "behavior_vector" if args.side == "over" else "under_behavior_vector"
    unit_key = f"{vector_key}_unit"
    result = dict(payload)
    result.update({
        "method": "balanced_counterfactual_answer_token_transition_pc1_vector",
        "method_short_name": args.name,
        "parent_payload": str(source),
        "selected_method": args.method,
        "aggregation": args.method,
        vector_key: vector,
        unit_key: vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8),
        "layer_norms": vector.norm(dim=1),
    })
    if args.side == "under":
        result["under_B_behavior_vector"] = result[vector_key]
        result["under_B_behavior_vector_unit"] = result[unit_key]

    output = path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print("Source:", source)
    print("Saved:", output)
    print("Selected:", args.method)
    print("Shape:", tuple(vector.shape))


if __name__ == "__main__":
    main()
