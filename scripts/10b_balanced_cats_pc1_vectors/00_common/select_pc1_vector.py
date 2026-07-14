#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


METHOD_KEY = "sign_aligned_pc1"


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_pc1_vector(source, output, side):
    source_path = project_path(source)
    output_path = project_path(output)
    payload = torch.load(source_path, map_location="cpu")
    vector = payload["method_vectors"][METHOD_KEY].float()

    result = dict(payload)
    result.update({
        "method": "balanced_counterfactual_answer_token_transition_pc1_vector",
        "method_short_name": "Balanced CATS-PC1",
        "parent_payload": str(source_path),
        "selected_method": METHOD_KEY,
        "aggregation": METHOD_KEY,
        "behavior_vector": vector,
        "behavior_vector_unit": vector / vector.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-8),
        "layer_norms": vector.norm(dim=1),
    })
    if side == "under":
        result["under_B_behavior_vector"] = result["behavior_vector"]
        result["under_B_behavior_vector_unit"] = result["behavior_vector_unit"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print("Source:", source_path)
    print("Saved:", output_path)
    print("Side:", side)
    print("Selected method:", METHOD_KEY)
    print("Transition count:", result.get("transition_count"))
    print("Layer 28 norm:", float(result["layer_norms"][28]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--side", choices=["over", "under"], required=True)
    args = parser.parse_args()
    select_pc1_vector(args.source, args.output, args.side)


if __name__ == "__main__":
    main()
