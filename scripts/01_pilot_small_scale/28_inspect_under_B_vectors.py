#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn.functional as F


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()

UTIL_PATH = ROOT / "outputs" / "vectors" / "utility_condition_vectors_qwen_vl_train.pt"
UNDER_PATH = ROOT / "outputs" / "vectors" / "under_B_behavior_vectors_qwen_vl_train.pt"

OVER_COND_PATH = ROOT / "outputs" / "vectors" / "condition_vectors_qwen_vl_train.pt"
OVER_BEHAV_PATH = ROOT / "outputs" / "vectors" / "behavior_vectors_qwen_vl_train.pt"


def topk_layers(norms, k=10):
    values, indices = torch.topk(norms, k=min(k, len(norms)))
    return [(int(i), float(v)) for i, v in zip(indices, values)]


def main():
    util = torch.load(UTIL_PATH, map_location="cpu")
    under = torch.load(UNDER_PATH, map_location="cpu")

    over_cond = torch.load(OVER_COND_PATH, map_location="cpu")
    over_behav = torch.load(OVER_BEHAV_PATH, map_location="cpu")

    c_util = util["utility_condition_vector"].float()
    v_under = under["under_B_behavior_vector"].float()

    c_over = over_cond["condition_vector"].float()
    v_over = over_behav["behavior_vector"].float()

    util_norms = c_util.norm(dim=1)
    under_norms = v_under.norm(dim=1)

    cosine_util_under = F.cosine_similarity(c_util, v_under, dim=1)
    cosine_under_over = F.cosine_similarity(v_under, v_over, dim=1)
    cosine_util_overcond = F.cosine_similarity(c_util, c_over, dim=1)

    print("Utility condition vector:")
    print(" shape:", tuple(c_util.shape))
    print(" positive_count:", util["positive_count"])
    print(" negative_count:", util["negative_count"])

    print("\nUnder-B behavior vector:")
    print(" shape:", tuple(v_under.shape))
    print(" positive_count:", under["positive_count"])
    print(" negative_count:", under["negative_count"])

    print("\nTop utility-condition layers by norm:")
    for layer, norm in topk_layers(util_norms):
        print(f" layer {layer:02d}: norm={norm:.4f}")

    print("\nTop under-B behavior layers by norm:")
    for layer, norm in topk_layers(under_norms):
        print(f" layer {layer:02d}: norm={norm:.4f}")

    print("\nTop layers by abs cosine: utility_condition vs under_B_behavior")
    vals, idxs = torch.topk(cosine_util_under.abs(), k=10)
    for i, val in zip(idxs, vals):
        i = int(i)
        print(
            f" layer {i:02d}: cosine={float(cosine_util_under[i]):.4f}, "
            f"util_norm={float(util_norms[i]):.4f}, "
            f"under_norm={float(under_norms[i]):.4f}"
        )

    print("\nCosine: under_B_behavior vs over_behavior")
    vals, idxs = torch.topk(cosine_under_over.abs(), k=10)
    for i, val in zip(idxs, vals):
        i = int(i)
        print(
            f" layer {i:02d}: cosine={float(cosine_under_over[i]):.4f}"
        )

    print("\nCosine: utility_condition vs privacy_condition")
    vals, idxs = torch.topk(cosine_util_overcond.abs(), k=10)
    for i, val in zip(idxs, vals):
        i = int(i)
        print(
            f" layer {i:02d}: cosine={float(cosine_util_overcond[i]):.4f}"
        )

    print("\nReminder:")
    print(" layer index 0 = model.model.language_model.layers[0]")


if __name__ == "__main__":
    main()
