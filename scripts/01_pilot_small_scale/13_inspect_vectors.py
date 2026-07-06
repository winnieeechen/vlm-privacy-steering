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

COND_PATH = ROOT / "outputs" / "vectors" / "condition_vectors_qwen_vl_40.pt"
BEHAV_PATH = ROOT / "outputs" / "vectors" / "behavior_vectors_qwen_vl_40.pt"


def topk_layers(norms, k=10):
    values, indices = torch.topk(norms, k=min(k, len(norms)))
    return [(int(i), float(v)) for i, v in zip(indices, values)]


def main():
    cond = torch.load(COND_PATH, map_location="cpu")
    behav = torch.load(BEHAV_PATH, map_location="cpu")

    c = cond["condition_vector"].float()
    v = behav["behavior_vector"].float()

    print("Condition vector:")
    print(" path:", COND_PATH)
    print(" shape:", tuple(c.shape))
    print(" positive_count:", cond["positive_count"])
    print(" negative_count:", cond["negative_count"])

    print("\nBehavior vector:")
    print(" path:", BEHAV_PATH)
    print(" shape:", tuple(v.shape))
    print(" positive_count:", behav["positive_count"])
    print(" negative_count:", behav["negative_count"])

    c_norms = c.norm(dim=1)
    v_norms = v.norm(dim=1)

    print("\nTop condition-vector layers by norm:")
    for layer, norm in topk_layers(c_norms, k=10):
        print(f" layer {layer:02d}: norm={norm:.4f}")

    print("\nTop behavior-vector layers by norm:")
    for layer, norm in topk_layers(v_norms, k=10):
        print(f" layer {layer:02d}: norm={norm:.4f}")

    cosine = F.cosine_similarity(c, v, dim=1)

    print("\nLayer-wise cosine similarity between c_privacy and v_privacy:")
    for i, score in enumerate(cosine.tolist()):
        print(f" layer {i:02d}: cosine={score:.4f}")

    print("\nTop layers by absolute cosine similarity:")
    abs_values, abs_indices = torch.topk(cosine.abs(), k=10)
    for i, val in zip(abs_indices, abs_values):
        i = int(i)
        print(
            f" layer {i:02d}: cosine={float(cosine[i]):.4f}, "
            f"c_norm={float(c_norms[i]):.4f}, v_norm={float(v_norms[i]):.4f}"
        )

    print("\nReminder:")
    print(" layer index 0 means model.model.language_model.layers[0]")
    print(" layer index 35 means model.model.language_model.layers[35]")


if __name__ == "__main__":
    main()
