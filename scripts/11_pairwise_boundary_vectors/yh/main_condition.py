#!/usr/bin/env python3
"""Extract prompt-token condition vectors grouped by ground-truth label.

Like src/main.py but for condition vectors: the activation is the last
prompt-token hidden state (image + question, no answer), and rows are grouped
by true_label. Default grouping builds the A-B and B-C pairwise vectors; the
case grouping reproduces the classic over/under condition definitions
(over: A/B vs C, under: B/C vs A). All variants from build_vectors
(mean_diff / pca / fisher_pca / ensemble) are saved.

Model loading, activation extraction/caching, and vector building are reused
from src/main.py.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_behavior import (  # noqa: E402
    BASE_CSV,
    LABEL_PAIRS,
    ROOT,
    build_and_save,
    collect_or_load_activations,
    print_label_distribution
)

ACTIVATION_DEFINITION = "last prompt-token hidden state (image + question, no answer)"

CONDITION_CASES = {
    "over": {
        "positive": {"A", "B"},
        "negative": {"C"},
    },
    "under": {
        "positive": {"B", "C"},
        "negative": {"A"},
    },
}




def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=BASE_CSV)
    parser.add_argument("--side", choices=["over", "under"], default="over")
    parser.add_argument(
        "--grouping",
        choices=["label_pair", "case"],
        default="label_pair",
        help="label_pair: build A-B and B-C vectors grouped by true_label. "
        "case: classic over/under condition definition for --side.",
    )
    parser.add_argument(
        "--method",
        choices=["mean_diff", "pca_projected", "pca_residual", "fisher_pca", "ensemble"],
        default="mean_diff",
    )
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--fisher-rank", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Only used for case grouping; label_pair writes one file per pair.",
    )
    parser.add_argument("--activation-cache", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    out_root = ROOT / "outputs" / "condition_vectors"
    if args.activation_cache is None:
        # The cache holds all base rows with A/B/C true labels, so every grouping shares it.
        args.activation_cache = out_root / "cache" / "prompt_last_token_activations.pt"

    with open(args.base_csv, "r", encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    print("Loaded base rows:", len(base_rows))
    print("Grouping:", args.grouping)
    print("Selected method:", args.method)
    true_label_counts = print_label_distribution(base_rows, column="true_label")
    
    activations, labels = collect_or_load_activations(
        rows=base_rows,
        cache_path=args.activation_cache,
        refresh_cache=args.refresh_cache,
        label_column="true_label",
        use_answer=False,
    )

    groups = []
    if args.grouping == "label_pair":
        if args.output is not None:
            print("Note: --output is ignored for label_pair grouping (one file per pair).")
        for pos_label, neg_label in LABEL_PAIRS:
            indices = []
            binary_labels = []
            for i, label in enumerate(labels):
                if label == pos_label:
                    binary_labels.append(1)
                elif label == neg_label:
                    binary_labels.append(-1)
                else:
                    continue
                indices.append(i)

            name = f"{pos_label}_minus_{neg_label}"
            output = out_root / "vectors" / f"condition_vectors_{name}.pt"
            definition = {
                "activation": ACTIVATION_DEFINITION,
                "positive": f"true_label {pos_label} inputs, last prompt token",
                "negative": f"true_label {neg_label} inputs, last prompt token",
            }
            groups.append((name, indices, binary_labels, output, definition))
    else:
        cases = CONDITION_CASES[args.side]
        indices = []
        binary_labels = []
        for i, label in enumerate(labels):
            if label in cases["positive"]:
                binary_labels.append(1)
            elif label in cases["negative"]:
                binary_labels.append(-1)
            else:
                continue
            indices.append(i)

        output = args.output or (
            out_root / f"0{2 if args.side == 'over' else 3}_{args.side}" / "vectors" / "condition_vectors.pt"
        )
        definition = {
            "activation": ACTIVATION_DEFINITION,
            "positive": f"{args.side}: true_label in {sorted(cases['positive'])}",
            "negative": f"{args.side}: true_label in {sorted(cases['negative'])}",
        }
        groups.append((args.side, indices, binary_labels, output, definition))

    for name, indices, binary_labels, output, definition in groups:
        subset = activations[torch.tensor(indices, dtype=torch.long)]
        print(f"\n[{name}] rows: {len(indices)}")
        print(f"[{name}] positive/negative: {binary_labels.count(1)} / {binary_labels.count(-1)}")
        build_and_save(
            args=args,
            activations=subset,
            binary_labels=binary_labels,
            output_path=output,
            group_name=name,
            definition=definition,
            label_counts=true_label_counts,
            vector_key="condition_vector",
        )


if __name__ == "__main__":
    main()
