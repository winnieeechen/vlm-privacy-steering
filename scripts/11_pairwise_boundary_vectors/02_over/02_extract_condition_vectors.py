#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "11_pairwise_boundary_vectors" / "00_common"))

from pairwise_boundary_extraction import (  # noqa: E402
    LABEL_PAIRS,
    base_parser,
    build_payload,
    collect_or_load_activations,
    compose_axis,
    print_label_distribution,
    project_path,
    read_rows,
    save_payload,
)


def parse_args():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--activation-cache",
        default="outputs/11_pairwise_boundary_vectors/00_cache/prompt_last_token_activations.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/11_pairwise_boundary_vectors/02_over/vectors",
    )
    parser.add_argument(
        "--compose-under",
        action="store_true",
        help="Also write the reverse composed axis for under gating.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.answer_max_tokens = None
    rows = read_rows(args.base_csv)
    print("Loaded base rows:", len(rows))
    label_counts = print_label_distribution(rows, "true_label")
    activations, labels = collect_or_load_activations(
        rows=rows,
        cache_path=args.activation_cache,
        refresh_cache=args.refresh_cache,
        label_column="true_label",
        use_answer=False,
        model_name=args.model_name,
    )

    pair_payloads = {}
    output_dir = project_path(args.output_dir)
    for pos_label, neg_label in LABEL_PAIRS:
        indices = []
        binary_labels = []
        for i, label in enumerate(labels):
            if label == pos_label:
                indices.append(i)
                binary_labels.append(1)
            elif label == neg_label:
                indices.append(i)
                binary_labels.append(-1)
        group = f"{pos_label}_minus_{neg_label}"
        payload = build_payload(
            args=args,
            activations=activations[indices],
            binary_labels=binary_labels,
            group_name=group,
            definition={
                "activation": "last prompt-token hidden state (image + question, no answer)",
                "positive": f"true_label {pos_label} inputs",
                "negative": f"true_label {neg_label} inputs",
                "direction": f"{neg_label} -> {pos_label}",
            },
            label_counts=label_counts,
            vector_key="condition_vector",
        )
        pair_payloads[group] = payload
        save_payload(payload, output_dir / f"condition_vectors_{group}.pt")

    compose_axis(
        pair_payloads,
        args,
        vector_key="condition_vector",
        output_path=output_dir / "condition_vectors_pairwise_axis_over.pt",
        side="over",
    )
    if args.compose_under:
        compose_axis(
            pair_payloads,
            args,
            vector_key="condition_vector",
            output_path=project_path("outputs/11_pairwise_boundary_vectors/03_under/vectors/under_condition_vectors_pairwise_axis.pt"),
            side="under",
        )


if __name__ == "__main__":
    main()

