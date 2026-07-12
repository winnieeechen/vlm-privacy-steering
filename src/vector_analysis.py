#!/usr/bin/env python3
"""Check whether the A-B and B-C pairwise vectors share a direction.

For behavior vectors (answer-token mean) and condition vectors (last prompt
token) separately, compute the per-layer cosine similarity between the
A_minus_B and B_minus_C vectors for every method variant. High cosine means
the model encodes disclosure granularity as one ordinal axis; near-zero means
A-vs-B and B-vs-C are distinct distinctions and deserve separate vectors.

Random baseline in d dimensions: |cos| ~ 1/sqrt(d) (~0.022 for d=2048).
"""
import argparse
import csv
from pathlib import Path

import torch


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()

FAMILIES = {
    "behavior": ROOT / "outputs" / "behavior_vectors" / "vectors" / "behavior_vectors_{pair}.pt",
    "condition": ROOT / "outputs" / "condition_vectors" / "vectors" / "condition_vectors_{pair}.pt",
}
PAIRS = ("A_minus_B", "B_minus_C")

FAMILY_TITLES = {
    "behavior": "behavior (answer-token mean, pred_label pairs)",
    "condition": "condition (last prompt token, true_label pairs)",
}

# Fixed color per method (validated categorical palette); color follows the
# entity regardless of plotting order.
METHOD_COLORS = {
    "mean_diff": "#2a78d6",
    "pca_projected": "#1baf7a",
    "pca_residual": "#eda100",
    "fisher_pca": "#008300",
    "ensemble": "#4a3aa7",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
STEERING_LAYER = 32


def layer_cosines(a, b):
    """Compute per-layer cosine similarity between two sets of vectors."""
    a = a.float()
    b = b.float()
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    return num / den


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=ROOT / "outputs" / "vector_analysis" / "pairwise_direction_cosine.png",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs" / "vector_analysis" / "pairwise_direction_cosine.csv",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def plot_families(results, output_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILIES if f in results]
    fig, axes = plt.subplots(
        1, len(families), figsize=(6.0 * len(families), 4.6), sharey=True, dpi=150
    )
    if len(families) == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        per_method = results[family]
        num_layers = len(next(iter(per_method.values())))
        layers = range(num_layers)

        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1, alpha=0.6)
        ax.axvline(
            STEERING_LAYER, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)), alpha=0.5
        )
        ax.text(
            STEERING_LAYER + 0.4, -0.96, f"layer {STEERING_LAYER}",
            color=TEXT_SECONDARY, fontsize=8, va="bottom",
        )

        # Draw in reverse so mean_diff (the headline series) ends up on top;
        # pca_projected tracks it almost exactly and would otherwise hide it.
        for method, color in reversed(METHOD_COLORS.items()):
            if method not in per_method:
                continue
            ax.plot(layers, per_method[method].tolist(), color=color, linewidth=2, label=method)

        ax.set_title(FAMILY_TITLES.get(family, family), color=TEXT_PRIMARY, fontsize=11)
        ax.set_xlabel("layer", color=TEXT_SECONDARY)
        ax.set_xlim(0, num_layers - 1)
        ax.set_ylim(-1, 1)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    axes[0].set_ylabel("cos(A−B, B−C)", color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    order = list(METHOD_COLORS)
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: order.index(t[0])))
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), frameon=False,
        fontsize=9, labelcolor=TEXT_PRIMARY, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Do the A−B and B−C vectors share a direction?",
        color=TEXT_PRIMARY, fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_png)

def write_results(output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "method", "layer", "cosine"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved:", output_csv)

def main():
    args = parse_args()
    rows = []
    results = {}

    for family, template in FAMILIES.items():
        payloads = {}
        for pair in PAIRS:
            path = Path(str(template).format(pair=pair))
            if not path.exists():
                print(f"[{family}] missing {path}, skipping family")
                payloads = None
                break
            payloads[pair] = torch.load(path, map_location="cpu")
        if payloads is None:
            continue

        methods = sorted(payloads[PAIRS[0]]["method_vectors"])
        num_layers = payloads[PAIRS[0]]["num_layers"]
        hidden_size = payloads[PAIRS[0]]["hidden_size"]

        print(f"\n=== {family}: cos(A_minus_B, B_minus_C) ===")
        print(f"layers={num_layers} hidden={hidden_size} random |cos| ~ {hidden_size ** -0.5:.3f}")

        per_method = {}
        results[family] = per_method
        for method in methods:
            cos = layer_cosines(
                payloads["A_minus_B"]["method_vectors"][method],
                payloads["B_minus_C"]["method_vectors"][method],
            )
            per_method[method] = cos
            for layer, value in enumerate(cos.tolist()):
                rows.append(
                    {"family": family, "method": method, "layer": layer, "cosine": f"{value:.4f}"}
                )
            print(
                f"{method:>14}: mean={cos.mean():+.3f} min={cos.min():+.3f} "
                f"max={cos.max():+.3f} layer32={cos[32]:+.3f}"
                if num_layers > 32
                else f"{method:>14}: mean={cos.mean():+.3f} min={cos.min():+.3f} max={cos.max():+.3f}"
            )

        print("\nlayer  " + "  ".join(f"{m:>14}" for m in methods))
        for layer in range(num_layers):
            print(f"{layer:>5}  " + "  ".join(f"{per_method[m][layer]:>14.3f}" for m in methods))
    
    write_to_csv = False
    if write_to_csv:
        write_results(args.output_csv)

    if results and not args.no_plot:
        plot_families(results, args.output_png)


if __name__ == "__main__":
    main()
