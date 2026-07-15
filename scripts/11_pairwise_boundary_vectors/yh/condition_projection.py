#!/usr/bin/env python3
"""Do prompt-token activations cluster by true label under the condition vectors?

Projects every cached training example onto the plane spanned by the A-B and
B-C condition vectors (orthonormalized) and onto the shared sensitivity axis
(the average of the two unit vectors). Scatter is colored by true_label; the
1-D panel shows per-class score distributions plus the best two-threshold
ordinal classifier accuracy — an upper bound for score-based routing.

Runs entirely from the activation cache; no model forward needed.
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

CACHE = ROOT / "outputs" / "condition_vectors" / "cache" / "prompt_last_token_activations.pt"
BASE_CSV = ROOT / "outputs" / "02_formal_full1200" / "00_base" / "base_qwen_vl_train_717.csv"
CONDITION_VECTORS = {
    "A_minus_B": ROOT / "outputs" / "condition_vectors" / "vectors" / "condition_vectors_A_minus_B.pt",
    "B_minus_C": ROOT / "outputs" / "condition_vectors" / "vectors" / "condition_vectors_B_minus_C.pt",
}

LABELS = ["A", "B", "C"]
LABEL_COLORS = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=ROOT / "outputs" / "vector_analysis" / "condition_projection_scatter.png",
    )
    return parser.parse_args()


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def base_pred_accuracy(base_csv):
    """Accuracy of the unsteered base model: pred_label == true_label in the base CSV."""
    with open(base_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    correct = sum(row["pred_label"] == row["true_label"] for row in rows)
    return correct / len(rows), correct, len(rows)


def ordinal_threshold_accuracy(scores, labels):
    """Exhaustively search the best two-threshold ordinal classifier on a 1-D score.

    The classifier assumes scores are ordered A > B > C and predicts, for a
    pair of thresholds t1 >= t2:

        A if s >= t1,  B if t2 <= s < t1,  C if s < t2

    Every observed score value is tried as a candidate for both thresholds
    (O(n^2) pairs), so the returned accuracy is the exact in-sample optimum
    for this classifier family — an upper bound for score-based routing.
    t1 == t2 is allowed, which degenerates to a binary A/C split (no B band).

    Args:
        scores: 1-D float tensor, shape [n]. Projection of each example onto
            the chosen axis u (higher = more privacy-sensitive / more A-like).
        labels: length-n sequence of true labels, each one of "A" / "B" / "C",
            aligned with `scores` by index.

    Returns:
        (best_acc, (t1, t2)):
        best_acc: float in [0, 1], the best fraction of examples classified
            correctly over all threshold pairs.
        t1, t2: floats (t1 >= t2), the maximizing thresholds in score units.
    """
    candidates = torch.unique(scores)
    y = torch.tensor([LABELS.index(l) for l in labels])
    best_acc, best = 0.0, (None, None)
    for i, t1 in enumerate(candidates):
        for t2 in candidates[: i + 1]:
            pred = torch.where(scores >= t1, 0, torch.where(scores >= t2, 1, 2))
            acc = (pred == y).float().mean().item()
            if acc > best_acc:
                best_acc, best = acc, (t1.item(), t2.item())
    return best_acc, best


def main():
    args = parse_args()

    cache = torch.load(CACHE, map_location="cpu")
    acts = cache["activations"][:, args.layer, :].float()
    labels = cache["labels"]
    acts = acts - acts.mean(dim=0, keepdim=True)

    v_ab = torch.load(CONDITION_VECTORS["A_minus_B"], map_location="cpu")["method_vectors"][args.method][args.layer].float()
    v_bc = torch.load(CONDITION_VECTORS["B_minus_C"], map_location="cpu")["method_vectors"][args.method][args.layer].float()

    # Orthonormal basis of the plane spanned by the two condition vectors.
    e1 = unit(v_ab)
    e2 = unit(v_bc - (v_bc @ e1) * e1)
    x = acts @ e1
    y = acts @ e2

    # Shared sensitivity axis: average of the two unit directions.
    u = unit(unit(v_ab) + unit(v_bc))   # The angle bisector between v_ab and v_bc
    # u = unit(v_ab)
    # u = unit(v_bc)
    s = acts @ u

    print(f"layer={args.layer} method={args.method} n={len(labels)}")
    print(f"cos(A-B, B-C) at this layer: {unit(v_ab) @ unit(v_bc):.3f}")
    for label in LABELS:
        mask = torch.tensor([l == label for l in labels])
        print(
            f"  {label}: n={int(mask.sum())} "
            f"s mean={s[mask].mean():+.2f} std={s[mask].std():.2f}"
        )

    acc, (t1, t2) = ordinal_threshold_accuracy(s, labels)
    majority = max(labels.count(l) for l in LABELS) / len(labels)
    print(f"Shared-axis 2-threshold accuracy: {acc:.3f} (t1={t1:.2f}, t2={t2:.2f})")
    print(f"Majority-class baseline: {majority:.3f}")
    base_acc, base_correct, base_n = base_pred_accuracy(BASE_CSV)
    print(f"Base model pred accuracy: {base_acc:.3f} ({base_correct}/{base_n})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    # Scatter plot of the condition-vector plane.
    for label in LABELS:
        mask = torch.tensor([l == label for l in labels])
        ax1.scatter(
            x[mask], y[mask], s=18, alpha=0.65, linewidths=0.5,
            color=LABEL_COLORS[label], edgecolors="#fcfcfb", label=f"true {label}",
        )
    ax1.set_xlabel("projection on unit(A−B)", color=TEXT_SECONDARY)
    ax1.set_ylabel("projection on unit(B−C ⊥ A−B)", color=TEXT_SECONDARY)
    ax1.set_title(f"condition-vector plane (layer {args.layer})", color=TEXT_PRIMARY, fontsize=11)

    # Histogram of the shared sensitivity axis.
    bins = 40
    lo, hi = s.min().item(), s.max().item()
    for label in LABELS:
        mask = torch.tensor([l == label for l in labels])
        ax2.hist(
            s[mask].tolist(), bins=bins, range=(lo, hi), histtype="step",
            linewidth=2, color=LABEL_COLORS[label], label=f"true {label}",
        )
    for t, name in ((t1, "t1"), (t2, "t2")):
        ax2.axvline(t, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)), alpha=0.6)
        ax2.text(t, ax2.get_ylim()[1] * 0.95, name, color=TEXT_SECONDARY, fontsize=8, ha="left")
    ax2.set_xlabel("score on shared sensitivity axis u", color=TEXT_SECONDARY)
    ax2.set_ylabel("count", color=TEXT_SECONDARY)
    ax2.set_title(
        f"shared-axis score (2-threshold acc {acc:.1%})", color=TEXT_PRIMARY, fontsize=11
    )

    for ax in (ax1, ax2):
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    fig.suptitle(
        "Do inputs cluster by true label under the condition vectors?",
        color=TEXT_PRIMARY, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, facecolor="#fcfcfb", bbox_inches="tight")
    print("Saved:", args.output_png)


if __name__ == "__main__":
    main()
