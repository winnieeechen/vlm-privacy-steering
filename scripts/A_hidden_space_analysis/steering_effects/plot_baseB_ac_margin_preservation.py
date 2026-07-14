#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


COLORS = {"A": "#d95f59", "C": "#59a14f"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "outputs").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_base_b_rows(path):
    with path.open("r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["base_pred"] == "B"]
    if not rows:
        raise ValueError(f"No base-prediction B samples in {path}")
    unexpected = sorted(set(row["steered_pred"] for row in rows) - {"A", "C"})
    if unexpected:
        raise ValueError(f"Unexpected steered predictions among base-B samples: {unexpected}")
    return rows


def compute(rows):
    base_margin = np.array([
        float(row["base_score_A"]) - float(row["base_score_C"]) for row in rows
    ])
    steered_margin = np.array([
        float(row["steered_score_A"]) - float(row["steered_score_C"]) for row in rows
    ])
    steered_labels = np.array([row["steered_pred"] for row in rows])
    target_a = steered_labels == "A"

    # torch.argmax chooses the first class on an exact A/C tie, so margin=0 maps to A.
    sign_prediction = np.where(base_margin >= 0, "A", "C")
    correlation = np.corrcoef(base_margin, steered_margin)[0, 1]
    accuracy = np.mean(sign_prediction == steered_labels)
    auc = roc_auc_score(target_a, base_margin)
    side_preservation = np.mean((base_margin >= 0) == (steered_margin >= 0))
    return {
        "base_margin": base_margin,
        "steered_margin": steered_margin,
        "steered_labels": steered_labels,
        "sign_prediction": sign_prediction,
        "correlation": correlation,
        "accuracy": accuracy,
        "auc": auc,
        "side_preservation": side_preservation,
        "base_ties": int(np.sum(base_margin == 0)),
        "steered_ties": int(np.sum(steered_margin == 0)),
    }


def plot_scatter(ax, result):
    base = result["base_margin"]
    steered = result["steered_margin"]
    labels = result["steered_labels"]
    limits = [min(base.min(), steered.min()) - 0.15, max(base.max(), steered.max()) + 0.15]
    for label in ("A", "C"):
        mask = labels == label
        ax.scatter(
            base[mask], steered[mask], s=40, color=COLORS[label], alpha=0.72,
            edgecolor="white", linewidth=0.4,
            label=f"steered {label} (n={mask.sum()})",
        )
    ax.plot(limits, limits, color="#555555", linestyle="--", linewidth=1.3, label="y = x")
    ax.axhline(0, color="#aaaaaa", linewidth=0.9)
    ax.axvline(0, color="#aaaaaa", linewidth=0.9)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("base score A - score C")
    ax.set_ylabel("steered score A - score C")
    ax.set_title("A/C relative margin is largely preserved", weight="bold")
    ax.text(
        0.03, 0.97,
        f"Pearson r = {result['correlation']:.3f}\n"
        f"same A/C side = {100 * result['side_preservation']:.1f}%",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92, "pad": 6},
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_distribution(ax, result):
    base = result["base_margin"]
    labels = result["steered_labels"]
    bins = np.linspace(base.min() - 0.1, base.max() + 0.1, 25)
    for label in ("A", "C"):
        values = base[labels == label]
        ax.hist(
            values, bins=bins, density=True, alpha=0.48, color=COLORS[label],
            edgecolor="white", linewidth=0.5, label=f"steered {label}",
        )
        ax.axvline(
            np.median(values), color=COLORS[label], linewidth=2.0, linestyle=":"
        )
    ax.axvline(0, color="#111111", linestyle="--", linewidth=1.5, label="sign threshold")
    ax.set_xlabel("base score A - score C")
    ax.set_ylabel("density")
    ax.set_title("The pre-steering margin predicts the final A/C choice", weight="bold")
    ax.text(
        0.03, 0.97,
        f"sign-rule accuracy = {100 * result['accuracy']:.1f}%\n"
        f"ROC-AUC = {result['auc']:.3f}\n"
        f"tie rule: A-C >= 0 predicts A",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92, "pad": 6},
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def write_summary(path, rows, result):
    labels = result["steered_labels"]
    prediction = result["sign_prediction"]
    fields = [
        "n_base_B", "n_steered_A", "n_steered_C", "pearson_base_vs_steered_AC",
        "sign_rule_accuracy", "roc_auc_predict_steered_A", "ac_side_preservation",
        "base_exact_ties", "steered_exact_ties", "sign_rule_correct", "sign_rule_incorrect",
        "mean_delta_AC_margin", "sd_delta_AC_margin",
    ]
    delta = result["steered_margin"] - result["base_margin"]
    record = {
        "n_base_B": len(rows),
        "n_steered_A": int(np.sum(labels == "A")),
        "n_steered_C": int(np.sum(labels == "C")),
        "pearson_base_vs_steered_AC": result["correlation"],
        "sign_rule_accuracy": result["accuracy"],
        "roc_auc_predict_steered_A": result["auc"],
        "ac_side_preservation": result["side_preservation"],
        "base_exact_ties": result["base_ties"],
        "steered_exact_ties": result["steered_ties"],
        "sign_rule_correct": int(np.sum(prediction == labels)),
        "sign_rule_incorrect": int(np.sum(prediction != labels)),
        "mean_delta_AC_margin": float(delta.mean()),
        "sd_delta_AC_margin": float(delta.std(ddof=1)),
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(record)


def main(args):
    input_path = project_path(args.input_csv)
    output_path = project_path(args.output)
    summary_path = project_path(args.summary_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_base_b_rows(input_path)
    result = compute(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), constrained_layout=True)
    plot_scatter(axes[0], result)
    plot_distribution(axes[1], result)
    fig.suptitle(
        "Base-prediction B samples: steering suppresses B while preserving the A/C preference",
        fontsize=15, weight="bold",
    )
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    write_summary(summary_path, rows, result)

    print("Base-B samples:", len(rows))
    print("Steered counts:", {label: int(np.sum(result["steered_labels"] == label)) for label in ("A", "C")})
    print("Pearson(base A-C, steered A-C):", f"{result['correlation']:.6f}")
    print("Sign-rule accuracy:", f"{result['accuracy']:.6f}")
    print("ROC-AUC:", f"{result['auc']:.6f}")
    print("A/C side preservation:", f"{result['side_preservation']:.6f}")
    print("Wrote", output_path)
    print("Wrote", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot preservation of the A-vs-C margin for base-B samples.")
    parser.add_argument(
        "--input-csv",
        default="notes/A_hidden_space_analysis/output_margins/output_margin_changes.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "notes/A_hidden_space_analysis/output_margins/"
            "baseB_A_vs_C_margin_preservation.png"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=(
            "notes/A_hidden_space_analysis/output_margins/"
            "baseB_A_vs_C_margin_preservation_summary.csv"
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
