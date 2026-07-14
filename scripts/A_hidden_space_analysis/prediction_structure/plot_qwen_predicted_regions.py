#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABELS = ("A", "B", "C")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
COLORS = {"A": "#d95f59", "B": "#4c78a8", "C": "#59a14f"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "outputs").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def default_cache(split, layer):
    return (
        "outputs/02_formal_full1200/06_target_aware_probe_guided_steering/"
        f"cache/{split}_hidden_states_layer{layer}.pt"
    )


def load_hidden_cache(path):
    payload = torch.load(project_path(path), map_location="cpu")
    x = payload["features"].float().numpy()
    pred_labels = []
    true_labels = []
    full_ids = []
    for row in payload["rows"]:
        pred = row.get("pred_label", "")
        true = row.get("true_label", "")
        if pred not in LABEL_TO_ID:
            raise ValueError(f"Missing/unknown pred_label={pred!r} in {path}")
        pred_labels.append(LABEL_TO_ID[pred])
        true_labels.append(true)
        full_ids.append(row.get("full_id", ""))
    return {
        "path": str(project_path(path)),
        "payload": payload,
        "x": x,
        "y_pred": np.array(pred_labels, dtype=int),
        "true_labels": np.array(true_labels, dtype=object),
        "full_ids": np.array(full_ids, dtype=object),
    }


def fit_probe(train, c, class_weight, max_iter, estimator):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train["x"])
    pca = PCA(n_components=2, random_state=0)
    train_xy = pca.fit_transform(x_train)

    if estimator == "pca-svm":
        probe = SVC(
            C=c,
            gamma="scale",
            class_weight=None if class_weight == "none" else class_weight,
            random_state=0,
        )
        probe.fit(train_xy, train["y_pred"])
    elif estimator == "highdim-logreg":
        probe = LogisticRegression(
            C=c,
            class_weight=None if class_weight == "none" else class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=0,
        )
        probe.fit(x_train, train["y_pred"])
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return scaler, probe, pca


def transform_split(split, scaler, pca, probe, estimator):
    x_scaled = scaler.transform(split["x"])
    xy = pca.transform(x_scaled)
    if estimator == "pca-svm":
        surrogate_pred = probe.predict(xy)
    else:
        surrogate_pred = probe.predict(x_scaled)
    return {
        **split,
        "x_scaled": x_scaled,
        "xy": xy,
        "surrogate_pred": surrogate_pred,
        "surrogate_acc": accuracy_score(split["y_pred"], surrogate_pred),
        "confusion": confusion_matrix(split["y_pred"], surrogate_pred, labels=[0, 1, 2]),
    }


def make_grid(train_xy, pca, probe, estimator, padding=0.12, grid_n=420):
    mins = train_xy.min(axis=0)
    maxs = train_xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - padding * span
    maxs = maxs + padding * span
    xs = np.linspace(mins[0], maxs[0], grid_n)
    ys = np.linspace(mins[1], maxs[1], grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.c_[xx.ravel(), yy.ravel()]

    if estimator == "pca-svm":
        # Estimate regions directly in the displayed 2-D hidden-state projection.
        zz = probe.predict(grid_xy).reshape(xx.shape)
    else:
        # Optional diagnostic mode: lift the PCA plane back into standardized
        # hidden-state space, then evaluate a high-dimensional surrogate.
        grid_scaled = pca.inverse_transform(grid_xy)
        zz = probe.predict(grid_scaled).reshape(xx.shape)
    return xx, yy, zz


def plot_regions(ax, xx, yy, zz):
    cmap = ListedColormap([COLORS[label] for label in LABELS])
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.18)
    ax.contour(xx, yy, zz, levels=[0.5, 1.5], colors="#262626", linewidths=1.0, alpha=0.65)


def scatter_split(ax, split, title, pca):
    for label in LABELS:
        idx = split["y_pred"] == LABEL_TO_ID[label]
        ax.scatter(
            split["xy"][idx, 0],
            split["xy"][idx, 1],
            s=22,
            color=COLORS[label],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=f"Qwen pred {label} ({idx.sum()})",
        )
    ax.set_title(f"{title}\nsurrogate matches Qwen pred: {split['surrogate_acc']:.3f}", weight="bold")
    ax.set_xlabel(f"PC1 of hidden states ({100 * pca.explained_variance_ratio_[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% var)")
    ax.spines[["top", "right"]].set_visible(False)


def draw_confusion(ax, split, title):
    matrix = split["confusion"]
    image = ax.imshow(matrix, cmap="Greys", vmin=0)
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            color = "white" if value > matrix.max() * 0.55 else "#111111"
            ax.text(j, i, str(value), ha="center", va="center", color=color, weight="bold")
    ax.set_xticks(range(3), [f"probe {label}" for label in LABELS])
    ax.set_yticks(range(3), [f"Qwen {label}" for label in LABELS])
    ax.set_title(title, weight="bold")
    ax.set_xlabel("estimated region label")
    ax.set_ylabel("Qwen base predicted label")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def write_metrics(path, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "n",
                "qwen_pred_A",
                "qwen_pred_B",
                "qwen_pred_C",
                "surrogate_accuracy",
                "confusion_rows_qwen_cols_probe",
            ],
        )
        writer.writeheader()
        for split in splits:
            counts = Counter(ID_TO_LABEL[int(v)] for v in split["y_pred"])
            writer.writerow({
                "split": split["payload"]["split"],
                "n": len(split["y_pred"]),
                "qwen_pred_A": counts["A"],
                "qwen_pred_B": counts["B"],
                "qwen_pred_C": counts["C"],
                "surrogate_accuracy": f"{split['surrogate_acc']:.6f}",
                "confusion_rows_qwen_cols_probe": split["confusion"].tolist(),
            })


def make_figure(args):
    train = load_hidden_cache(args.train_cache or default_cache("train", args.layer))
    val = load_hidden_cache(args.val_cache or default_cache("val", args.layer))
    test = load_hidden_cache(args.test_cache or default_cache("test", args.layer))

    scaler, probe, pca = fit_probe(train, args.c, args.class_weight, args.max_iter, args.estimator)
    train = transform_split(train, scaler, pca, probe, args.estimator)
    val = transform_split(val, scaler, pca, probe, args.estimator)
    test = transform_split(test, scaler, pca, probe, args.estimator)
    xx, yy, zz = make_grid(train["xy"], pca, probe, args.estimator, grid_n=args.grid_n)

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    for ax, split, title in [
        (axes[0, 0], train, "Train hidden states"),
        (axes[0, 1], val, "Validation hidden states"),
        (axes[1, 0], test, "Test hidden states"),
    ]:
        plot_regions(ax, xx, yy, zz)
        scatter_split(ax, split, title, pca)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    draw_confusion(axes[1, 1], test, "Test: Qwen predicted label vs estimated region")
    fig.suptitle(
        f"Estimated Qwen predicted-label regions from layer-{args.layer} hidden states ({args.estimator})",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    metrics_path = project_path(args.metrics_csv)
    write_metrics(metrics_path, [train, val, test])
    print(f"Wrote {output}")
    print(f"Wrote {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Qwen A/B/C predicted-label regions from hidden states. "
            "Default trains a PCA(hidden) -> base pred_label surrogate in the displayed 2-D plane."
        )
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--train-cache")
    parser.add_argument("--val-cache")
    parser.add_argument("--test-cache")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--estimator", choices=["pca-svm", "highdim-logreg"], default="pca-svm")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--grid-n", type=int, default=420)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output",
        default=(
            "notes/A_hidden_space_analysis/prediction_structure/"
            "qwen_predicted_regions_layer28.png"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default=(
            "notes/A_hidden_space_analysis/prediction_structure/"
            "qwen_predicted_regions_layer28_metrics.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
