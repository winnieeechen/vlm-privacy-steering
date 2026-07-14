#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


LABELS = ("A", "B", "C")
COLORS = {"A": "#d95f59", "B": "#4c78a8", "C": "#59a14f"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "outputs").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_base_hidden_states(cache_path, base_csv, layer):
    payload = torch.load(cache_path, map_location="cpu")
    activations = payload["activations"]
    if activations.ndim != 3:
        raise ValueError(f"Expected [N, layers, D] activations, got {tuple(activations.shape)}")
    if not 0 <= layer < activations.shape[1]:
        raise ValueError(f"Layer {layer} outside available range 0..{activations.shape[1] - 1}")

    rows_by_id = {row["full_id"]: row for row in read_csv(base_csv)}
    full_ids = payload["full_ids"]
    missing = [full_id for full_id in full_ids if full_id not in rows_by_id]
    if missing:
        raise ValueError(f"{len(missing)} activation IDs are absent from base CSV")
    rows = [rows_by_id[full_id] for full_id in full_ids]

    for column in ("true_label", "pred_label"):
        unknown = Counter(row.get(column, "") for row in rows if row.get(column, "") not in LABELS)
        if unknown:
            raise ValueError(f"Unknown values in {column}: {dict(unknown)}")

    return activations[:, layer, :].double().numpy(), rows, payload


def compute_metrics(x, labels):
    centroids = np.stack([x[labels == label].mean(axis=0) for label in LABELS])
    mu_a, mu_b, mu_c = centroids
    ab = mu_b - mu_a
    bc = mu_c - mu_b
    ac = mu_c - mu_a
    d_ab = np.linalg.norm(ab)
    d_bc = np.linalg.norm(bc)
    d_ac = np.linalg.norm(ac)

    denom = np.linalg.norm(ab) * np.linalg.norm(bc)
    direction_cosine = np.dot(ab, bc) / denom if denom > 0 else np.nan
    ac_norm_sq = np.dot(ac, ac)
    t = np.dot(ab, ac) / ac_norm_sq if ac_norm_sq > 0 else np.nan
    projection_b = mu_a + t * ac
    residual = np.linalg.norm(mu_b - projection_b)
    mean_adjacent_distance = (d_ab + d_bc) / 2.0
    normalized_residual = residual / mean_adjacent_distance if mean_adjacent_distance > 0 else np.nan

    centered = centroids - centroids.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    sigma1 = singular_values[0]
    sigma2 = singular_values[1]
    sigma_ratio = sigma2 / sigma1 if sigma1 > 0 else np.nan
    sigma_energy = sigma1**2 + sigma2**2
    explained_2nd = sigma2**2 / sigma_energy if sigma_energy > 0 else np.nan
    return {
        "centroids": centroids,
        "projection_b": projection_b,
        "basis": vh[:2],
        "grand_centroid": centroids.mean(axis=0),
        "d_AB": d_ab,
        "d_BC": d_bc,
        "d_AC": d_ac,
        "direction_cosine": direction_cosine,
        "projection_t": t,
        "residual": residual,
        "normalized_residual": normalized_residual,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma2_over_sigma1": sigma_ratio,
        "explained_2nd": explained_2nd,
    }


def bootstrap_metrics(x, labels, n_bootstrap, rng):
    indices = {label: np.flatnonzero(labels == label) for label in LABELS}
    draws = []
    for iteration in range(n_bootstrap):
        sampled_x = np.concatenate([
            x[rng.choice(indices[label], size=len(indices[label]), replace=True)]
            for label in LABELS
        ])
        sampled_labels = np.concatenate([
            np.full(len(indices[label]), label, dtype="U1") for label in LABELS
        ])
        metrics = compute_metrics(sampled_x, sampled_labels)
        draws.append({
            "bootstrap_iteration": iteration + 1,
            "direction_cosine": metrics["direction_cosine"],
            "normalized_residual": metrics["normalized_residual"],
            "sigma2_over_sigma1": metrics["sigma2_over_sigma1"],
        })
    return draws


def confidence_interval(draws, key):
    values = np.array([draw[key] for draw in draws], dtype=float)
    return np.nanpercentile(values, [2.5, 97.5])


def format_ci(low, high):
    return f"[{low:.3f}, {high:.3f}]"


def plot_group(ax, x, labels, metrics, title, counts, ci, c_warning_threshold):
    origin = metrics["grand_centroid"]
    basis = metrics["basis"]
    xy = (x - origin) @ basis.T
    centroid_xy = (metrics["centroids"] - origin) @ basis.T
    projection_xy = (metrics["projection_b"] - origin) @ basis.T

    for label in LABELS:
        mask = labels == label
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=18,
            color=COLORS[label],
            alpha=0.38,
            edgecolors="none",
            label=f"{label} (n={counts[label]})",
        )

    ax.plot(centroid_xy[:, 0], centroid_xy[:, 1], color="#252525", linewidth=1.5, zorder=4)
    ax.plot(
        [centroid_xy[0, 0], centroid_xy[2, 0]],
        [centroid_xy[0, 1], centroid_xy[2, 1]],
        color="#6b6b6b",
        linewidth=1.2,
        linestyle="--",
        zorder=4,
        label="A-C line",
    )
    ax.plot(
        [centroid_xy[1, 0], projection_xy[0]],
        [centroid_xy[1, 1], projection_xy[1]],
        color="#111111",
        linewidth=2.0,
        linestyle=":",
        zorder=5,
        label="B orthogonal residual",
    )
    ax.scatter(
        projection_xy[0], projection_xy[1], marker="x", s=75, color="#111111",
        linewidth=2.0, zorder=6, label="projection of B",
    )
    for i, label in enumerate(LABELS):
        ax.scatter(
            centroid_xy[i, 0], centroid_xy[i, 1], s=150, marker="X",
            color=COLORS[label], edgecolor="white", linewidth=1.2, zorder=7,
        )
        ax.annotate(
            rf"$\mu_{label}$", centroid_xy[i], xytext=(7, 7), textcoords="offset points",
            fontsize=11, weight="bold",
        )

    summary = (
        f"cos = {metrics['direction_cosine']:.3f}  "
        f"95% CI {format_ci(*ci['direction_cosine'])}\n"
        f"normalized residual = {metrics['normalized_residual']:.3f}  "
        f"95% CI {format_ci(*ci['normalized_residual'])}\n"
        f"sigma2/sigma1 = {metrics['sigma2_over_sigma1']:.3f}  "
        f"95% CI {format_ci(*ci['sigma2_over_sigma1'])}"
    )
    ax.text(
        0.02, 0.98, summary, transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.9, "pad": 6},
    )
    if counts["C"] < c_warning_threshold:
        ax.text(
            0.98, 0.02, f"WARNING: C centroid is unstable (n={counts['C']})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color="#9c2f2f", weight="bold",
            bbox={"facecolor": "#fff4f2", "edgecolor": "#d95f59", "alpha": 0.95, "pad": 5},
        )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("centroid SVD direction 1")
    ax.set_ylabel("centroid SVD direction 2")
    ax.axhline(0, color="#dddddd", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92, ncol=2)


def write_metrics(path, layer, x, analyses, c_warning_threshold):
    fields = [
        "layer", "grouping", "n", "n_A", "n_B", "n_C", "hidden_dim",
        "d_AB", "d_BC", "d_AC", "direction_cosine", "direction_cosine_ci_low",
        "direction_cosine_ci_high", "projection_t", "residual", "normalized_residual",
        "normalized_residual_ci_low", "normalized_residual_ci_high", "sigma1", "sigma2",
        "sigma2_over_sigma1", "sigma2_over_sigma1_ci_low", "sigma2_over_sigma1_ci_high",
        "explained_2nd",
        "predicted_C_centroid_warning",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grouping, analysis in analyses.items():
            m, counts, ci = analysis["metrics"], analysis["counts"], analysis["ci"]
            writer.writerow({
                "layer": layer,
                "grouping": grouping,
                "n": len(x),
                "n_A": counts["A"],
                "n_B": counts["B"],
                "n_C": counts["C"],
                "hidden_dim": x.shape[1],
                "d_AB": f"{m['d_AB']:.9g}",
                "d_BC": f"{m['d_BC']:.9g}",
                "d_AC": f"{m['d_AC']:.9g}",
                "direction_cosine": f"{m['direction_cosine']:.9g}",
                "direction_cosine_ci_low": f"{ci['direction_cosine'][0]:.9g}",
                "direction_cosine_ci_high": f"{ci['direction_cosine'][1]:.9g}",
                "projection_t": f"{m['projection_t']:.9g}",
                "residual": f"{m['residual']:.9g}",
                "normalized_residual": f"{m['normalized_residual']:.9g}",
                "normalized_residual_ci_low": f"{ci['normalized_residual'][0]:.9g}",
                "normalized_residual_ci_high": f"{ci['normalized_residual'][1]:.9g}",
                "sigma1": f"{m['sigma1']:.9g}",
                "sigma2": f"{m['sigma2']:.9g}",
                "sigma2_over_sigma1": f"{m['sigma2_over_sigma1']:.9g}",
                "sigma2_over_sigma1_ci_low": f"{ci['sigma2_over_sigma1'][0]:.9g}",
                "sigma2_over_sigma1_ci_high": f"{ci['sigma2_over_sigma1'][1]:.9g}",
                "explained_2nd": f"{m['explained_2nd']:.9g}",
                "predicted_C_centroid_warning": (
                    grouping == "base_predicted_label" and counts["C"] < c_warning_threshold
                ),
            })


def write_bootstrap(path, layer, analyses):
    fields = [
        "layer", "grouping", "bootstrap_iteration", "direction_cosine",
        "normalized_residual", "sigma2_over_sigma1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grouping, analysis in analyses.items():
            for draw in analysis["draws"]:
                writer.writerow({"layer": layer, "grouping": grouping, **draw})


def write_bootstrap_ci(path, layer, analyses):
    fields = ["layer", "grouping", "metric", "estimate", "ci_low", "ci_high", "n_bootstrap"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grouping, analysis in analyses.items():
            for metric in ("direction_cosine", "normalized_residual", "sigma2_over_sigma1"):
                writer.writerow({
                    "layer": layer,
                    "grouping": grouping,
                    "metric": metric,
                    "estimate": analysis["metrics"][metric],
                    "ci_low": analysis["ci"][metric][0],
                    "ci_high": analysis["ci"][metric][1],
                    "n_bootstrap": len(analysis["draws"]),
                })


def save_group_figure(path, x, labels, analysis, title, layer, c_warning_threshold, dpi):
    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    plot_group(
        ax, x, labels, analysis["metrics"], title, analysis["counts"],
        analysis["ci"], c_warning_threshold,
    )
    fig.suptitle(
        f"Layer-{layer} centroid geometry in original hidden space\n"
        "Display plane is defined only by the three high-dimensional centroids",
        fontsize=14, weight="bold",
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(args):
    cache_path = project_path(args.activation_cache)
    base_csv = project_path(args.base_csv)
    x, rows, payload = load_base_hidden_states(cache_path, base_csv, args.layer)
    label_sets = {
        "true_label": np.array([row["true_label"] for row in rows]),
        "base_predicted_label": np.array([row["pred_label"] for row in rows]),
    }

    analyses = {}
    seed_sequence = np.random.SeedSequence(args.seed).spawn(len(label_sets))
    for (grouping, labels), child_seed in zip(label_sets.items(), seed_sequence):
        counts = Counter(labels)
        if any(counts[label] == 0 for label in LABELS):
            raise ValueError(f"Cannot analyze {grouping}; class counts are {dict(counts)}")
        metrics = compute_metrics(x, labels)
        draws = bootstrap_metrics(x, labels, args.n_bootstrap, np.random.default_rng(child_seed))
        ci = {
            key: confidence_interval(draws, key)
            for key in ("direction_cosine", "normalized_residual", "sigma2_over_sigma1")
        }
        analyses[grouping] = {"metrics": metrics, "draws": draws, "ci": ci, "counts": counts}

    true_output = project_path(args.true_output)
    pred_output = project_path(args.pred_output)
    true_output.parent.mkdir(parents=True, exist_ok=True)
    save_group_figure(
        true_output, x, label_sets["true_label"], analyses["true_label"],
        "Grouped by true label", args.layer, args.c_warning_threshold, args.dpi,
    )
    save_group_figure(
        pred_output, x, label_sets["base_predicted_label"], analyses["base_predicted_label"],
        "Grouped by base predicted label", args.layer, args.c_warning_threshold, args.dpi,
    )

    metrics_csv = project_path(args.metrics_csv)
    bootstrap_csv = project_path(args.bootstrap_csv)
    bootstrap_draws_csv = project_path(args.bootstrap_draws_csv)
    write_metrics(metrics_csv, args.layer, x, analyses, args.c_warning_threshold)
    write_bootstrap_ci(bootstrap_csv, args.layer, analyses)
    write_bootstrap(bootstrap_draws_csv, args.layer, analyses)

    print(f"Model: {payload.get('model_name', 'unknown')}")
    print(f"Base hidden states: {tuple(x.shape)}, layer={args.layer}")
    for grouping, analysis in analyses.items():
        m, counts, ci = analysis["metrics"], analysis["counts"], analysis["ci"]
        print(f"\n{grouping}: counts={dict(counts)}")
        print(f"  d_AB={m['d_AB']:.6f} d_BC={m['d_BC']:.6f} d_AC={m['d_AC']:.6f}")
        print(f"  direction_cosine={m['direction_cosine']:.6f} CI={format_ci(*ci['direction_cosine'])}")
        print(f"  projection_t={m['projection_t']:.6f} residual={m['residual']:.6f}")
        print(f"  normalized_residual={m['normalized_residual']:.6f} CI={format_ci(*ci['normalized_residual'])}")
        print(f"  sigma1={m['sigma1']:.6f} sigma2={m['sigma2']:.6f}")
        print(f"  sigma2/sigma1={m['sigma2_over_sigma1']:.6f} CI={format_ci(*ci['sigma2_over_sigma1'])}")
    print(f"\nWrote {true_output}")
    print(f"Wrote {pred_output}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {bootstrap_csv}")
    print(f"Wrote {bootstrap_draws_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether A/B/C centroids are collinear in original Qwen hidden space."
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250714)
    parser.add_argument("--c-warning-threshold", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--activation-cache",
        default=(
            "outputs/04_low_rank_discriminant_vectors/02_over/cache/"
            "condition_train_last_token_activations.pt"
        ),
    )
    parser.add_argument(
        "--base-csv", default="outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv"
    )
    parser.add_argument(
        "--true-output",
        default="notes/A_hidden_space_analysis/centroid_geometry/geometry_true_labels.png",
    )
    parser.add_argument(
        "--pred-output",
        default="notes/A_hidden_space_analysis/centroid_geometry/geometry_predicted_labels.png",
    )
    parser.add_argument(
        "--metrics-csv",
        default="notes/A_hidden_space_analysis/centroid_geometry/centroid_geometry_by_layer.csv",
    )
    parser.add_argument(
        "--bootstrap-csv",
        default="notes/A_hidden_space_analysis/centroid_geometry/bootstrap_geometry_ci.csv",
    )
    parser.add_argument(
        "--bootstrap-draws-csv",
        default=(
            "notes/A_hidden_space_analysis/centroid_geometry/"
            "bootstrap_geometry_draws_layer28.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
