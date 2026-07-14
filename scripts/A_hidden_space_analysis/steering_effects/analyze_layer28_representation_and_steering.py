#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


LABELS = ("A", "B", "C")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
COLORS = {
    "helpful_over": "#2a9d8f",
    "helpful_under": "#4c78a8",
    "harmful": "#d95f59",
    "correct_to_correct": "#6a994e",
    "wrong_to_wrong": "#8d99ae",
}
GROUPS = tuple(COLORS)


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


def index_unique(rows, key, name):
    values = [row[key] for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {key} values in {name}: {duplicates[:10]}")
    return {row[key]: row for row in rows}


def align_effects(full_ids, transition_csv):
    rows = read_csv(transition_csv)
    by_id = index_unique(rows, "sample_id", transition_csv)
    missing = sorted(set(full_ids) - set(by_id))
    extra = sorted(set(by_id) - set(full_ids))
    if missing or extra:
        raise ValueError(f"Transition/cache ID mismatch: missing={missing[:10]}, extra={extra[:10]}")
    aligned = [by_id[sample_id] for sample_id in full_ids]
    groups = []
    for row in aligned:
        effect = row["effect_group"]
        if effect in ("harmful_over", "harmful_under"):
            effect = "harmful"
        groups.append(effect)
    return aligned, np.array(groups)


def load_train(cache_path, base_csv):
    payload = torch.load(cache_path, map_location="cpu")
    rows = read_csv(base_csv)
    by_id = index_unique(rows, "full_id", base_csv)
    full_ids = payload["full_ids"]
    missing = [sample_id for sample_id in full_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Train cache IDs missing from CSV: {missing[:10]}")
    aligned = [by_id[sample_id] for sample_id in full_ids]
    return payload["activations"].float().numpy(), aligned


def labels_from_rows(rows, target):
    key = "true_label" if target == "true_label" else "pred_label"
    return np.array([LABEL_TO_ID[row[key]] for row in rows], dtype=int)


def labels_from_bundle(bundle, target):
    key = "true_labels" if target == "true_label" else "base_pred_labels"
    return np.array([LABEL_TO_ID[label] for label in bundle[key]], dtype=int)


def metric_dict(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    }


def threshold_candidates(scores, max_points=100):
    unique = np.unique(scores)
    if len(unique) > max_points:
        unique = np.unique(np.quantile(unique, np.linspace(0.01, 0.99, max_points)))
    if len(unique) < 2:
        return unique
    return (unique[:-1] + unique[1:]) / 2


def predict_threshold(scores, low, high):
    return np.where(scores < low, 0, np.where(scores < high, 1, 2))


def best_thresholds(scores, y):
    candidates = threshold_candidates(scores)
    best = None
    for i, low in enumerate(candidates[:-1]):
        for high in candidates[i + 1:]:
            pred = predict_threshold(scores, low, high)
            score = balanced_accuracy_score(y, pred)
            macro = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
            key = (score, macro)
            if best is None or key > best[0]:
                best = (key, low, high)
    if best is None:
        raise RuntimeError("Not enough distinct 1-D scores to choose two thresholds")
    return best[1], best[2], best[0]


def fit_ordinal_1d(x_train, y_train, x_val, y_val):
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    all_classes_in_val = set(y_val) == {0, 1, 2}
    selection_name = "validation" if all_classes_in_val else "train_plus_validation_missing_val_class"
    best = None
    for alpha in candidates:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train.astype(float))
        if all_classes_in_val:
            scores = model.predict(x_val)
            labels = y_val
        else:
            scores = np.concatenate([model.predict(x_train), model.predict(x_val)])
            labels = np.concatenate([y_train, y_val])
        low, high, objective = best_thresholds(scores, labels)
        if best is None or objective > best[0]:
            best = (objective, model, low, high, alpha)
    return best[1], best[2], best[3], {
        "ridge_alpha": best[4],
        "threshold_low": best[2],
        "threshold_high": best[3],
        "selection_data": selection_name,
    }


def fit_logistic(x_train, y_train, x_val, y_val):
    best = None
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=28,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        objective = (
            balanced_accuracy_score(y_val, pred),
            f1_score(y_val, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        )
        if best is None or objective > best[0]:
            best = (objective, model, c)
    return best[1], best[2]


def compare_models(x_train, y_train, x_val, y_val, x_test, y_test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    val_scaled = scaler.transform(x_val)
    test_scaled = scaler.transform(x_test)

    ordinal, low, high, ordinal_info = fit_ordinal_1d(train_scaled, y_train, val_scaled, y_val)
    ordinal_pred = predict_threshold(ordinal.predict(test_scaled), low, high)

    highdim, c_highdim = fit_logistic(train_scaled, y_train, val_scaled, y_val)
    pred_highdim = highdim.predict(test_scaled)

    centered_weights = highdim.coef_ - highdim.coef_.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_weights, full_matrices=False)
    supervised_basis = vh[:2]
    train_2d = train_scaled @ supervised_basis.T
    val_2d = val_scaled @ supervised_basis.T
    test_2d = test_scaled @ supervised_basis.T
    model_2d, c_2d = fit_logistic(train_2d, y_train, val_2d, y_val)
    pred_2d = model_2d.predict(test_2d)
    rank2_energy = float(
        np.sum(singular_values[:2] ** 2) / np.sum(singular_values ** 2)
    )

    return {
        "ordinal_1d": {**metric_dict(y_test, ordinal_pred), **ordinal_info},
        "supervised_rank2": {
            **metric_dict(y_test, pred_2d), "C": c_2d,
            "rank2_weight_energy": rank2_energy,
        },
        "highdim_linear": {**metric_dict(y_test, pred_highdim), "C": c_highdim},
    }


def plot_confusion(ax, matrix, title):
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    cutoff = matrix.max() * 0.55
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(matrix[i, j]), ha="center", va="center", weight="bold",
                color="white" if matrix[i, j] > cutoff else "#111111",
            )
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("prediction")
    ax.set_ylabel("target")
    ax.set_title(title, weight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_model_figure(path, target, results, dpi):
    names = list(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1")
    x = np.arange(len(names))
    width = 0.24
    for i, metric in enumerate(metric_names):
        axes[0, 0].bar(
            x + (i - 1) * width, [results[name][metric] for name in names], width,
            label=metric.replace("_", " "),
        )
    axes[0, 0].set_xticks(x, [name.replace("_", "\n") for name in names])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("test score")
    axes[0, 0].set_title("Test performance", weight="bold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].spines[["top", "right"]].set_visible(False)
    for ax, name in zip((axes[0, 1], axes[1, 0], axes[1, 1]), names):
        plot_confusion(ax, results[name]["confusion"], name.replace("_", " "))
    fig.suptitle(f"Layer-28: 1-D vs 2-D vs high-dimensional model ({target})", weight="bold")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_model_metrics(path, all_results):
    fields = [
        "target", "model", "accuracy", "balanced_accuracy", "macro_f1", "confusion",
        "ridge_alpha", "threshold_low", "threshold_high", "selection_data", "C",
        "rank2_weight_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for target, results in all_results.items():
            for model_name, result in results.items():
                writer.writerow({
                    "target": target,
                    "model": model_name,
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "macro_f1": result["macro_f1"],
                    "confusion": result["confusion"].tolist(),
                    "ridge_alpha": result.get("ridge_alpha", ""),
                    "threshold_low": result.get("threshold_low", ""),
                    "threshold_high": result.get("threshold_high", ""),
                    "selection_data": result.get("selection_data", ""),
                    "C": result.get("C", ""),
                    "rank2_weight_energy": result.get("rank2_weight_energy", ""),
                })


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)) if denom > 0 else np.nan


def analyze_deltas(bundle, groups, output_dir, dpi):
    layers = bundle["recorded_layers"]
    delta = (bundle["steered_hidden_states"] - bundle["base_hidden_states"]).float().numpy()
    summaries = []
    mean_vectors = {}
    for layer_index, layer in enumerate(layers):
        for group in GROUPS:
            mask = groups == group
            vectors = delta[mask, layer_index]
            mean_vector = vectors.mean(axis=0)
            mean_vectors[(layer, group)] = mean_vector
            summaries.append({
                "layer": layer,
                "effect_group": group,
                "n": int(mask.sum()),
                "mean_sample_delta_norm": float(np.linalg.norm(vectors, axis=1).mean()),
                "sd_sample_delta_norm": float(np.linalg.norm(vectors, axis=1).std(ddof=1)),
                "mean_delta_vector_norm": float(np.linalg.norm(mean_vector)),
                "mean_distance_from_layer_group_mean": float(
                    np.linalg.norm(vectors - mean_vector, axis=1).mean()
                ),
            })
    with (output_dir / "downstream_delta_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    pair_rows = []
    for layer in layers:
        for first, second in combinations(GROUPS, 2):
            pair_rows.append({
                "layer": layer,
                "group_1": first,
                "group_2": second,
                "mean_delta_cosine": cosine(mean_vectors[(layer, first)], mean_vectors[(layer, second)]),
            })
    with (output_dir / "delta_pairwise_cosine_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plotted_values = []
    for first, second in combinations(GROUPS, 2):
        values = [cosine(mean_vectors[(layer, first)], mean_vectors[(layer, second)]) for layer in layers]
        plotted_values.extend(values)
        highlight = {first, second} == {"helpful_over", "helpful_under"}
        ax.plot(
            layers, values, marker="o" if highlight else None,
            linewidth=3 if highlight else 1, alpha=1 if highlight else 0.3,
            color="#111111" if highlight else "#7f8c8d",
            label="helpful over vs helpful under" if highlight else None,
        )
    ax.axhline(1, color="#bbbbbb", linestyle="--", linewidth=1)
    lower = max(-1.0, min(plotted_values) - 0.002)
    ax.set_ylim(lower, 1.0005)
    ax.set_xlabel("layer after layer-28 intervention")
    ax.set_ylabel("cosine between group mean delta vectors")
    ax.set_title("Do identical layer-28 interventions diverge downstream?", weight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "delta_cosine_by_layer.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for group in GROUPS:
        rows = [row for row in summaries if row["effect_group"] == group]
        ax.plot(
            layers, [row["mean_sample_delta_norm"] for row in rows], marker="o",
            color=COLORS[group], label=group.replace("_", " "),
        )
    ax.set_xlabel("layer after layer-28 intervention")
    ax.set_ylabel("mean per-sample delta norm")
    ax.set_title("Downstream delta magnitude by actual steering effect", weight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "delta_norm_by_effect_group.png", dpi=dpi)
    plt.close(fig)
    return delta, mean_vectors


def analyze_margins(bundle, transition_rows, groups, output_dir, dpi):
    true_ids = np.array([LABEL_TO_ID[label] for label in bundle["true_labels"]])
    base_scores = bundle["base_label_logits"].float().numpy()
    steered_scores = bundle["steered_label_logits"].float().numpy()

    def margins(scores):
        result = []
        for i, true_id in enumerate(true_ids):
            others = np.delete(scores[i], true_id)
            result.append(scores[i, true_id] - others.max())
        return np.array(result)

    base_margin = margins(base_scores)
    steered_margin = margins(steered_scores)
    delta_margin = steered_margin - base_margin
    score_delta = steered_scores - base_scores
    records = []
    for i, row in enumerate(transition_rows):
        broad_group = groups[i]
        margin_group = (
            broad_group if broad_group in ("helpful_over", "helpful_under", "harmful") else "neutral"
        )
        records.append({
            "sample_id": row["sample_id"],
            "true_label": row["true_label"],
            "base_pred": row["base_pred"],
            "steered_pred": row["steered_pred"],
            "effect_group": row["effect_group"],
            "margin_group": margin_group,
            "base_score_A": base_scores[i, 0],
            "base_score_B": base_scores[i, 1],
            "base_score_C": base_scores[i, 2],
            "steered_score_A": steered_scores[i, 0],
            "steered_score_B": steered_scores[i, 1],
            "steered_score_C": steered_scores[i, 2],
            "base_true_margin": base_margin[i],
            "steered_true_margin": steered_margin[i],
            "delta_true_margin": delta_margin[i],
        })
    with (output_dir / "output_margin_changes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    margin_groups = ("helpful_over", "helpful_under", "harmful", "neutral")
    values = [delta_margin[np.array([record["margin_group"] == group for record in records])] for group in margin_groups]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    boxes = ax.boxplot(values, patch_artist=True, showfliers=False)
    box_colors = (COLORS["helpful_over"], COLORS["helpful_under"], COLORS["harmful"], "#8d99ae")
    for box, color in zip(boxes["boxes"], box_colors):
        box.set_facecolor(color)
        box.set_alpha(0.55)
    rng = np.random.default_rng(28)
    for i, (values_i, color) in enumerate(zip(values, box_colors), 1):
        ax.scatter(
            i + rng.uniform(-0.16, 0.16, len(values_i)), values_i,
            s=18, alpha=0.55, color=color, edgecolors="none",
        )
    ax.axhline(0, color="#222222", linestyle="--")
    ax.set_xticks(
        range(1, 5), [f"{group.replace('_', chr(10))}\n(n={len(v)})" for group, v in zip(margin_groups, values)]
    )
    ax.set_ylabel("steered true-label margin - base true-label margin")
    ax.set_title("True-label margin change from actual layer-28 steering", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "true_margin_change_by_effect.png", dpi=dpi)
    plt.close(fig)

    transitions = sorted(set((row["base_pred"], row["steered_pred"]) for row in transition_rows))
    matrix = np.zeros((len(transitions), 3))
    counts = []
    for i, transition in enumerate(transitions):
        mask = np.array([
            (row["base_pred"], row["steered_pred"]) == transition for row in transition_rows
        ])
        matrix[i] = score_delta[mask].mean(axis=0)
        counts.append(int(mask.sum()))
    scale = np.abs(matrix).max()
    fig, ax = plt.subplots(figsize=(8, max(4, 1.05 * len(transitions))), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    for i in range(len(transitions)):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), [f"delta score {label}" for label in LABELS])
    ax.set_yticks(
        range(len(transitions)),
        [f"{first}->{second} (n={n})" for (first, second), n in zip(transitions, counts)],
    )
    ax.set_title("Mean A/B/C label-score changes by actual prediction transition", weight="bold")
    plt.colorbar(image, ax=ax, label="steered logit - base logit")
    fig.savefig(output_dir / "label_score_change_by_transition.png", dpi=dpi)
    plt.close(fig)
    return records


def plot_movements(train_all, bundle, groups, output_dir, dpi):
    layers_to_plot = [28, 29, 32, 35]
    recorded = bundle["recorded_layers"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for ax, layer in zip(axes.flat, layers_to_plot):
        layer_index = recorded.index(layer)
        pca = PCA(n_components=2, random_state=28)
        pca.fit(train_all[:, layer, :])
        base_xy = pca.transform(bundle["base_hidden_states"][:, layer_index, :].float().numpy())
        steered_xy = pca.transform(bundle["steered_hidden_states"][:, layer_index, :].float().numpy())
        for group in GROUPS:
            mask = groups == group
            delta_xy = steered_xy[mask] - base_xy[mask]
            ax.scatter(
                base_xy[mask, 0], base_xy[mask, 1], s=13, color=COLORS[group], alpha=0.28,
                edgecolors="none",
            )
            ax.quiver(
                base_xy[mask, 0], base_xy[mask, 1], delta_xy[:, 0], delta_xy[:, 1],
                angles="xy", scale_units="xy", scale=1, color=COLORS[group], alpha=0.22,
                width=0.002,
            )
            base_mean = base_xy[mask].mean(axis=0)
            steer_mean = steered_xy[mask].mean(axis=0)
            ax.annotate(
                "", xy=steer_mean, xytext=base_mean,
                arrowprops={"arrowstyle": "->", "color": COLORS[group], "lw": 3},
            )
        ax.set_title(
            f"Layer {layer} (train PCA variance={100*pca.explained_variance_ratio_.sum():.1f}%)",
            weight="bold",
        )
        ax.set_xlabel("train-base PC1")
        ax.set_ylabel("train-base PC2")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=COLORS[group], lw=3, label=group.replace("_", " "))
        for group in GROUPS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    fig.suptitle(
        "Propagation of the same layer-28 intervention through downstream layers\n"
        "PCA is fit separately on train base states; arrows are visualization only",
        weight="bold",
    )
    fig.savefig(output_dir / "downstream_movement_from_layer28.png", dpi=dpi)
    plt.close(fig)


def main(args):
    output_root = project_path(args.output_root)
    decoding_dir = output_root / "representation_decoding"
    dynamics_dir = output_root / "downstream_dynamics"
    margins_dir = output_root / "output_margins"
    for path in (decoding_dir, dynamics_dir, margins_dir):
        path.mkdir(parents=True, exist_ok=True)
    train_all, train_rows = load_train(project_path(args.train_cache), project_path(args.train_csv))
    val = torch.load(project_path(args.val_cache), map_location="cpu")
    test = torch.load(project_path(args.test_cache), map_location="cpu")
    if not val.get("complete") or not test.get("complete"):
        raise RuntimeError("Validation/test analysis cache is incomplete")
    if val["recorded_layers"] != test["recorded_layers"]:
        raise ValueError("Validation and test recorded-layer metadata differ")
    layer_index = test["recorded_layers"].index(args.layer)

    transition_rows, groups = align_effects(
        test["full_ids"], project_path(args.transition_csv)
    )
    for i, row in enumerate(transition_rows):
        if row["true_label"] != test["true_labels"][i]:
            raise ValueError(f"True-label mismatch for {row['sample_id']}")
        if row["base_pred"] != test["base_pred_labels"][i]:
            raise ValueError(f"Base-prediction mismatch for {row['sample_id']}")
        if row["steered_pred"] != test["steered_pred_labels"][i]:
            raise ValueError(f"Steered-prediction mismatch for {row['sample_id']}")

    x_train = train_all[:, args.layer, :]
    x_val = val["base_hidden_states"][:, layer_index, :].float().numpy()
    x_test = test["base_hidden_states"][:, layer_index, :].float().numpy()
    all_results = {}
    for target in ("true_label", "base_predicted_label"):
        y_train = labels_from_rows(train_rows, target)
        y_val = labels_from_bundle(val, target)
        y_test = labels_from_bundle(test, target)
        results = compare_models(x_train, y_train, x_val, y_val, x_test, y_test)
        all_results[target] = results
        save_model_figure(
            decoding_dir / f"model_comparison_{target}_layer28.png", target, results, args.dpi
        )
    write_model_metrics(decoding_dir / "model_comparison_layer28.csv", all_results)

    _, mean_vectors = analyze_deltas(test, groups, dynamics_dir, args.dpi)
    analyze_margins(test, transition_rows, groups, margins_dir, args.dpi)
    plot_movements(train_all, test, groups, dynamics_dir, args.dpi)

    label_argmax_base = np.argmax(test["base_label_logits"].numpy(), axis=1)
    label_argmax_steered = np.argmax(test["steered_label_logits"].numpy(), axis=1)
    actual_base = labels_from_bundle(test, "base_predicted_label")
    actual_steered = np.array([LABEL_TO_ID[label] for label in test["steered_pred_labels"]])
    print("Aligned test samples:", len(test["full_ids"]))
    print("Base label-logit argmax agreement:", np.mean(label_argmax_base == actual_base))
    print("Steered label-logit argmax agreement:", np.mean(label_argmax_steered == actual_steered))
    print("\nModel comparison:")
    for target, results in all_results.items():
        print(target)
        for name, result in results.items():
            print(
                f"  {name}: accuracy={result['accuracy']:.4f} "
                f"balanced={result['balanced_accuracy']:.4f} macro_f1={result['macro_f1']:.4f}"
            )
    print("\nHelpful-over vs helpful-under mean-delta cosine:")
    for layer in test["recorded_layers"]:
        print(
            f"  layer {layer}: "
            f"{cosine(mean_vectors[(layer, 'helpful_over')], mean_vectors[(layer, 'helpful_under')]):.6f}"
        )
    print("Wrote representation decoding to", decoding_dir)
    print("Wrote downstream dynamics to", dynamics_dir)
    print("Wrote output-margin analysis to", margins_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Parts 3-6 for the layer-28 alpha-14 intervention.")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument(
        "--train-cache",
        default=(
            "outputs/04_low_rank_discriminant_vectors/02_over/cache/"
            "condition_train_last_token_activations.pt"
        ),
    )
    parser.add_argument(
        "--train-csv", default="outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv"
    )
    parser.add_argument(
        "--val-cache",
        default="outputs/A_hidden_space_analysis/cache/val_base_prompt_states_intervention_layer28.pt",
    )
    parser.add_argument(
        "--test-cache",
        default=(
            "outputs/A_hidden_space_analysis/cache/"
            "test_base_steered_alpha14_prompt_states_intervention_layer28.pt"
        ),
    )
    parser.add_argument(
        "--transition-csv",
        default="notes/A_hidden_space_analysis/steering_transitions/per_sample_transitions.csv",
    )
    parser.add_argument(
        "--output-root", default="notes/A_hidden_space_analysis"
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
