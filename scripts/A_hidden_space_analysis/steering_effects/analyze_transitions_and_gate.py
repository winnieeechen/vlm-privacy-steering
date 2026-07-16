#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = ("A", "B", "C")
RANK = {label: i for i, label in enumerate(LABELS)}
STATUS_ORDER = ("correct", "over", "under")
EFFECT_ORDER = (
    "helpful_over",
    "helpful_under",
    "harmful_over",
    "harmful_under",
    "wrong_to_wrong",
    "correct_to_correct",
)
EFFECT_COLORS = {
    "helpful_over": "#2a9d8f",
    "helpful_under": "#4c78a8",
    "harmful_over": "#e76f51",
    "harmful_under": "#d95f59",
    "wrong_to_wrong": "#8d99ae",
    "correct_to_correct": "#6a994e",
}
EXPECTED_BASE = {"correct": 87, "over": 44, "under": 112}
EXPECTED_STEERED = {"correct": 147, "over": 14, "under": 82}


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


def index_unique(rows, path):
    ids = [row.get("full_id", "") for row in rows]
    missing_ids = [i for i, sample_id in enumerate(ids) if not sample_id]
    duplicates = sorted(sample_id for sample_id, n in Counter(ids).items() if n > 1)
    if missing_ids or duplicates:
        raise ValueError(
            f"Invalid sample IDs in {path}: missing rows={missing_ids[:10]}, "
            f"duplicate IDs={duplicates[:10]}"
        )
    return {row["full_id"]: row for row in rows}


def require_same_ids(named_indices):
    names = list(named_indices)
    reference_name = names[0]
    reference_ids = set(named_indices[reference_name])
    for name in names[1:]:
        ids = set(named_indices[name])
        missing = sorted(reference_ids - ids)
        extra = sorted(ids - reference_ids)
        if missing or extra:
            raise ValueError(
                f"sample_id mismatch: {reference_name} vs {name}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )


def status(true_label, pred_label):
    if true_label == pred_label:
        return "correct"
    return "over" if RANK[pred_label] > RANK[true_label] else "under"


def effect_group(base_status, steered_status):
    if base_status == "correct" and steered_status == "correct":
        return "correct_to_correct"
    if base_status == "over" and steered_status == "correct":
        return "helpful_over"
    if base_status == "under" and steered_status == "correct":
        return "helpful_under"
    if base_status == "correct" and steered_status == "over":
        return "harmful_over"
    if base_status == "correct" and steered_status == "under":
        return "harmful_under"
    return "wrong_to_wrong"


def validate_label(label, field, sample_id, source):
    if label not in LABELS:
        raise ValueError(f"Invalid {field}={label!r} for {sample_id} in {source}")


def build_records(base_path, steered_path, condition_path):
    base = index_unique(read_csv(base_path), base_path)
    steered = index_unique(read_csv(steered_path), steered_path)
    condition = index_unique(read_csv(condition_path), condition_path)
    require_same_ids({"base": base, "steered": steered, "condition": condition})

    records = []
    for sample_id in sorted(base):
        b, s, c = base[sample_id], steered[sample_id], condition[sample_id]
        true_label = b["true_label"]
        base_pred = b["pred_label"]
        steered_pred = s["steered_pred_label"]
        for value, field, source in (
            (true_label, "true_label", "base"),
            (base_pred, "base_pred", "base"),
            (steered_pred, "steered_pred", "steered"),
        ):
            validate_label(value, field, sample_id, source)

        for field in ("true_label", "pred_label"):
            if c[field] != b[field]:
                raise ValueError(
                    f"Label mismatch for {sample_id}: base {field}={b[field]!r}, "
                    f"condition {field}={c[field]!r}"
                )
        if s["true_label"] != true_label or s["pred_label"] != base_pred:
            raise ValueError(f"Base-label columns disagree in steered file for {sample_id}")

        base_status = status(true_label, base_pred)
        steered_status = status(true_label, steered_pred)
        records.append({
            "sample_id": sample_id,
            "true_label": true_label,
            "base_pred": base_pred,
            "steered_pred": steered_pred,
            "base_case": f"{true_label}_to_{base_pred}",
            "steered_case": f"{true_label}_to_{steered_pred}",
            "base_status": base_status,
            "steered_status": steered_status,
            "effect_group": effect_group(base_status, steered_status),
            "prediction_changed": base_pred != steered_pred,
            "condition_score": float(c["condition_score"]),
            "condition_threshold": float(c["condition_threshold"]),
            "condition_gate": c["condition_gate"].strip().lower() == "true",
        })
    return records


def assert_expected_totals(records):
    base_counts = Counter(row["base_status"] for row in records)
    steered_counts = Counter(row["steered_status"] for row in records)
    if dict(base_counts) != EXPECTED_BASE:
        raise RuntimeError(f"Base totals do not reproduce expected values: {dict(base_counts)}")
    if dict(steered_counts) != EXPECTED_STEERED:
        raise RuntimeError(f"Steered totals do not reproduce expected values: {dict(steered_counts)}")


def prediction_matrix(records):
    matrix = np.zeros((3, 3), dtype=int)
    for row in records:
        matrix[RANK[row["base_pred"]], RANK[row["steered_pred"]]] += 1
    return matrix


def true_base_matrix(records):
    matrix = np.zeros((9, 3), dtype=int)
    for row in records:
        row_index = 3 * RANK[row["true_label"]] + RANK[row["base_pred"]]
        matrix[row_index, RANK[row["steered_pred"]]] += 1
    return matrix


def annotate_matrix(ax, matrix, row_labels, column_labels, title, cmap="Blues"):
    image = ax.imshow(matrix, cmap=cmap, vmin=0)
    cutoff = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j, i, str(value), ha="center", va="center", weight="bold",
                color="white" if value > cutoff else "#111111",
            )
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("10b unconditional steered prediction")
    ax.set_ylabel("base prediction" if len(row_labels) == 3 else "true label -> base prediction")
    ax.set_title(title, weight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_transition_figures(records, output_dir, dpi):
    matrix = prediction_matrix(records)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    annotate_matrix(
        ax, matrix, [f"base {label}" for label in LABELS],
        [f"steered {label}" for label in LABELS],
        "Layer-28 alpha-14: base to steered predictions",
    )
    fig.savefig(output_dir / "transition_matrix_base_to_steered.png", dpi=dpi)
    plt.close(fig)

    matrix9 = true_base_matrix(records)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
    vmax = max(1, int(matrix9.max()))
    cutoff = vmax * 0.55
    image = None
    for true_index, (true_label, ax) in enumerate(zip(LABELS, axes)):
        panel = matrix9[3 * true_index:3 * (true_index + 1)]
        image = ax.imshow(panel, cmap="Purples", vmin=0, vmax=vmax)
        for base_index in range(3):
            for steered_index in range(3):
                value = panel[base_index, steered_index]
                ax.text(
                    steered_index,
                    base_index,
                    str(value),
                    ha="center",
                    va="center",
                    weight="bold",
                    color="white" if value > cutoff else "#111111",
                )
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_title(f"True label: {true_label}", weight="bold")

    axes[0].set_ylabel("Base prediction")
    axes[1].set_xlabel("10b unconditional steered prediction")
    fig.suptitle(
        "True label -> base prediction -> steered prediction",
        weight="bold",
    )
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025, label="Test samples")
    fig.savefig(output_dir / "true_base_to_steered_matrix.png", dpi=dpi)
    plt.close(fig)

    counts = Counter(row["effect_group"] for row in records)
    values = [counts[group] for group in EFFECT_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    bars = ax.bar(
        range(len(EFFECT_ORDER)), values,
        color=[EFFECT_COLORS[group] for group in EFFECT_ORDER], width=0.72,
    )
    ax.bar_label(bars, padding=4, weight="bold")
    ax.set_xticks(range(len(EFFECT_ORDER)), [name.replace("_", "\n") for name in EFFECT_ORDER])
    ax.set_ylabel("test samples")
    ax.set_title("Actual effects of the layer-28 alpha-14 unconditional over vector", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.50, 0.97,
        f"Base: correct={EXPECTED_BASE['correct']}, over={EXPECTED_BASE['over']}, under={EXPECTED_BASE['under']}\n"
        f"Steered: correct={EXPECTED_STEERED['correct']}, over={EXPECTED_STEERED['over']}, under={EXPECTED_STEERED['under']}",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92, "pad": 6},
    )
    fig.savefig(output_dir / "steering_effect_counts.png", dpi=dpi)
    plt.close(fig)


def write_records(path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def gate_metrics(records):
    scores = np.array([row["condition_score"] for row in records])
    targets = {
        "true_A_or_B_vs_true_C": np.array([row["true_label"] in ("A", "B") for row in records]),
        "steering_helpful_vs_non_helpful": np.array([
            row["effect_group"] in ("helpful_over", "helpful_under") for row in records
        ]),
    }
    metrics = []
    for target_name, target in targets.items():
        metrics.append({
            "target": target_name,
            "positive_count": int(target.sum()),
            "negative_count": int((~target).sum()),
            "roc_auc": roc_auc_score(target, scores),
            "pr_auc": average_precision_score(target, scores),
        })
    return metrics


def write_gate_tables(output_dir, records):
    grouped = {group: [row for row in records if row["effect_group"] == group] for group in EFFECT_ORDER}
    with (output_dir / "gate_on_rate_by_transition.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["effect_group", "n", "gate_on_count", "gate_on_rate", "mean_condition_score"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in EFFECT_ORDER:
            rows = grouped[group]
            writer.writerow({
                "effect_group": group,
                "n": len(rows),
                "gate_on_count": sum(row["condition_gate"] for row in rows),
                "gate_on_rate": np.mean([row["condition_gate"] for row in rows]),
                "mean_condition_score": np.mean([row["condition_score"] for row in rows]),
            })

    metrics = gate_metrics(records)
    with (output_dir / "gate_target_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def save_gate_figure(output_dir, records, dpi):
    groups = [group for group in EFFECT_ORDER if any(r["effect_group"] == group for r in records)]
    values = [np.array([r["condition_score"] for r in records if r["effect_group"] == group]) for group in groups]
    threshold_values = {row["condition_threshold"] for row in records}
    if len(threshold_values) != 1:
        raise ValueError(f"Expected one condition threshold, got {sorted(threshold_values)}")
    threshold = threshold_values.pop()

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(EFFECT_COLORS[group])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(28)
    for i, (group, scores) in enumerate(zip(groups, values), 1):
        jitter = rng.uniform(-0.16, 0.16, size=len(scores))
        ax.scatter(
            np.full(len(scores), i) + jitter, scores, s=17,
            color=EFFECT_COLORS[group], alpha=0.58, edgecolors="none",
        )
    ax.axhline(threshold, color="#111111", linestyle="--", linewidth=1.5, label=f"gate threshold={threshold:.3f}")
    ax.set_xticks(range(1, len(groups) + 1), [f"{g.replace('_', chr(10))}\n(n={len(v)})" for g, v in zip(groups, values)])
    ax.set_ylabel("condition score")
    ax.set_title("Does the current condition score identify samples helped by steering?", weight="bold")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "gate_score_by_effect.png", dpi=dpi)
    plt.close(fig)


def main(args):
    base_path = project_path(args.base_csv)
    steered_path = project_path(args.steered_csv)
    condition_path = project_path(args.condition_csv)
    transition_output_dir = project_path(args.transition_output_dir)
    gate_output_dir = project_path(args.gate_output_dir)
    transition_output_dir.mkdir(parents=True, exist_ok=True)
    gate_output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(base_path, steered_path, condition_path)
    assert_expected_totals(records)
    save_transition_figures(records, transition_output_dir, args.dpi)
    write_records(transition_output_dir / "per_sample_transitions.csv", records)
    save_gate_figure(gate_output_dir, records, args.dpi)
    gate_results = write_gate_tables(gate_output_dir, records)

    print(f"Validated {len(records)} unique, perfectly aligned test sample IDs")
    print("Base statuses:", dict(Counter(row["base_status"] for row in records)))
    print("Steered statuses:", dict(Counter(row["steered_status"] for row in records)))
    print("Effect groups:", dict(Counter(row["effect_group"] for row in records)))
    print("Prediction transitions:\n", prediction_matrix(records))
    for result in gate_results:
        print(
            f"{result['target']}: ROC-AUC={result['roc_auc']:.4f}, "
            f"PR-AUC={result['pr_auc']:.4f}"
        )
    print("Wrote transitions to", transition_output_dir)
    print("Wrote gate analysis to", gate_output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze sample-aligned layer-28 alpha-14 steering transitions and condition gate."
    )
    parser.add_argument(
        "--base-csv",
        default="outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv",
    )
    parser.add_argument(
        "--steered-csv",
        default=(
            "outputs/10b_balanced_cats_pc1_vectors/02_over/test/"
            "steered_qwen_vl_test_layer28_alpha14.0.csv"
        ),
    )
    parser.add_argument(
        "--condition-csv",
        default=(
            "outputs/10b_balanced_cats_pc1_vectors/02_over/test/"
            "conditional_over_pc1_test_layer28_alpha14.0_thr-0.07901761.csv"
        ),
    )
    parser.add_argument(
        "--transition-output-dir",
        default="notes/A_hidden_space_analysis/steering_transitions",
    )
    parser.add_argument(
        "--gate-output-dir",
        default="notes/A_hidden_space_analysis/condition_gate",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
