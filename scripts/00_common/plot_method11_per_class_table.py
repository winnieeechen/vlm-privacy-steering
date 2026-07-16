#!/usr/bin/env python3
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "notes" / "presentation"
OUTPUT_CSV = OUTPUT_DIR / "method11_per_class_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "method11_per_class_results_table.png"

EXPERIMENTS = [
    {
        "model": "Qwen2.5-VL-3B",
        "path": (
            "outputs/11_pairwise_boundary_vectors/04_pairwise_router/test/"
            "routed_test_layer28_aa1.5_ac1.5_bisector.csv"
        ),
    },
    {
        "model": "Llama-3.2-11B-Vision",
        "path": (
            "outputs/03_other_vlms/llama32_11b_vision/"
            "11_pairwise_boundary_router/router/test/"
            "routed_llama32_vision_test_layer28_aa1.5_ac1.0_bisector.csv"
        ),
    },
]


def read_rows(relative_path):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calculate(experiment):
    rows = read_rows(experiment["path"])
    required = {
        "full_id", "true_label", "pred_label", "steered_pred_label", "route_target"
    }
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing columns in {experiment['path']}: {sorted(missing)}")

    ids = [row["full_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate full_id values in {experiment['path']}")

    n = len(rows)
    true_counts = Counter(row["true_label"] for row in rows)
    base_counts = Counter(row["pred_label"] for row in rows)
    final_counts = Counter(row["steered_pred_label"] for row in rows)
    route_counts = Counter(row["route_target"] for row in rows)
    correct_counts = Counter(
        row["true_label"]
        for row in rows
        if row["true_label"] == row["steered_pred_label"]
    )

    if set(true_counts) != {"A", "B", "C"}:
        raise ValueError(f"Unexpected true labels in {experiment['path']}: {true_counts}")

    result = {
        "model": experiment["model"],
        "n": n,
        "base_b": base_counts["B"],
        "method11_b": final_counts["B"],
        "route_a": route_counts["A"],
        "route_b": route_counts["B"],
        "route_c": route_counts["C"],
        "path": experiment["path"],
        "full_ids": ids,
    }
    for label in ("A", "B", "C"):
        result[f"true_{label.lower()}"] = true_counts[label]
        result[f"correct_{label.lower()}"] = correct_counts[label]
    result["overall_correct"] = sum(correct_counts.values())
    return result


def pct(count, total):
    return 100.0 * count / total


def metric(count, total):
    return f"{count}/{total}\n({pct(count, total):.1f}%)"


def save_csv(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "n", "base_b", "method11_b",
        "true_a", "correct_a", "true_b", "correct_b", "true_c", "correct_c",
        "overall_correct", "route_a", "route_b", "route_c", "path",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def draw_table(results):
    headers = [
        "Model",
        "Base\noutput B",
        "Method 11\noutput B",
        "True A\ncorrect",
        "True B\ncorrect",
        "True C\ncorrect",
        "Overall\ncorrect",
    ]
    rows = []
    for result in results:
        rows.append([
            result["model"],
            metric(result["base_b"], result["n"]),
            metric(result["method11_b"], result["n"]),
            metric(result["correct_a"], result["true_a"]),
            metric(result["correct_b"], result["true_b"]),
            metric(result["correct_c"], result["true_c"]),
            metric(result["overall_correct"], result["n"]),
        ])

    figure, axis = plt.subplots(figsize=(14.2, 4.5))
    figure.patch.set_facecolor("white")
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20, 0.13, 0.15, 0.13, 0.13, 0.13, 0.14],
        bbox=[0.02, 0.13, 0.96, 0.57],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)

    header_color = "#17222f"
    grid_color = "#cbd5e1"
    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold", fontsize=11.5)
        cell.set_edgecolor("white")

    for row_index in range(1, len(results) + 1):
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor("#ffffff" if row_index % 2 == 0 else "#f4f7fa")
            cell.set_edgecolor(grid_color)
            cell.set_linewidth(0.9)
            cell.set_text_props(color="#17222f")
        table[(row_index, 0)].set_text_props(weight="bold")
        table[(row_index, 2)].set_facecolor("#fff3cd")
        table[(row_index, 4)].set_facecolor("#fce8e6")
        table[(row_index, 4)].set_text_props(weight="bold", color="#9b1c1c")
        table[(row_index, 6)].set_facecolor("#e4f3e8")
        table[(row_index, 6)].set_text_props(weight="bold", color="#14532d")

    axis.text(
        0.5, 0.96,
        "Method 11: Overall Accuracy vs. Middle-Class (B) Behavior",
        transform=axis.transAxes,
        ha="center", va="top",
        fontsize=20, fontweight="bold", color="#17222f",
    )
    axis.text(
        0.5, 0.875,
        "Test set: 243 examples (A=98, B=49, C=96). The pairwise router selects only A or C; B is not a routing target.",
        transform=axis.transAxes,
        ha="center", va="top",
        fontsize=11.5, color="#526172",
    )
    axis.text(
        0.5, 0.055,
        "Method 11 improves overall correctness mainly through A/C decisions, while almost eliminating correct B predictions.",
        transform=axis.transAxes,
        ha="center", va="bottom",
        fontsize=11.5, color="#9b1c1c", fontweight="bold",
    )

    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    results = [calculate(experiment) for experiment in EXPERIMENTS]
    if results[0]["full_ids"] != results[1]["full_ids"]:
        raise ValueError("Qwen and Llama test files do not use the same ordered full_ids")
    save_csv(results)
    draw_table(results)
    for result in results:
        print(
            f"{result['model']}: Method 11 B={result['method11_b']}/{result['n']}, "
            f"B correct={result['correct_b']}/{result['true_b']}, "
            f"overall={result['overall_correct']}/{result['n']}, "
            f"routes=A:{result['route_a']} B:{result['route_b']} C:{result['route_c']}"
        )
    print("Saved audit CSV:", OUTPUT_CSV)
    print("Saved table PNG:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
