#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "notes" / "presentation"
OUTPUT_CSV = OUTPUT_DIR / "qwen_selected_test_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "qwen_selected_test_results_table.png"
TITLE = "Qwen2.5-VL-3B: Selected Test Results"
SUBTITLE = "Counts and percentages on the same 243 test examples"

EXPERIMENTS = [
    {
        "method": "Base",
        "setting": "No steering",
        "path": "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv",
        "case_key": "case_type",
    },
    {
        "method": "Mean Diff",
        "setting": "Over uncond. | L28, alpha=1.0",
        "path": "outputs/02_formal_full1200/02_over/test/steered_mean_qwen_vl_test_layer28_alpha1.0.csv",
        "case_key": "steered_case_type",
    },
    {
        "method": "CATS-PCA",
        "setting": "Over uncond. | L28, alpha=2.0",
        "path": "outputs/06_cats_pca_behavior_vectors/02_over/test/steered_qwen_vl_test_layer28_alpha2.0.csv",
        "case_key": "steered_case_type",
    },
    {
        "method": "Balanced CATS-PC1",
        "setting": "Over uncond. | L28, alpha=14.0",
        "path": "outputs/10b_balanced_cats_pc1_vectors/02_over/test/steered_qwen_vl_test_layer28_alpha14.0.csv",
        "case_key": "steered_case_type",
    },
    {
        "method": "Pairwise Router",
        "setting": "L28 | alpha_A=1.5, alpha_C=1.5",
        "path": "outputs/11_pairwise_boundary_vectors/04_pairwise_router/test/routed_test_layer28_aa1.5_ac1.5_bisector.csv",
        "case_key": "steered_case_type",
    },
]


def read_rows(path):
    with open(ROOT / path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calculate(experiment, expected_ids=None):
    rows = read_rows(experiment["path"])
    ids = [row["full_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate full_id values in {experiment['path']}")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError(f"Row order/full_id mismatch in {experiment['path']}")
    if experiment["case_key"] not in rows[0]:
        raise KeyError(
            f"Missing {experiment['case_key']} in {experiment['path']}"
        )

    cases = [row[experiment["case_key"]] for row in rows]
    unknown = [case for case in cases if case not in CORRECT | OVER | UNDER]
    if unknown:
        raise ValueError(
            f"Unrecognized cases in {experiment['path']}: {sorted(set(unknown))}"
        )

    n = len(rows)
    correct = sum(case in CORRECT for case in cases)
    over = sum(case in OVER for case in cases)
    under = sum(case in UNDER for case in cases)
    if correct + over + under != n:
        raise AssertionError("Correct/over/under counts do not partition the rows")
    return {
        "method": experiment["method"],
        "setting": experiment["setting"],
        "n": n,
        "correct": correct,
        "correct_pct": 100.0 * correct / n,
        "over": over,
        "over_pct": 100.0 * over / n,
        "under": under,
        "under_pct": 100.0 * under / n,
        "path": experiment["path"],
        "full_ids": ids,
    }


def save_audit_csv(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "setting", "n",
        "correct", "correct_pct",
        "over", "over_pct",
        "under", "under_pct", "path",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def metric_text(result, metric):
    return f"{result[metric]}\n({result[f'{metric}_pct']:.1f}%)"


def draw_table(results):
    headers = ["Method", "Selected test configuration", "Correct", "Over", "Under"]
    table_rows = [
        [
            result["method"],
            result["setting"],
            metric_text(result, "correct"),
            metric_text(result, "over"),
            metric_text(result, "under"),
        ]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(14, 6.3))
    figure.patch.set_facecolor("#ffffff")
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.34, 0.145, 0.145, 0.145],
        bbox=[0.02, 0.08, 0.96, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)

    header_color = "#17222f"
    grid_color = "#cbd5e1"
    row_colors = ["#eef3f8", "#ffffff"]
    column_colors = {2: "#eaf2fb", 3: "#fceeee", 4: "#fff6df"}
    best_color = "#d8f0df"

    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold", fontsize=12)
        cell.set_edgecolor("white")
        cell.set_linewidth(1.1)

    for row_index, result in enumerate(results, start=1):
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor(column_colors.get(column, row_colors[(row_index - 1) % 2]))
            cell.set_edgecolor(grid_color)
            cell.set_linewidth(0.8)
            cell.set_text_props(color="#17222f")
        table[(row_index, 0)].set_text_props(weight="bold", color="#17222f")

    best_correct = max(result["correct"] for result in results)
    best_over = min(result["over"] for result in results)
    best_under = min(result["under"] for result in results)
    for row_index, result in enumerate(results, start=1):
        for column, metric, best in [
            (2, "correct", best_correct),
            (3, "over", best_over),
            (4, "under", best_under),
        ]:
            if result[metric] == best:
                table[(row_index, column)].set_facecolor(best_color)
                table[(row_index, column)].set_text_props(
                    weight="bold", color="#14532d"
                )

    axis.text(
        0.5,
        0.96,
        TITLE,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#17222f",
    )
    axis.text(
        0.5,
        0.895,
        SUBTITLE,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        color="#526172",
    )


    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    base_ids = None
    results = []
    for experiment in EXPERIMENTS:
        result = calculate(experiment, base_ids)
        if base_ids is None:
            base_ids = result["full_ids"]
        results.append(result)
    save_audit_csv(results)
    draw_table(results)

    print("Verified all files use the same ordered 243 full_ids.")
    for result in results:
        print(
            f"{result['method']}: "
            f"correct={result['correct']} ({result['correct_pct']:.2f}%), "
            f"over={result['over']} ({result['over_pct']:.2f}%), "
            f"under={result['under']} ({result['under_pct']:.2f}%)"
        )
    print("Saved audit CSV:", OUTPUT_CSV)
    print("Saved table PNG:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
