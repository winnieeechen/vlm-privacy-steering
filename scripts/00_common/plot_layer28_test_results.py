#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
N_TEST = 243


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


PANELS = {
    "Over unconditional": [
        ("Base", "no steering", "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv", "case_type"),
        ("02 Mean", "a=1.0", "outputs/02_formal_full1200/02_over/test/steered_mean_qwen_vl_test_layer28_alpha1.0.csv", "steered_case_type"),
        ("06 CATS-PCA", "a=2.0", "outputs/06_cats_pca_behavior_vectors/02_over/test/steered_qwen_vl_test_layer28_alpha2.0.csv", "steered_case_type"),
        ("10 Balanced", "a=2.5", "outputs/10_balanced_cats_transition_vectors/02_over/test/steered_qwen_vl_test_layer28_alpha2.5.csv", "steered_case_type"),
        ("10b PC1", "a=14.0", "outputs/10b_balanced_cats_pc1_vectors/02_over/test/steered_qwen_vl_test_layer28_alpha14.0.csv", "steered_case_type"),
    ],
    "Over conditional": [
        ("Base", "no steering", "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv", "case_type"),
        ("02 Mean", "a=1.0\nthr=0.011", "outputs/02_formal_full1200/02_over/test/conditional_mean_qwen_vl_test_layer28_alpha1.0_thr0.01074012.csv", "conditional_case_type"),
        ("06 CATS-PCA", "a=2.0\nthr=-0.079", "outputs/06_cats_pca_behavior_vectors/02_over/test/conditional_cats_pca_qwen_vl_test_layer28_alpha2.0_thr-0.07901761.csv", "conditional_case_type"),
        ("10 Balanced", "a=2.5\nthr=-0.012", "outputs/10_balanced_cats_transition_vectors/02_over/test/conditional_balanced_cats_qwen_vl_test_layer28_alpha2.5_thr-0.01161985.csv", "conditional_case_type"),
        ("10b PC1", "a=14.0\nthr=-0.079", "outputs/10b_balanced_cats_pc1_vectors/02_over/test/conditional_over_pc1_test_layer28_alpha14.0_thr-0.07901761.csv", "conditional_case_type"),
    ],
    "Under unconditional": [
        ("Base", "no steering", "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv", "case_type"),
        ("02 Mean", "a=1.0", "outputs/02_formal_full1200/03_under/test/steered_under_mean_qwen_vl_test_layer28_alpha1.0.csv", "under_case_type"),
        ("06 CATS-PCA", "a=1.0", "outputs/06_cats_pca_behavior_vectors/03_under/test/steered_under_qwen_vl_test_layer28_alpha1.0.csv", "under_case_type"),
        ("10 Balanced", "a=1.5", "outputs/10_balanced_cats_transition_vectors/03_under/test/steered_under_qwen_vl_test_layer28_alpha1.5.csv", "under_case_type"),
        ("10b PC1", "a=0.5", "outputs/10b_balanced_cats_pc1_vectors/03_under/test/steered_under_qwen_vl_test_layer28_alpha0.5.csv", "under_case_type"),
    ],
    "Under conditional": [
        ("Base", "no steering", "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv", "case_type"),
        ("02 Mean", "a=1.0\nthr=-0.047", "outputs/02_formal_full1200/03_under/test/conditional_under_mean_qwen_vl_test_layer28_alpha1.0_thr-0.04742758.csv", "conditional_case_type"),
        ("06 CATS-PCA", "a=1.0\nthr=-0.064", "outputs/06_cats_pca_behavior_vectors/03_under/test/conditional_under_cats_pca_qwen_vl_test_layer28_alpha1.0_thr-0.06363814.csv", "conditional_case_type"),
        ("10 Balanced", "a=1.5\nthr=-0.064", "outputs/10_balanced_cats_transition_vectors/03_under/test/conditional_under_balanced_cats_qwen_vl_test_layer28_alpha1.5_thr-0.06363814.csv", "conditional_case_type"),
        ("10b PC1", "a=0.5\nthr=-0.047", "outputs/10b_balanced_cats_pc1_vectors/03_under/test/conditional_under_pc1_test_layer28_alpha0.5_thr-0.04742758.csv", "conditional_case_type"),
    ],
    "Dual conditional": [
        ("Base", "no steering", "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv", "case_type"),
        ("02 Mean", "o=1.0, u=1.0\nthr=0.011/0.086", "outputs/02_formal_full1200/05_dual_additive/optimized_layer28/test/dual_conditional_mean_test_layer28_over1.0_under1.0_overthr0.01074012_underthr0.08586932.csv", "dual_additive_case_type"),
        ("06 CATS-PCA", "o=2.0, u=1.0\nthr=-0.079/0.086", "outputs/06_cats_pca_behavior_vectors/05_dual_additive/test/dual_conditional_cats_pca_test_layer28_over2.0_under1.0_overthr-0.07901761_underthr0.08586932.csv", "dual_additive_case_type"),
        ("10 Balanced", "o=2.5, u=1.5\nthr=0.032/-0.021", "outputs/10_balanced_cats_transition_vectors/05_dual_additive/test/dual_conditional_balanced_cats_test_layer28_over2.5_under1.5_overthr0.03188088_underthr-0.02138572.csv", "dual_additive_case_type"),
        ("10b PC1", "o=14.0, u=0.5\nthr=-0.079/0.086", "outputs/10b_balanced_cats_pc1_vectors/05_dual_additive/test/dual_conditional_balanced_cats_pc1_test_layer28_over14.0_under0.5_overthr-0.07901761_underthr0.08586932.csv", "dual_additive_case_type"),
    ],
}


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def centered_text(draw, xy, value, text_font, fill, spacing=4):
    box = draw.multiline_textbbox((0, 0), value, font=text_font, spacing=spacing, align="center")
    width = box[2] - box[0]
    draw.multiline_text((xy[0] - width / 2, xy[1]), value, font=text_font, fill=fill, spacing=spacing, align="center")


def dashed_line(draw, xy, fill, width=3, dash=14, gap=9):
    x1, y1, x2, y2 = xy
    if y1 != y2:
        raise ValueError("dashed_line only supports horizontal lines")
    x = x1
    while x < x2:
        draw.line((x, y1, min(x + dash, x2), y2), fill=fill, width=width)
        x += dash + gap


def summarize(path, case_key):
    path = ROOT / path
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_TEST:
        raise ValueError(f"Expected {N_TEST} rows in {path}, found {len(rows)}")
    return {
        "n": len(rows),
        "correct": sum(row[case_key] in CORRECT for row in rows),
        "over": sum(row[case_key] in OVER for row in rows),
        "under": sum(row[case_key] in UNDER for row in rows),
    }


def collect():
    records = []
    for panel, entries in PANELS.items():
        for method, setting, path, case_key in entries:
            metrics = summarize(path, case_key)
            record = {
                "panel": panel,
                "method": method,
                "setting": setting,
                "path": path,
                "available": metrics is not None,
            }
            if metrics:
                record.update(metrics)
            records.append(record)
    return records


def write_csv(records, output):
    fields = ["panel", "method", "setting", "available", "n", "correct", "correct_pct", "over", "over_pct", "under", "under_pct", "path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            if record["available"]:
                for key in ("correct", "over", "under"):
                    row[f"{key}_pct"] = f"{100 * record[key] / record['n']:.2f}"
            writer.writerow(row)


def draw_chart(records, output):
    width, height = 3900, 1700
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    title_font = font(46, bold=True)
    subtitle_font = font(24)
    panel_font = font(26, bold=True)
    axis_font = font(17)
    method_font = font(17, bold=True)
    setting_font = font(14)
    value_font = font(16, bold=True)
    legend_font = font(21)

    colors = {"correct": "#2f855a", "over": "#d95f59", "under": "#4c78a8"}
    dark = "#17212b"
    muted = "#52606d"
    grid = "#d9e2ec"

    centered_text(draw, (width / 2, 32), "Layer 28 Test Results", title_font, dark)
    centered_text(draw, (width / 2, 92), "Qwen2.5-VL-3B-Instruct | n = 243 | Bars show steering methods; dashed lines show Base/no steering", subtitle_font, muted)

    legend_y = 145
    legend_items = (("correct", "Correct"), ("over", "Over-disclosure"), ("under", "Under-disclosure"))
    legend_total = 760
    legend_x = (width - legend_total) / 2
    for i, (key, label) in enumerate(legend_items):
        x = legend_x + i * 205
        draw.rounded_rectangle((x, legend_y, x + 28, legend_y + 28), radius=4, fill=colors[key])
        draw.text((x + 39, legend_y + 1), label, font=legend_font, fill=dark)
    base_x = legend_x + 615
    dashed_line(draw, (base_x, legend_y + 14, base_x + 52, legend_y + 14), fill=dark, width=4)
    draw.text((base_x + 65, legend_y + 1), "Base", font=legend_font, fill=dark)

    by_panel = {panel: [record for record in records if record["panel"] == panel] for panel in PANELS}

    def axis_max(metrics):
        values = [
            record[key]
            for record in records
            if record["available"]
            for key in metrics
        ]
        return max(20, math.ceil((max(values) + 18) / 20) * 20)

    panel_margin = 38
    panel_gap = 24
    panel_width = (
        width - 2 * panel_margin - (len(PANELS) - 1) * panel_gap
    ) / len(PANELS)
    panel_height = 625
    row_tops = (205, 865)

    def subplot(panel, metrics, row_index, column_index, y_max):
        left = panel_margin + column_index * (panel_width + panel_gap)
        top = row_tops[row_index]
        right = left + panel_width
        bottom = top + panel_height
        draw.rounded_rectangle((left, top, right, bottom), radius=10, fill="#f7f9fb", outline="#cbd5df", width=2)

        metric_title = "Correct" if metrics == ("correct",) else "Over / Under"
        centered_text(draw, ((left + right) / 2, top + 18), f"{panel} - {metric_title}", panel_font, dark)

        plot_left = left + 66
        plot_right = right - 22
        chart_top = top + 88
        chart_bottom = bottom - 112
        plot_width = plot_right - plot_left

        for tick in range(0, y_max + 1, 20):
            y = chart_bottom - (chart_bottom - chart_top) * tick / y_max
            draw.line((plot_left, y, plot_right, y), fill=grid, width=1)
            tick_text = str(tick)
            box = draw.textbbox((0, 0), tick_text, font=axis_font)
            draw.text((plot_left - 12 - (box[2] - box[0]), y - 9), tick_text, font=axis_font, fill=muted)

        draw.line((plot_left, chart_top, plot_left, chart_bottom), fill="#829ab1", width=2)
        draw.line((plot_left, chart_bottom, plot_right, chart_bottom), fill="#829ab1", width=2)
        draw.text((plot_left, chart_top - 27), "Count", font=axis_font, fill=muted)

        entries = by_panel[panel]
        base = next((record for record in entries if record["method"] == "Base"), None)
        entries = [record for record in entries if record["method"] != "Base"]

        if base and base["available"]:
            for key in metrics:
                base_value = base[key]
                y = chart_bottom - (chart_bottom - chart_top) * base_value / y_max
                dashed_line(draw, (plot_left, y, plot_right, y), fill=colors[key], width=4)

        group_step = (plot_width - 24) / len(entries)
        bar_width = 54 if len(metrics) == 1 else 37
        metric_gap = 9
        group_width = len(metrics) * bar_width + (len(metrics) - 1) * metric_gap

        for group_index, record in enumerate(entries):
            center = plot_left + 12 + group_step * (group_index + 0.5)
            if not record["available"]:
                draw.rounded_rectangle((center - group_width / 2, chart_top + 70, center + group_width / 2, chart_bottom), radius=5, outline="#9fb3c8", width=2)
                centered_text(draw, (center, (chart_top + chart_bottom) / 2), "Not run", method_font, "#627d98")
            else:
                for metric_index, key in enumerate(metrics):
                    value = record[key]
                    x1 = center - group_width / 2 + metric_index * (bar_width + metric_gap)
                    x2 = x1 + bar_width
                    y1 = chart_bottom - (chart_bottom - chart_top) * value / y_max
                    draw.rounded_rectangle((x1, y1, x2, chart_bottom), radius=4, fill=colors[key])
                    label = f"{value}\n{100 * value / record['n']:.1f}%"
                    centered_text(draw, ((x1 + x2) / 2, y1 - 45), label, value_font, dark, spacing=1)

            centered_text(draw, (center, chart_bottom + 20), record["method"], method_font, dark)
            centered_text(draw, (center, chart_bottom + 46), record["setting"], setting_font, muted)

    for column_index, panel in enumerate(PANELS):
        correct_metrics = ("correct",)
        error_metrics = ("over", "under")
        subplot(
            panel,
            correct_metrics,
            0,
            column_index,
            axis_max(correct_metrics),
        )
        subplot(
            panel,
            error_metrics,
            1,
            column_index,
            axis_max(error_metrics),
        )

    centered_text(draw, (width / 2, 1580), "Base/no steering: Correct 87 (35.8%), Over 44 (18.1%), Under 112 (46.1%). Alpha and threshold values were selected on validation data; this figure uses test data only.", subtitle_font, muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-png", default="notes/layer28_test_results.png")
    parser.add_argument("--output-csv", default="notes/layer28_test_results.csv")
    args = parser.parse_args()
    records = collect()
    png_path = ROOT / args.output_png
    csv_path = ROOT / args.output_csv
    draw_chart(records, png_path)
    write_csv(records, csv_path)
    print("Saved PNG:", png_path)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
