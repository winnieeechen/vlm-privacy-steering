#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("correct_pct", "over_pct", "under_pct"):
            row[key] = float(row[key])
    return rows


def add_deltas(rows):
    base_by_panel = {}
    first_base = None
    for row in rows:
        if row["method"] == "Base":
            base_by_panel[row["panel"]] = row
            if first_base is None:
                first_base = row

    enriched = []
    if first_base is not None:
        base_item = dict(first_base)
        base_item["panel"] = "Base"
        for key in ("correct", "over", "under"):
            base_item[f"delta_{key}_pct"] = 0.0
        enriched.append(base_item)

    for row in rows:
        if row["method"] == "Base":
            continue
        base = base_by_panel[row["panel"]]
        item = dict(row)
        for key in ("correct", "over", "under"):
            item[f"delta_{key}_pct"] = item[f"{key}_pct"] - base[f"{key}_pct"]
        enriched.append(item)
    return enriched


def delta_fill(value, good_when_positive):
    good = value >= 0 if good_when_positive else value <= 0
    magnitude = min(abs(value) / 15, 1)
    if good:
        return (
            int(237 - 55 * magnitude),
            int(247 - 83 * magnitude),
            int(239 - 68 * magnitude),
        )
    return (
        int(253 - 21 * magnitude),
        int(239 - 107 * magnitude),
        int(236 - 116 * magnitude),
    )


def draw_centered(draw, box, text, text_font, fill="#17212b", spacing=3):
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) / 2
    y = box[1] + (box[3] - box[1] - text_h) / 2
    draw.multiline_text((x, y), text, font=text_font, fill=fill, spacing=spacing, align="center")


def make_table(rows, output):
    columns = [
        ("panel", "Setting", 250),
        ("method", "Method", 160),
        ("correct_pct", "Correct", 118),
        ("delta_correct_pct", "Δ Correct", 128),
        ("over_pct", "Over", 104),
        ("delta_over_pct", "Δ Over", 112),
        ("under_pct", "Under", 112),
        ("delta_under_pct", "Δ Under", 122),
    ]
    row_h = 54
    header_h = 62
    title_h = 96
    margin = 34
    width = margin * 2 + sum(col[2] for col in columns)
    height = title_h + header_h + row_h * len(rows) + margin + 36

    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = font(30, bold=True)
    subtitle_font = font(16)
    header_font = font(16, bold=True)
    cell_font = font(15)
    cell_bold = font(15, bold=True)

    draw_centered(
        draw,
        (margin, 20, width - margin, 54),
        "Layer 28 Test Results - Delta Table",
        title_font,
    )
    draw_centered(
        draw,
        (margin, 56, width - margin, 86),
        "Base is shown for reference. Green delta is favorable: higher Correct, lower Over, lower Under.",
        subtitle_font,
        "#52606d",
    )

    x = margin
    y = title_h
    for _, label, col_w in columns:
        draw.rectangle((x, y, x + col_w, y + header_h), fill="#17212b")
        draw_centered(draw, (x, y, x + col_w, y + header_h), label, header_font, "#ffffff")
        x += col_w

    y += header_h
    last_panel = None
    for idx, row in enumerate(rows):
        x = margin
        is_base = row["method"] == "Base"
        base_fill = "#eef2f6" if is_base else ("#ffffff" if idx % 2 == 0 else "#f7f9fb")
        for key, _, col_w in columns:
            fill = base_fill
            if is_base and key.startswith("delta_"):
                fill = "#e4ebf2"
            elif key == "delta_correct_pct":
                fill = delta_fill(row[key], good_when_positive=True)
            elif key in {"delta_over_pct", "delta_under_pct"}:
                fill = delta_fill(row[key], good_when_positive=False)
            draw.rectangle((x, y, x + col_w, y + row_h), fill=fill, outline="#d9e2ec")

            if key == "panel":
                text = row[key] if row[key] != last_panel else ""
                text_font = cell_bold
            elif key == "method":
                text = row[key]
                text_font = cell_bold
            elif key.startswith("delta_"):
                text = f"{row[key]:+0.1f} pp"
                text_font = cell_bold
            else:
                text = f"{row[key]:0.1f}%"
                text_font = cell_font
            draw_centered(draw, (x + 4, y, x + col_w - 4, y + row_h), text, text_font)
            x += col_w

        last_panel = row["panel"]
        y += row_h

    draw.text(
        (margin, height - 30),
        "Baseline per setting: Base/no steering on the same 243 test examples.",
        font=subtitle_font,
        fill="#52606d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def make_delta_plot(rows, output):
    labels = [f"{row['panel']}\n{row['method']}" for row in rows]
    y = list(range(len(rows)))

    fig_height = max(8, len(rows) * 0.43)
    fig, axes = plt.subplots(1, 3, figsize=(14, fig_height), sharey=True)
    metrics = [
        ("delta_correct_pct", "Correct vs Base", "#2f855a", "higher is better"),
        ("delta_over_pct", "Over vs Base", "#d95f59", "lower is better"),
        ("delta_under_pct", "Under vs Base", "#4c78a8", "lower is better"),
    ]

    for ax, (key, title, color, subtitle) in zip(axes, metrics):
        values = [row[key] for row in rows]
        ax.axvline(0, color="#17212b", linewidth=1)
        ax.barh(y, values, color=color, alpha=0.88)
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
        ax.set_axisbelow(True)
        limit = max(18, max(abs(value) for value in values) + 3)
        ax.set_xlim(-limit, limit)
        ax.tick_params(axis="both", labelsize=9)
        for yi, value in zip(y, values):
            ha = "left" if value >= 0 else "right"
            x = value + (0.6 if value >= 0 else -0.6)
            ax.text(x, yi, f"{value:+.1f}", va="center", ha=ha, fontsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Layer 28 Test Results - Percentage Point Change from Base", fontsize=16, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor="#2f855a", label="Correct"),
            Patch(facecolor="#d95f59", label="Over-disclosure"),
            Patch(facecolor="#4c78a8", label="Under-disclosure"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="notes/layer28_test_results.csv")
    parser.add_argument("--output-table", default="notes/layer28_test_results_delta_table.png")
    parser.add_argument("--output-delta", default="notes/layer28_test_results_delta_plot.png")
    args = parser.parse_args()

    rows = load_rows(ROOT / args.input_csv)
    deltas = add_deltas(rows)
    make_table(deltas, ROOT / args.output_table)
    make_delta_plot([row for row in deltas if row["method"] != "Base"], ROOT / args.output_delta)
    print("Saved table:", ROOT / args.output_table)
    print("Saved delta plot:", ROOT / args.output_delta)


if __name__ == "__main__":
    main()
