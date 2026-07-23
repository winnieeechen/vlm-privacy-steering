#!/usr/bin/env python3
"""Summarize and rank method 14 validation confidence sweeps."""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "outputs/14_free/val/14_free_val_a1.0-1.5_c3.0-4.0.csv"
SWEEP_DIR = ROOT / "outputs/14_free/val_confidence_sweep"
SUMMARY = SWEEP_DIR / "summary.csv"
LABELS = ("A", "B", "C")


def read_unique(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return list({row["full_id"]: row for row in rows}.values())


def summarize(path):
    rows = read_unique(path)
    totals = Counter(row["true_label"] for row in rows)
    correct = Counter(
        row["true_label"]
        for row in rows
        if row["true_label"] == row.get("Q7-label")
    )
    overall_correct = sum(correct.values())
    recalls = {label: correct[label] / totals[label] for label in LABELS}
    first = rows[0]
    return {
        "file": str(path.relative_to(ROOT)),
        "threshold_a": float(first["min_confidence_a"]),
        "threshold_c": float(first["min_confidence_c"]),
        "completed": len(rows),
        "correct": overall_correct,
        "accuracy": overall_correct / len(rows),
        "recall_a": recalls["A"],
        "recall_b": recalls["B"],
        "recall_c": recalls["C"],
        "macro_recall": sum(recalls.values()) / len(recalls),
    }


def main():
    paths = [BASELINE, *sorted(SWEEP_DIR.glob("14_free_val_*.csv"))]
    summaries = [summarize(path) for path in paths if path.exists()]
    summaries.sort(key=lambda row: (row["macro_recall"], row["accuracy"]), reverse=True)
    if not summaries:
        raise RuntimeError("No completed validation outputs found")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("rank  thresholds   correct   accuracy   recall A/B/C       macro")
    for rank, row in enumerate(summaries, 1):
        recalls = "/".join(
            f"{100 * row[key]:.1f}" for key in ("recall_a", "recall_b", "recall_c")
        )
        print(
            f"{rank:>4}  {row['threshold_a']:.2f}/{row['threshold_c']:.2f}"
            f"     {row['correct']:>3}/{row['completed']:<3}"
            f"    {100 * row['accuracy']:>5.1f}%"
            f"     {recalls:<16} {100 * row['macro_recall']:>5.1f}%"
        )
    print(f"Saved: {SUMMARY}")


if __name__ == "__main__":
    main()
