#!/usr/bin/env python3
"""Audit a completed method-14 CSV and print routing and final metrics."""

import argparse
from collections import Counter, defaultdict

import method14_common as common


LABELS = common.LABELS


def ratio(numerator, denominator):
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv")
    parser.add_argument("--base-csv")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = common.read_rows(args.result_csv)
    identifiers = [row["full_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Result CSV contains duplicate full_id values")
    judged = [row for row in rows if row.get("Q7-label", "").strip()]
    if args.require_complete and len(judged) != len(rows):
        raise RuntimeError(f"Only {len(judged)}/{len(rows)} rows are judged")
    print(f"Judged: {len(judged)}/{len(rows)}")
    if not judged:
        return

    correct = sum(row["Q7-label"] == row["true_label"] for row in judged)
    print("Final correct:", ratio(correct, len(judged)))
    route_rows = [row for row in judged if row.get("route_target") in LABELS]
    route_correct = sum(
        row["route_target"] == row["true_label"] for row in route_rows
    )
    print("Route correct:", ratio(route_correct, len(route_rows)))
    print("Route counts:", dict(Counter(row["route_target"] for row in route_rows)))

    print("\nPer-class final recall")
    for label in LABELS:
        selected = [row for row in judged if row["true_label"] == label]
        hits = sum(row["Q7-label"] == label for row in selected)
        print(f"  {label}: {ratio(hits, len(selected))}")

    transitions = defaultdict(int)
    for row in judged:
        transitions[(row["true_label"], row["Q7-label"])] += 1
    print("\nTrue -> final matrix")
    print("      " + " ".join(f"{label:>5s}" for label in LABELS) + " other")
    for truth in LABELS:
        values = [transitions[(truth, prediction)] for prediction in LABELS]
        other = sum(
            count
            for (row_truth, prediction), count in transitions.items()
            if row_truth == truth and prediction not in LABELS
        )
        print(f"  {truth}: " + " ".join(f"{value:5d}" for value in values) + f" {other:5d}")

    if args.base_csv:
        baseline = {row["full_id"]: row for row in common.read_rows(args.base_csv)}
        paired = [row for row in judged if row["full_id"] in baseline]
        baseline_correct = sum(
            baseline[row["full_id"]].get("pred_label") == row["true_label"]
            for row in paired
        )
        gains = sum(
            row["Q7-label"] == row["true_label"]
            and baseline[row["full_id"]].get("pred_label") != row["true_label"]
            for row in paired
        )
        losses = sum(
            row["Q7-label"] != row["true_label"]
            and baseline[row["full_id"]].get("pred_label") == row["true_label"]
            for row in paired
        )
        print("\nPaired base comparison")
        print("  Base correct:", ratio(baseline_correct, len(paired)))
        print(f"  Gains={gains}, losses={losses}, net={gains - losses:+d}")


if __name__ == "__main__":
    main()
