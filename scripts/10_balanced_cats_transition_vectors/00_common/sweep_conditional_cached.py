#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path


OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def read_csv(path):
    with open(ROOT / path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(rows, case_key):
    n = len(rows)
    return {
        "correct": sum(row[case_key] in CORRECT for row in rows),
        "over": sum(row[case_key] in OVER for row in rows),
        "under": sum(row[case_key] in UNDER for row in rows),
        "n": n,
        "case_counts": dict(Counter(row[case_key] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--steered-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--score-key", default="condition_score")
    parser.add_argument("--threshold-mode", choices=["quantile", "fixed"], default="quantile")
    parser.add_argument("--min-gate-rate", type=float, default=0.2)
    parser.add_argument("--max-gate-rate", type=float, default=0.8)
    args = parser.parse_args()

    base_rows = read_csv(args.base_csv)
    steered_rows = read_csv(args.steered_csv)
    score_rows = read_csv(args.score_csv)
    if not (len(base_rows) == len(steered_rows) == len(score_rows)):
        raise ValueError("Base, steered, and score CSV lengths do not match")
    for base, steered, score in zip(base_rows, steered_rows, score_rows):
        if not (base["full_id"] == steered["full_id"] == score["full_id"]):
            raise ValueError(f"Row alignment mismatch at {base['full_id']}")

    scores = [float(row[args.score_key]) for row in score_rows]
    if args.threshold_mode == "quantile":
        thresholds = sorted(set(quantile(scores, q) for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
    else:
        thresholds = [-0.12, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]

    results = []
    for threshold in thresholds:
        conditional = []
        gate_true = 0
        for base, steered, score_row in zip(base_rows, steered_rows, score_rows):
            use_steer = float(score_row[args.score_key]) > threshold
            gate_true += int(use_steer)
            row = dict(base)
            row["conditional_case_type"] = steered["steered_case_type"] if use_steer else base["case_type"]
            conditional.append(row)
        metrics = summarize(conditional, "conditional_case_type")
        gate_rate = gate_true / len(conditional)
        eligible = args.min_gate_rate <= gate_rate <= args.max_gate_rate
        results.append({
            "threshold": threshold,
            "gate_true": gate_true,
            "gate_rate": gate_rate,
            "eligible": eligible,
            "correct": metrics["correct"],
            "correct_rate": metrics["correct"] / metrics["n"],
            "over": metrics["over"],
            "over_rate": metrics["over"] / metrics["n"],
            "under": metrics["under"],
            "under_rate": metrics["under"] / metrics["n"],
            "case_counts": metrics["case_counts"],
        })

    eligible = [row for row in results if row["eligible"]]
    if not eligible:
        raise RuntimeError("No threshold satisfies the requested gate-rate range")
    best = max(eligible, key=lambda row: (row["correct"], -row["over"], -row["gate_rate"]))

    output = ROOT / args.output_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("threshold, gate_rate, correct, over, under, eligible")
    for row in results:
        print(f"{row['threshold']:.8f}, {row['gate_rate']:.3f}, {row['correct']}, {row['over']}, {row['under']}, {row['eligible']}")
    print("\nSelected val threshold:", best["threshold"])
    print("Gate:", best["gate_true"], "/", len(base_rows), f"= {best['gate_rate']:.3f}")
    print("Correct / over / under:", best["correct"], best["over"], best["under"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
