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


def summarize(rows, case_key):
    counts = Counter(row[case_key] for row in rows)
    return (
        sum(counts[key] for key in CORRECT),
        sum(counts[key] for key in OVER),
        sum(counts[key] for key in UNDER),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--steered-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--score-key", default="condition_score")
    args = parser.parse_args()

    base_rows = read_csv(args.base_csv)
    steered_rows = read_csv(args.steered_csv)
    score_rows = read_csv(args.score_csv)
    if not (len(base_rows) == len(steered_rows) == len(score_rows)):
        raise ValueError("Base, steered, and score CSV lengths do not match")

    results = []
    gate_true = 0
    for base, steered, score_row in zip(base_rows, steered_rows, score_rows):
        if not (base["full_id"] == steered["full_id"] == score_row["full_id"]):
            raise ValueError(f"Row alignment mismatch at {base['full_id']}")
        score = float(score_row[args.score_key])
        use_steer = score > args.threshold
        gate_true += int(use_steer)
        row = dict(base)
        row.update({
            "condition_score": score,
            "threshold": args.threshold,
            "use_steer": use_steer,
            "conditional_pred_label": steered["steered_pred_label"] if use_steer else base["pred_label"],
            "conditional_case_type": steered["steered_case_type"] if use_steer else base["case_type"],
            "conditional_answer": steered["steered_answer"] if use_steer else base["model_answer"],
        })
        results.append(row)

    output = ROOT / args.output_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    base_metrics = summarize(base_rows, "case_type")
    unconditional_metrics = summarize(steered_rows, "steered_case_type")
    conditional_metrics = summarize(results, "conditional_case_type")
    print("Threshold:", args.threshold)
    print("Gate:", gate_true, "/", len(results), f"= {gate_true / len(results):.3f}")
    print("Base correct / over / under:", *base_metrics)
    print("Unconditional correct / over / under:", *unconditional_metrics)
    print("Conditional correct / over / under:", *conditional_metrics)
    print("Saved:", output)


if __name__ == "__main__":
    main()
