#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with open(project_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a validation-selected condition threshold to cached test outputs."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--steered-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--score-key", required=True)
    parser.add_argument("--steered-pred-key", required=True)
    parser.add_argument("--steered-case-key", required=True)
    parser.add_argument("--steered-answer-key", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base = read_csv(args.base_csv)
    steered = read_csv(args.steered_csv)
    scores = read_csv(args.score_csv)
    if not (len(base) == len(steered) == len(scores)):
        raise ValueError("Base, steered, and score CSV lengths do not match")

    results = []
    gate_true = 0
    for base_row, steered_row, score_row in zip(base, steered, scores):
        if not (
            base_row["full_id"]
            == steered_row["full_id"]
            == score_row["full_id"]
        ):
            raise ValueError("CSV row order/full_id mismatch")
        score = float(score_row[args.score_key])
        use_steer = score >= args.threshold
        gate_true += use_steer
        result = dict(base_row)
        result["condition_score"] = score
        result["condition_threshold"] = args.threshold
        result["condition_gate"] = use_steer
        result["conditional_action"] = "steered" if use_steer else "base"
        result["conditional_pred_label"] = (
            steered_row[args.steered_pred_key]
            if use_steer
            else base_row["pred_label"]
        )
        result["conditional_case_type"] = (
            steered_row[args.steered_case_key]
            if use_steer
            else base_row["case_type"]
        )
        result["conditional_answer"] = (
            steered_row[args.steered_answer_key]
            if use_steer
            else base_row["model_answer"]
        )
        results.append(result)

    cases = [row["conditional_case_type"] for row in results]
    correct = sum(case in CORRECT for case in cases)
    over = sum(case in OVER for case in cases)
    under = sum(case in UNDER for case in cases)
    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Threshold: {args.threshold:.8f}")
    print(f"Gate: {gate_true}/{len(results)} = {gate_true / len(results):.3f}")
    print(f"Correct / over / under: {correct} / {over} / {under}")
    print("Saved:", output)


if __name__ == "__main__":
    main()
