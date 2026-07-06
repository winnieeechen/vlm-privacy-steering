#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()

SCORE_CSV = ROOT / "outputs" / "condition_scores_qwen_vl_40.csv"

THRESHOLDS = [
    -0.080, -0.070, -0.060, -0.050,
    -0.045, -0.042, -0.040, -0.035,
    -0.030, -0.025, -0.020, -0.015,
    -0.010, 0.000,
]

OVER_CASES = {"A_to_B", "A_to_C", "B_to_C"}
UNDER_CASES = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT_CASES = {"A_to_A", "B_to_B", "C_to_C"}


def load_by_id(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["full_id"]: r for r in rows}


def summarize(cases):
    n = len(cases)
    correct = sum(c in CORRECT_CASES for c in cases)
    over = sum(c in OVER_CASES for c in cases)
    under = sum(c in UNDER_CASES for c in cases)
    return correct, over, under


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steered-csv", required=True)
    args = parser.parse_args()

    steered_csv = Path(args.steered_csv)

    score_by_id = load_by_id(SCORE_CSV)
    steered_by_id = load_by_id(steered_csv)

    rows = list(steered_by_id.values())

    print("Rows:", len(rows))
    print("Using cached condition scores:", SCORE_CSV)
    print("Using cached steering:", steered_csv)

    print("\nthreshold, gate_on, correct, over, under, case_counts")

    for thr in THRESHOLDS:
        final_cases = []
        gate_on = 0

        for r in rows:
            full_id = r["full_id"]
            score = float(score_by_id[full_id]["condition_score"])

            if score > thr:
                final_case = r["steered_case_type"]
                gate_on += 1
            else:
                final_case = r["case_type"]

            final_cases.append(final_case)

        correct, over, under = summarize(final_cases)
        counts = Counter(final_cases)

        print(
            f"{thr: .3f}, "
            f"gate_on={gate_on:2d}, "
            f"correct={correct:2d}/40={correct/40:.3f}, "
            f"over={over:2d}/40={over/40:.3f}, "
            f"under={under:2d}/40={under/40:.3f}, "
            f"{dict(counts)}"
        )


if __name__ == "__main__":
    main()
