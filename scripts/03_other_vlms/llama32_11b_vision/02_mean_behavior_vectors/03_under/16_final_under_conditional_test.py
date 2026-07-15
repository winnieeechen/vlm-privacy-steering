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

OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics(rows, pred_key, case_key):
    n = len(rows)
    correct = sum(r[case_key] in CORRECT for r in rows)
    over = sum(r[case_key] in OVER for r in rows)
    under = sum(r[case_key] in UNDER for r in rows)
    return {
        "n": n,
        "correct": correct,
        "correct_rate": correct / n,
        "over": over,
        "over_rate": over / n,
        "under": under,
        "under_rate": under / n,
        "pred_counts": Counter(r[pred_key] for r in rows),
        "case_counts": Counter(r[case_key] for r in rows),
    }


def print_metrics(name, m):
    print(f"\n{name}")
    print("Pred counts:", m["pred_counts"])
    print("Case counts:", m["case_counts"])
    print(f"Correct: {m['correct']}/{m['n']} = {m['correct_rate']:.3f}")
    print(f"Over-disclosure: {m['over']}/{m['n']} = {m['over_rate']:.3f}")
    print(f"Under-disclosure: {m['under']}/{m['n']} = {m['under_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-csv",
        default=(
            "outputs/03_other_vlms/llama32_11b_vision/00_base/test/"
            "base_llama32_vision_test_243.csv"
        ),
    )
    parser.add_argument(
        "--under-csv",
        default=(
            "outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/03_under/test/"
            "under_steered_llama32_vision_test_layer32_alpha0.5.csv"
        ),
    )
    parser.add_argument(
        "--score-csv",
        default=(
            "outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/03_under/test/"
            "under_condition_scores_test_layer32.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/03_under/test/"
            "final_under_conditional_test_layer32_alpha0.5_thr0.10.csv"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    base_rows = read_csv(ROOT / args.base_csv)
    under_rows = read_csv(ROOT / args.under_csv)
    score_rows = read_csv(ROOT / args.score_csv)
    assert len(base_rows) == len(under_rows) == len(score_rows)

    results = []
    gate_true = 0

    for b, u, sc in zip(base_rows, under_rows, score_rows):
        assert b["full_id"] == u["full_id"] == sc["full_id"]

        score = float(sc["under_condition_score"])
        use_under = score >= args.threshold

        rr = dict(b)
        rr["under_condition_score"] = score
        rr["threshold"] = args.threshold
        rr["use_under"] = str(use_under)

        if use_under:
            gate_true += 1
            rr["conditional_under_pred_label"] = u["under_pred_label"]
            rr["conditional_under_case_type"] = u["under_case_type"]
        else:
            rr["conditional_under_pred_label"] = b["pred_label"]
            rr["conditional_under_case_type"] = b["case_type"]

        results.append(rr)

    out_path = ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Threshold:", args.threshold)
    print(f"Gate true: {gate_true}/{len(results)}")
    print_metrics("Base", metrics(base_rows, "pred_label", "case_type"))
    print_metrics("Unconditional under", metrics(under_rows, "under_pred_label", "under_case_type"))
    print_metrics(
        "Final conditional under",
        metrics(results, "conditional_under_pred_label", "conditional_under_case_type"),
    )
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
