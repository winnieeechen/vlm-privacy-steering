#!/usr/bin/env python3
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

BASE_CSV = ROOT / "outputs" / "base_qwen_vl_test_133.csv"
STEERED_CSV = ROOT / "outputs" / "steered_qwen_vl_test_layer32_alpha0.5.csv"
SCORE_CSV = ROOT / "outputs" / "condition_scores_qwen_vl_test_layer32.csv"

OUT_CSV = ROOT / "outputs" / "conditional_qwen_vl_test_layer32_alpha0.5_thr-0.09.csv"

THRESHOLD = -0.09

OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows, pred_key, case_key, name):
    n = len(rows)
    correct = sum(r[case_key] in CORRECT for r in rows)
    over = sum(r[case_key] in OVER for r in rows)
    under = sum(r[case_key] in UNDER for r in rows)

    print(f"\n{name}")
    print("Pred counts:", Counter(r[pred_key] for r in rows))
    print("Case counts:", Counter(r[case_key] for r in rows))
    print(f"Correct: {correct}/{n} = {correct/n:.3f}")
    print(f"Over-disclosure: {over}/{n} = {over/n:.3f}")
    print(f"Under-disclosure: {under}/{n} = {under/n:.3f}")


def main():
    base_rows = read_csv(BASE_CSV)
    steered_rows = read_csv(STEERED_CSV)
    score_rows = read_csv(SCORE_CSV)

    assert len(base_rows) == len(steered_rows) == len(score_rows)

    results = []
    gate_true = 0

    for b, s, sc in zip(base_rows, steered_rows, score_rows):
        assert b["full_id"] == s["full_id"] == sc["full_id"]

        score = float(sc["condition_score"])
        use_steer = score > THRESHOLD

        rr = dict(b)
        rr["condition_score"] = score
        rr["threshold"] = THRESHOLD
        rr["use_steer"] = str(use_steer)

        if use_steer:
            gate_true += 1
            rr["conditional_pred_label"] = s["steered_pred_label"]
            rr["conditional_case_type"] = s["steered_case_type"]
            rr["conditional_answer"] = s["steered_answer"]
        else:
            rr["conditional_pred_label"] = b["pred_label"]
            rr["conditional_case_type"] = b["case_type"]
            rr["conditional_answer"] = b["model_answer"]

        results.append(rr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Final fixed setting:")
    print("layer = 32")
    print("alpha = 0.5")
    print("threshold = -0.09")
    print("gate_true:", gate_true, "/", len(results))

    summarize(base_rows, "pred_label", "case_type", "Base")
    summarize(steered_rows, "steered_pred_label", "steered_case_type", "Unconditional steered")
    summarize(results, "conditional_pred_label", "conditional_case_type", "Final conditional steered")

    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
