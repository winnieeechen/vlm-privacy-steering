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
OVER_CSV = ROOT / "outputs" / "steered_qwen_vl_test_layer32_alpha0.5.csv"
UNDER_CSV = ROOT / "outputs" / "under_B_steered_qwen_vl_test_layer32_alpha0.3.csv"
PRIVACY_SCORE_CSV = ROOT / "outputs" / "condition_scores_qwen_vl_test_layer32.csv"
UTILITY_SCORE_CSV = ROOT / "outputs" / "utility_scores_qwen_vl_test_layer32.csv"

OUT_CSV = ROOT / "outputs" / "dual_router_qwen_vl_test_layer32_over0.5_under0.3_pthr-0.09_uthr0.08.csv"

PRIVACY_THRESHOLD = -0.09
UTILITY_THRESHOLD = 0.08

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
    over_rows = read_csv(OVER_CSV)
    under_rows = read_csv(UNDER_CSV)
    privacy_rows = read_csv(PRIVACY_SCORE_CSV)
    utility_rows = read_csv(UTILITY_SCORE_CSV)

    assert len(base_rows) == len(over_rows) == len(under_rows) == len(privacy_rows) == len(utility_rows)

    results = []
    over_gate = 0
    under_gate = 0

    for b, o, u, ps, us in zip(base_rows, over_rows, under_rows, privacy_rows, utility_rows):
        assert b["full_id"] == o["full_id"] == u["full_id"] == ps["full_id"] == us["full_id"]

        base_pred = b["pred_label"]
        privacy_score = float(ps["condition_score"])
        utility_score = float(us["utility_score"])

        rr = dict(b)
        rr["privacy_score"] = privacy_score
        rr["utility_score"] = utility_score
        rr["privacy_threshold"] = PRIVACY_THRESHOLD
        rr["utility_threshold"] = UTILITY_THRESHOLD
        rr["router_action"] = "base"

        if base_pred == "B" and privacy_score > PRIVACY_THRESHOLD:
            over_gate += 1
            rr["router_action"] = "over"
            rr["dual_pred_label"] = o["steered_pred_label"]
            rr["dual_case_type"] = o["steered_case_type"]
            rr["dual_answer"] = o["steered_answer"]

        elif base_pred == "A" and utility_score > UTILITY_THRESHOLD:
            under_gate += 1
            rr["router_action"] = "under"
            rr["dual_pred_label"] = u["under_pred_label"]
            rr["dual_case_type"] = u["under_case_type"]
            rr["dual_answer"] = u["under_answer"]

        else:
            rr["dual_pred_label"] = b["pred_label"]
            rr["dual_case_type"] = b["case_type"]
            rr["dual_answer"] = b["model_answer"]

        results.append(rr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Final dual-router fixed setting:")
    print("layer = 32")
    print("alpha_over = 0.5")
    print("alpha_under = 0.3")
    print("privacy_threshold = -0.09")
    print("utility_threshold = 0.08")
    print("over_gate:", over_gate, "/", len(results))
    print("under_gate:", under_gate, "/", len(results))

    summarize(base_rows, "pred_label", "case_type", "Base")
    summarize(over_rows, "steered_pred_label", "steered_case_type", "Over-only steering")
    summarize(under_rows, "under_pred_label", "under_case_type", "Under-only steering")
    summarize(results, "dual_pred_label", "dual_case_type", "Final dual-router")

    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
