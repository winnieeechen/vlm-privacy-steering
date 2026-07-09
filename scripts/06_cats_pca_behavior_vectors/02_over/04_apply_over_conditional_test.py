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
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    base_csv = ROOT / "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv"
    steered_csv = ROOT / (
        "outputs/06_cats_pca_behavior_vectors/02_over/test/"
        f"steered_qwen_vl_test_layer{args.layer}_alpha{args.alpha}.csv"
    )
    score_csv = ROOT / "outputs/02_formal_full1200/02_over/test/condition_scores_test_layer32.csv"
    out_csv = ROOT / (
        "outputs/06_cats_pca_behavior_vectors/02_over/test/"
        f"conditional_cats_pca_test_layer{args.layer}_alpha{args.alpha}_thr{args.threshold}.csv"
    )

    base_rows = read_csv(base_csv)
    steered_rows = read_csv(steered_csv)
    score_rows = read_csv(score_csv)
    assert len(base_rows) == len(steered_rows) == len(score_rows)

    results = []
    gate_true = 0
    for b, s, sc in zip(base_rows, steered_rows, score_rows):
        assert b["full_id"] == s["full_id"] == sc["full_id"]
        score = float(sc["condition_score"])
        use_steer = score > args.threshold
        if use_steer:
            gate_true += 1

        rr = dict(b)
        rr["condition_score"] = score
        rr["threshold"] = args.threshold
        rr["use_steer"] = str(use_steer)
        rr["conditional_pred_label"] = s["steered_pred_label"] if use_steer else b["pred_label"]
        rr["conditional_case_type"] = s["steered_case_type"] if use_steer else b["case_type"]
        rr["conditional_answer"] = s["steered_answer"] if use_steer else b["model_answer"]
        results.append(rr)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("CATS-PCA over conditional test")
    print("Layer:", args.layer)
    print("Alpha:", args.alpha)
    print("Threshold:", args.threshold)
    print("Gate true:", gate_true, "/", len(results))
    summarize(base_rows, "pred_label", "case_type", "Base")
    summarize(steered_rows, "steered_pred_label", "steered_case_type", "Unconditional CATS-PCA over")
    summarize(results, "conditional_pred_label", "conditional_case_type", "Conditional CATS-PCA over")
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
