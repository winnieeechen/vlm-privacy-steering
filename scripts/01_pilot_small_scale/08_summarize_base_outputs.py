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
CSV_PATH = ROOT / "outputs" / "base_qwen_vl_40.csv"

with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("Rows:", len(rows))

true_counts = Counter(r["true_label"] for r in rows)
pred_counts = Counter(r["pred_label"] for r in rows)
case_counts = Counter(r["case_type"] for r in rows)

print("\nTrue label counts:")
print(true_counts)

print("\nPred label counts:")
print(pred_counts)

print("\nCase counts:")
print(case_counts)

over_cases = {"A_to_B", "A_to_C", "B_to_C"}
under_cases = {"B_to_A", "C_to_A", "C_to_B"}
correct_cases = {"A_to_A", "B_to_B", "C_to_C"}

n = len(rows)
n_over = sum(r["case_type"] in over_cases for r in rows)
n_under = sum(r["case_type"] in under_cases for r in rows)
n_correct = sum(r["case_type"] in correct_cases for r in rows)

print("\nMetrics:")
print(f"Correct: {n_correct}/{n} = {n_correct / n:.3f}")
print(f"Over-disclosure: {n_over}/{n} = {n_over / n:.3f}")
print(f"Under-disclosure: {n_under}/{n} = {n_under / n:.3f}")

print("\nOver-disclosure examples:")
shown = 0
for r in rows:
    if r["case_type"] in over_cases:
        print(r["full_id"], r["case_type"], "answer=" + r["model_answer"])
        shown += 1
        if shown >= 5:
            break
