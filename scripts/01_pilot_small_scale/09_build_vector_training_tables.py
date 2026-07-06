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

BASE_CSV = ROOT / "outputs" / "base_qwen_vl_40.csv"

CONDITION_OUT = ROOT / "data" / "processed" / "condition_vector_cases_qwen_vl_40.csv"
BEHAVIOR_OUT = ROOT / "data" / "processed" / "behavior_vector_cases_qwen_vl_40.csv"

CONDITION_POSITIVE = {"A", "B"}
CONDITION_NEGATIVE = {"C"}

BEHAVIOR_POSITIVE = {"A_to_A", "B_to_B"}
BEHAVIOR_NEGATIVE = {"A_to_B", "A_to_C", "B_to_C"}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with open(BASE_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("Loaded base results:", len(rows))
    print("Case counts:", Counter(r["case_type"] for r in rows))

    condition_rows = []
    behavior_rows = []

    for r in rows:
        true_label = r["true_label"]
        case = r["case_type"]

        # Condition vector cases:
        # positive = privacy-sensitive input A/B
        # negative = disclosure-allowed input C
        if true_label in CONDITION_POSITIVE:
            rr = dict(r)
            rr["condition_label"] = "positive"
            condition_rows.append(rr)
        elif true_label in CONDITION_NEGATIVE:
            rr = dict(r)
            rr["condition_label"] = "negative"
            condition_rows.append(rr)

        # Behavior vector cases:
        # positive = correct privacy-preserving answer
        # negative = over-disclosure answer
        if case in BEHAVIOR_POSITIVE:
            rr = dict(r)
            rr["behavior_label"] = "positive"
            behavior_rows.append(rr)
        elif case in BEHAVIOR_NEGATIVE:
            rr = dict(r)
            rr["behavior_label"] = "negative"
            behavior_rows.append(rr)

    condition_fieldnames = list(rows[0].keys()) + ["condition_label"]
    behavior_fieldnames = list(rows[0].keys()) + ["behavior_label"]

    write_csv(CONDITION_OUT, condition_rows, condition_fieldnames)
    write_csv(BEHAVIOR_OUT, behavior_rows, behavior_fieldnames)

    print("\nSaved condition cases:", CONDITION_OUT)
    print("Condition rows:", len(condition_rows))
    print("Condition label counts:", Counter(r["condition_label"] for r in condition_rows))
    print("Condition true labels:", Counter(r["true_label"] for r in condition_rows))

    print("\nSaved behavior cases:", BEHAVIOR_OUT)
    print("Behavior rows:", len(behavior_rows))
    print("Behavior label counts:", Counter(r["behavior_label"] for r in behavior_rows))
    print("Behavior case counts:", Counter(r["case_type"] for r in behavior_rows))


if __name__ == "__main__":
    main()
