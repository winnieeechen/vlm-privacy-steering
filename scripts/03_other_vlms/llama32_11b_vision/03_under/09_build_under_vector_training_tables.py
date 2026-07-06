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

BASE_CSV = (
    ROOT
    / "outputs"
    / "03_other_vlms"
    / "llama32_11b_vision"
    / "00_base"
    / "train"
    / "base_llama32_vision_train_717.csv"
)

UNDER_CONDITION_OUT = (
    ROOT
    / "outputs"
    / "03_other_vlms"
    / "llama32_11b_vision"
    / "03_under"
    / "vector_cases"
    / "under_condition_vector_cases_train_717.csv"
)
UNDER_BEHAVIOR_OUT = (
    ROOT
    / "outputs"
    / "03_other_vlms"
    / "llama32_11b_vision"
    / "03_under"
    / "vector_cases"
    / "under_behavior_vector_cases_train_717.csv"
)


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
    print("Base CSV:", BASE_CSV)
    print("Case counts:", Counter(r["case_type"] for r in rows))
    print("True label counts:", Counter(r["true_label"] for r in rows))

    condition_rows = []
    behavior_rows = []

    for r in rows:
        true_label = r["true_label"]
        case_type = r["case_type"]

        # Under condition vector:
        # positive = true B/C, negative = true A
        cr = dict(r)
        if true_label in {"B", "C"}:
            cr["condition_label"] = "positive"
        elif true_label == "A":
            cr["condition_label"] = "negative"
        else:
            continue
        condition_rows.append(cr)

        # Under behavior vector:
        # positive = B_to_B, C_to_C
        # negative = B_to_A, C_to_A, C_to_B
        if case_type in {"B_to_B", "C_to_C"}:
            br = dict(r)
            br["behavior_label"] = "positive"
            behavior_rows.append(br)
        elif case_type in {"B_to_A", "C_to_A", "C_to_B"}:
            br = dict(r)
            br["behavior_label"] = "negative"
            behavior_rows.append(br)

    condition_fieldnames = list(rows[0].keys()) + ["condition_label"]
    behavior_fieldnames = list(rows[0].keys()) + ["behavior_label"]

    write_csv(UNDER_CONDITION_OUT, condition_rows, condition_fieldnames)
    write_csv(UNDER_BEHAVIOR_OUT, behavior_rows, behavior_fieldnames)

    print()
    print("Saved under condition cases:", UNDER_CONDITION_OUT)
    print("Under condition rows:", len(condition_rows))
    print("Under condition label counts:", Counter(r["condition_label"] for r in condition_rows))
    print("Under condition true labels:", Counter(r["true_label"] for r in condition_rows))

    print()
    print("Saved under behavior cases:", UNDER_BEHAVIOR_OUT)
    print("Under behavior rows:", len(behavior_rows))
    print("Under behavior label counts:", Counter(r["behavior_label"] for r in behavior_rows))
    print("Under behavior case counts:", Counter(r["case_type"] for r in behavior_rows))


if __name__ == "__main__":
    main()
