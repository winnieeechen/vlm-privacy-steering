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

BASE_CSV = ROOT / "outputs" / "02_formal_full1200" / "00_base" / "base_qwen_vl_train_717.csv"
CONDITION_OUT = ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "02_over" / "vector_cases" / "condition_vector_cases_train_717.csv"
BEHAVIOR_OUT = ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "02_over" / "vector_cases" / "behavior_vector_cases_train_717.csv"

CONDITION_POSITIVE = {"A", "B"}
CONDITION_NEGATIVE = {"C"}

BEHAVIOR_POSITIVE = {"A_to_A", "B_to_B"}
BEHAVIOR_NEGATIVE = {"A_to_B", "A_to_C", "B_to_C"}


def normalize_image_path(row):
    image_name = row.get("image_name") or f"{row['image_stem']}.jpg"
    candidates = [
        ROOT / "data" / "02_full1200" / "images" / image_name,
        ROOT / "data" / "images_full1200" / image_name,
        Path(row["image_path"]),
    ]
    for candidate in candidates:
        if candidate.exists():
            rr = dict(row)
            rr["image_path"] = str(candidate)
            rr["image_exists"] = "True"
            rr["image_readable"] = "True"
            return rr
    raise FileNotFoundError(f"Cannot find image for {row.get('full_id')}: {image_name}")


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

    condition_rows = []
    behavior_rows = []

    for row in rows:
        row = normalize_image_path(row)
        true_label = row["true_label"]
        case_type = row["case_type"]

        if true_label in CONDITION_POSITIVE:
            rr = dict(row)
            rr["condition_label"] = "positive"
            condition_rows.append(rr)
        elif true_label in CONDITION_NEGATIVE:
            rr = dict(row)
            rr["condition_label"] = "negative"
            condition_rows.append(rr)

        if case_type in BEHAVIOR_POSITIVE:
            rr = dict(row)
            rr["behavior_label"] = "positive"
            behavior_rows.append(rr)
        elif case_type in BEHAVIOR_NEGATIVE:
            rr = dict(row)
            rr["behavior_label"] = "negative"
            behavior_rows.append(rr)

    write_csv(CONDITION_OUT, condition_rows, list(rows[0].keys()) + ["condition_label"])
    write_csv(BEHAVIOR_OUT, behavior_rows, list(rows[0].keys()) + ["behavior_label"])

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
