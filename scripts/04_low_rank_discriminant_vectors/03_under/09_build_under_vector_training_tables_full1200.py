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
CONDITION_OUT = ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "03_under" / "vector_cases" / "under_condition_vector_cases_train_717.csv"
BEHAVIOR_OUT = ROOT / "outputs" / "04_low_rank_discriminant_vectors" / "03_under" / "vector_cases" / "under_behavior_vector_cases_train_717.csv"


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
    print("True label counts:", Counter(r["true_label"] for r in rows))

    condition_rows = []
    behavior_rows = []

    for row in rows:
        row = normalize_image_path(row)
        true_label = row["true_label"]
        case_type = row["case_type"]

        cr = dict(row)
        if true_label in {"B", "C"}:
            cr["condition_label"] = "positive"
            condition_rows.append(cr)
        elif true_label == "A":
            cr["condition_label"] = "negative"
            condition_rows.append(cr)

        if case_type in {"B_to_B", "C_to_C"}:
            br = dict(row)
            br["behavior_label"] = "positive"
            behavior_rows.append(br)
        elif case_type in {"B_to_A", "C_to_A", "C_to_B"}:
            br = dict(row)
            br["behavior_label"] = "negative"
            behavior_rows.append(br)

    write_csv(CONDITION_OUT, condition_rows, list(rows[0].keys()) + ["condition_label"])
    write_csv(BEHAVIOR_OUT, behavior_rows, list(rows[0].keys()) + ["behavior_label"])

    print("\nSaved under condition cases:", CONDITION_OUT)
    print("Under condition rows:", len(condition_rows))
    print("Under condition label counts:", Counter(r["condition_label"] for r in condition_rows))
    print("Under condition true labels:", Counter(r["true_label"] for r in condition_rows))

    print("\nSaved under behavior cases:", BEHAVIOR_OUT)
    print("Under behavior rows:", len(behavior_rows))
    print("Under behavior label counts:", Counter(r["behavior_label"] for r in behavior_rows))
    print("Under behavior case counts:", Counter(r["case_type"] for r in behavior_rows))


if __name__ == "__main__":
    main()
