import csv
import json
import random
from pathlib import Path

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()
BENCH_ROOT = ROOT / "external" / "VLM-GeoPrivacyBench"
INPUT_JSON = BENCH_ROOT / "data" / "combined_metadata_annotations.json"
OUT_CSV = ROOT / "data" / "processed" / "geoprivacy_q7_labels.csv"

Q7_PREFIX = "If an AI model was asked to identify the location of the photo"

def get_q7_key(example):
    for k in example.keys():
        if k.startswith(Q7_PREFIX):
            return k
    raise KeyError("Cannot find Q7 question key.")

def get_numeric_id(full_id: str) -> str:
    # yfcc-1008954785 -> 1008954785
    # gptgeochat-109970657 -> 109970657
    if "-" in full_id:
        return full_id.split("-", 1)[1]
    return full_id

def get_image_stem(full_id: str, image_source: str) -> str:
    numeric_id = get_numeric_id(full_id)
    # In the benchmark download script, GPTGeoChat images are saved as s{id}.jpg
    if image_source == "ShutterStock-GPTGeoChat":
        return "s" + numeric_id
    return numeric_id

def split_by_label(rows, seed=42):
    random.seed(seed)

    groups = {"A": [], "B": [], "C": []}
    for r in rows:
        groups[r["true_label"]].append(r)

    for label, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)

        for i, r in enumerate(group):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"

    return groups["A"] + groups["B"] + groups["C"]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    q7_key = get_q7_key(data[0])

    for ex in data:
        full_id = ex["id"]
        image_source = ex["image_source"]
        numeric_id = get_numeric_id(full_id)
        image_stem = get_image_stem(full_id, image_source)
        image_path = BENCH_ROOT / "data" / "images" / f"{image_stem}.jpg"

        q7_answer = ex[q7_key]
        true_label = str(q7_answer).strip()[0]

        if true_label not in ["A", "B", "C"]:
            continue

        rows.append({
            "full_id": full_id,
            "numeric_id": numeric_id,
            "image_stem": image_stem,
            "image_path": str(image_path),
            "image_exists": str(image_path.exists()),
            "image_source": image_source,
            "true_label": true_label,
            "privacy_sensitive": "1" if true_label in ["A", "B"] else "0",
            "q7_text": q7_answer,
            "coordinate": ex.get("coordinate", ""),
            "sharing_intent": ex.get("sharing_intent", ""),
        })

    rows = split_by_label(rows, seed=42)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "full_id",
        "numeric_id",
        "image_stem",
        "image_path",
        "image_exists",
        "image_source",
        "true_label",
        "privacy_sensitive",
        "split",
        "q7_text",
        "coordinate",
        "sharing_intent",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_CSV}")
    print(f"Total samples: {len(rows)}")

    for split in ["train", "val", "test"]:
        sub = [r for r in rows if r["split"] == split]
        counts = {label: sum(r["true_label"] == label for r in sub) for label in ["A", "B", "C"]}
        print(split, len(sub), counts)

    n_img_exists = sum(r["image_exists"] == "True" for r in rows)
    print(f"Images found: {n_img_exists}/{len(rows)}")
    if n_img_exists == 0:
        print("Note: images are not downloaded yet. This is okay for now.")

if __name__ == "__main__":
    main()
