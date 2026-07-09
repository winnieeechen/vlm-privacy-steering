#!/usr/bin/env python3
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_train_cache_from_existing_activations(cache_path, layer):
    activation_cache = ROOT / (
        "outputs/04_low_rank_discriminant_vectors/02_over/cache/"
        "condition_train_last_token_activations.pt"
    )
    train_csv = ROOT / "outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv"

    if not activation_cache.exists() or not train_csv.exists():
        return False

    activation_payload = torch.load(activation_cache, map_location="cpu")
    rows = read_csv(train_csv)
    row_by_id = {row["full_id"]: row for row in rows}
    full_ids = activation_payload["full_ids"]

    aligned_rows = []
    labels = []
    for full_id in full_ids:
        row = row_by_id[full_id]
        aligned_rows.append(row)
        labels.append(LABEL_TO_ID[row["true_label"]])

    features = activation_payload["activations"][:, layer, :].float()
    payload = {
        "model_name": activation_payload.get("model_name"),
        "split": "train",
        "source_csv": str(train_csv),
        "source_activation_cache": str(activation_cache),
        "layer": layer,
        "features": features,
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": ["A", "B", "C"],
        "full_ids": list(full_ids),
        "rows": aligned_rows,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print("Built train cache from existing 04 activation cache:", cache_path)
    print("features:", tuple(features.shape))
    return True


def load_cache(path):
    if not path.exists():
        layer = int(path.stem.split("layer")[-1])
        if "train_hidden_states" in path.name and build_train_cache_from_existing_activations(path, layer):
            pass
        else:
            raise FileNotFoundError(
                f"Missing hidden-state cache: {path}\n"
                "Run 01_extract_probe_hidden_states.py first, for example:\n"
                "CUDA_VISIBLE_DEVICES=7 python "
                "scripts/02_formal_full1200/06_target_aware_probe_guided_steering/"
                "01_extract_probe_hidden_states.py --split train --layer 32"
            )
    payload = torch.load(path, map_location="cpu")
    x = payload["features"].float().numpy()
    y = payload["labels"].long().numpy()
    return payload, x, y


def write_predictions(path, cache_payload, probs):
    rows = []
    for row, prob in zip(cache_payload["rows"], probs):
        pred_id = int(prob.argmax())
        rr = dict(row)
        rr["probe_pred_label"] = ID_TO_LABEL[pred_id]
        rr["probe_confidence"] = float(prob[pred_id])
        rr["probe_prob_A"] = float(prob[0])
        rr["probe_prob_B"] = float(prob[1])
        rr["probe_prob_C"] = float(prob[2])
        rows.append(rr)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_eval(name, y, pred):
    print(f"\n{name}")
    print("Label counts:", Counter(ID_TO_LABEL[int(v)] for v in y))
    print("Pred counts:", Counter(ID_TO_LABEL[int(v)] for v in pred))
    print("accuracy:", f"{accuracy_score(y, pred):.4f}")
    print("confusion_matrix rows=true cols=pred A/B/C:")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    print(classification_report(y, pred, target_names=["A", "B", "C"], digits=4))


def default_path(kind, layer):
    base = "outputs/02_formal_full1200/06_target_aware_probe_guided_steering"
    paths = {
        "train_cache": f"{base}/cache/train_hidden_states_layer{layer}.pt",
        "val_cache": f"{base}/cache/val_hidden_states_layer{layer}.pt",
        "test_cache": f"{base}/cache/test_hidden_states_layer{layer}.pt",
        "probe": f"{base}/probe/granularity_probe_layer{layer}.pkl",
        "train_pred": f"{base}/probe/probe_predictions_train_layer{layer}.csv",
        "val_pred": f"{base}/probe/probe_predictions_val_layer{layer}.csv",
        "test_pred": f"{base}/probe/probe_predictions_test_layer{layer}.csv",
    }
    return paths[kind]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--train-cache")
    parser.add_argument("--val-cache")
    parser.add_argument("--test-cache")
    parser.add_argument("--output-probe")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()

    train_cache = project_path(args.train_cache or default_path("train_cache", args.layer))
    val_cache = project_path(args.val_cache or default_path("val_cache", args.layer))
    test_cache = project_path(args.test_cache or default_path("test_cache", args.layer))
    out_probe = project_path(args.output_probe or default_path("probe", args.layer))

    train_payload, x_train, y_train = load_cache(train_cache)
    class_weight = None if args.class_weight == "none" else args.class_weight

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            class_weight=class_weight,
            max_iter=args.max_iter,
            solver="lbfgs",
            random_state=0,
        ),
    )
    probe.fit(x_train, y_train)

    train_probs = probe.predict_proba(x_train)
    train_pred = train_probs.argmax(axis=1)
    print_eval("train", y_train, train_pred)
    write_predictions(project_path(default_path("train_pred", args.layer)), train_payload, train_probs)

    eval_paths = [
        ("val", val_cache, default_path("val_pred", args.layer)),
        ("test", test_cache, default_path("test_pred", args.layer)),
    ]
    for name, cache_path, pred_path in eval_paths:
        if cache_path.exists():
            payload, x, y = load_cache(cache_path)
            probs = probe.predict_proba(x)
            pred = probs.argmax(axis=1)
            print_eval(name, y, pred)
            write_predictions(project_path(pred_path), payload, probs)
        else:
            print(f"\n{name}: cache missing, skipped predictions: {cache_path}")

    out_probe.parent.mkdir(parents=True, exist_ok=True)
    with open(out_probe, "wb") as f:
        pickle.dump(
            {
                "probe": probe,
                "layer": args.layer,
                "label_names": ["A", "B", "C"],
                "train_cache": str(train_cache),
                "c": args.c,
                "class_weight": args.class_weight,
            },
            f,
        )
    print("\nSaved probe:", out_probe)


if __name__ == "__main__":
    main()
