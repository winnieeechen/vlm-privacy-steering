#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch


RANK = {"A": 0, "B": 1, "C": 2}


def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_layers(vector):
    return vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)


def scale_like(candidate, reference):
    if torch.dot(candidate.flatten(), reference.flatten()) < 0:
        candidate = -candidate
    return candidate / candidate.norm(dim=1, keepdim=True).clamp_min(1e-8) * reference.norm(dim=1, keepdim=True).clamp_min(1e-8)


def severity(row):
    true_label = row.get("true_label")
    pred_label = row.get("pred_label")
    if true_label not in RANK or pred_label not in RANK:
        return 1.0
    return float(max(1, abs(RANK[pred_label] - RANK[true_label])))


def weighted_mean(acts, weights):
    w = torch.tensor(weights, dtype=acts.dtype).view(-1, 1, 1)
    return (acts * w).sum(dim=0) / w.sum().clamp_min(1e-8)


def case_balanced_weights(rows, label_name, severity_beta=0.0):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row[label_name], row["case_type"])].append(i)

    weights = [0.0 for _ in rows]
    for (_label, _case_type), indices in groups.items():
        for idx in indices:
            sev = 1.0 + severity_beta * (severity(rows[idx]) - 1.0)
            weights[idx] = sev / len(indices)
    return weights


def severity_weights(rows, severity_beta):
    return [1.0 + severity_beta * (severity(row) - 1.0) for row in rows]


def mean_diff_from_weights(acts, labels, rows, pos_weights, neg_weights):
    pos_idx = [i for i, label in enumerate(labels.tolist()) if label == 1]
    neg_idx = [i for i, label in enumerate(labels.tolist()) if label == -1]
    pos_mean = weighted_mean(acts[pos_idx], [pos_weights[i] for i in pos_idx])
    neg_mean = weighted_mean(acts[neg_idx], [neg_weights[i] for i in neg_idx])
    return pos_mean - neg_mean


def interpolate(reference, candidate, lam):
    ref_unit = normalize_layers(reference)
    cand_unit = normalize_layers(scale_like(candidate, reference))
    mixed = (1.0 - lam) * ref_unit + lam * cand_unit
    return mixed / mixed.norm(dim=1, keepdim=True).clamp_min(1e-8) * reference.norm(dim=1, keepdim=True).clamp_min(1e-8)


def build_weighted_variants(acts, labels, rows, label_column, severity_beta):
    pos_mask = labels == 1
    neg_mask = labels == -1
    mean_diff = acts[pos_mask].mean(dim=0) - acts[neg_mask].mean(dim=0)

    cb_weights = case_balanced_weights(rows, label_column, severity_beta=0.0)
    cb_sev_weights = case_balanced_weights(rows, label_column, severity_beta=severity_beta)
    sev_weights = severity_weights(rows, severity_beta)
    ones = [1.0 for _ in rows]

    case_balanced = mean_diff_from_weights(acts, labels, rows, cb_weights, cb_weights)
    severity_weighted = mean_diff_from_weights(acts, labels, rows, sev_weights, sev_weights)
    case_balanced_severity = mean_diff_from_weights(acts, labels, rows, cb_sev_weights, cb_sev_weights)

    variants = {
        "mean_diff": mean_diff,
        "case_balanced": scale_like(case_balanced, mean_diff),
        "severity_weighted": scale_like(severity_weighted, mean_diff),
        "case_balanced_severity": scale_like(case_balanced_severity, mean_diff),
    }

    for base_name in ["case_balanced", "severity_weighted", "case_balanced_severity"]:
        for lam in [0.25, 0.5, 0.75, 1.0]:
            name = f"interp_{base_name}_lam{lam:g}"
            variants[name] = interpolate(mean_diff, variants[base_name], lam)

    return variants


def load_cache(cache_path):
    payload = torch.load(cache_path, map_location="cpu")
    return payload["activations"].float(), payload["labels"].long(), payload


def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-csv", default=defaults["case_csv"])
    parser.add_argument("--activation-cache", default=defaults["activation_cache"])
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument("--label-column", default=defaults["label_column"])
    parser.add_argument("--vector-key", default=defaults["vector_key"])
    parser.add_argument("--selected-method", default="interp_case_balanced_lam0.5")
    parser.add_argument("--severity-beta", type=float, default=0.5)
    return parser.parse_args()


def run_extraction(defaults):
    args = parse_args(defaults)
    case_csv = project_path(args.case_csv)
    cache_path = project_path(args.activation_cache)
    out_path = project_path(args.output)

    rows = read_csv(case_csv)
    acts, labels, cache_payload = load_cache(cache_path)
    if len(rows) != acts.shape[0]:
        raise RuntimeError(f"Row/cache mismatch: {len(rows)} rows vs {acts.shape[0]} activations")

    variants = build_weighted_variants(
        acts=acts,
        labels=labels,
        rows=rows,
        label_column=args.label_column,
        severity_beta=args.severity_beta,
    )
    if args.selected_method not in variants:
        raise KeyError(f"Unknown selected method: {args.selected_method}")

    selected = variants[args.selected_method]
    unit_key = f"{args.vector_key}_unit"
    payload = {
        "method": "weighted_mean_behavior_vectors",
        "model_name": cache_payload.get("model_name"),
        "source_csv": str(case_csv),
        "activation_cache": str(cache_path),
        "label_column": args.label_column,
        "selected_method": args.selected_method,
        "severity_beta": args.severity_beta,
        "num_layers": selected.shape[0],
        "hidden_size": selected.shape[1],
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == -1).sum()),
        "label_counts": dict(Counter(row[args.label_column] for row in rows)),
        "case_counts": dict(Counter(row["case_type"] for row in rows)),
        "layer_norms": selected.norm(dim=1),
        "method_vectors": variants,
        "method_layer_norms": {k: v.norm(dim=1) for k, v in variants.items()},
        "definition": {
            "mean_diff": "ordinary positive mean minus negative mean",
            "case_balanced": "average case means so each case_type has equal total weight within its label side",
            "severity_weighted": "sample weights = 1 + beta * (rank_distance - 1)",
            "case_balanced_severity": "case-balanced weights multiplied by severity weight",
            "interp_*": "unit-vector interpolation with mean_diff, rescaled to mean_diff layer norm",
        },
    }
    payload[args.vector_key] = selected
    payload[unit_key] = normalize_layers(selected)

    if args.vector_key == "behavior_vector" and "under" in str(case_csv):
        payload["under_B_behavior_vector"] = payload[args.vector_key]
        payload["under_B_behavior_vector_unit"] = payload[unit_key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print("Saved:", out_path)
    print("selected_method:", args.selected_method)
    print("severity_beta:", args.severity_beta)
    print("label_counts:", payload["label_counts"])
    print("case_counts:", payload["case_counts"])
    print("layer32_norm:", float(payload["layer_norms"][32]))


if __name__ == "__main__":
    raise SystemExit("Use the side-specific wrapper scripts.")
