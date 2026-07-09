#!/usr/bin/env python3
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import torch


LABELS = ["A", "B", "C"]
RANK = {"A": 0, "B": 1, "C": 2}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


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


def make_case(true_label, pred_label):
    if pred_label not in RANK:
        return f"{true_label}_to_UNKNOWN"
    return f"{true_label}_to_{pred_label}"


def metrics(rows, pred_key, case_key):
    n = len(rows)
    correct = sum(r[case_key] in CORRECT for r in rows)
    over = sum(r[case_key] in OVER for r in rows)
    under = sum(r[case_key] in UNDER for r in rows)
    return {
        "n": n,
        "correct": correct,
        "correct_rate": correct / n,
        "over": over,
        "over_rate": over / n,
        "under": under,
        "under_rate": under / n,
        "pred_counts": Counter(r[pred_key] for r in rows),
        "case_counts": Counter(r[case_key] for r in rows),
    }


def print_metrics(name, m):
    print(f"\n{name}")
    print("Pred counts:", m["pred_counts"])
    print("Case counts:", m["case_counts"])
    print(f"Correct: {m['correct']}/{m['n']} = {m['correct_rate']:.3f}")
    print(f"Over-disclosure: {m['over']}/{m['n']} = {m['over_rate']:.3f}")
    print(f"Under-disclosure: {m['under']}/{m['n']} = {m['under_rate']:.3f}")


def default_paths(split, layer, alpha_over, alpha_under, tau):
    n = 238 if split == "val" else 243
    base = "outputs/02_formal_full1200"
    method = f"{base}/06_target_aware_probe_guided_steering"
    return {
        "base_csv": f"{base}/00_base/{split}/base_qwen_vl_{split}_{n}.csv",
        "cache": f"{method}/cache/{split}_hidden_states_layer{layer}.pt",
        "probe": f"{method}/probe/granularity_probe_layer{layer}.pkl",
        "over_csv": f"{base}/02_over/{split}/steered_qwen_vl_{split}_layer{layer}_alpha{alpha_over}.csv",
        "under_csv": f"{base}/03_under/{split}/steered_under_qwen_vl_{split}_layer{layer}_alpha{alpha_under}.csv",
        "output": (
            f"{method}/{split}/probe_controller_{split}_layer{layer}_"
            f"tau{tau}_over{alpha_over}_under{alpha_under}.csv"
        ),
    }


def load_probe(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["probe"], payload


def choose_action(base_pred, probe_pred, confidence, tau):
    if confidence < tau:
        return "base"
    if base_pred not in RANK or probe_pred not in RANK:
        return "base"
    if RANK[base_pred] > RANK[probe_pred]:
        return "over"
    if RANK[base_pred] < RANK[probe_pred]:
        return "under"
    return "base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--alpha-over", type=float, default=0.5)
    parser.add_argument("--alpha-under", type=float, default=0.5)
    parser.add_argument("--base-csv")
    parser.add_argument("--hidden-cache")
    parser.add_argument("--probe")
    parser.add_argument("--over-csv")
    parser.add_argument("--under-csv")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    defaults = default_paths(args.split, args.layer, args.alpha_over, args.alpha_under, args.tau)
    base_csv = project_path(args.base_csv or defaults["base_csv"])
    cache_path = project_path(args.hidden_cache or defaults["cache"])
    probe_path = project_path(args.probe or defaults["probe"])
    over_csv = project_path(args.over_csv or defaults["over_csv"])
    under_csv = project_path(args.under_csv or defaults["under_csv"])
    out_csv = project_path(args.output_csv or defaults["output"])

    base_rows = read_csv(base_csv)
    over_rows = read_csv(over_csv)
    under_rows = read_csv(under_csv)
    cache = torch.load(cache_path, map_location="cpu")
    probe, probe_payload = load_probe(probe_path)

    assert len(base_rows) == len(over_rows) == len(under_rows) == len(cache["full_ids"])

    features = cache["features"].float().numpy()
    probs = probe.predict_proba(features)

    results = []
    action_counts = Counter()

    print("Split:", args.split)
    print("Rows:", len(base_rows))
    print("Layer:", args.layer)
    print("Tau:", args.tau)
    print("Alpha over:", args.alpha_over)
    print("Alpha under:", args.alpha_under)
    print("Base CSV:", base_csv)
    print("Probe:", probe_path)
    print("Probe train cache:", probe_payload.get("train_cache"))
    print("Hidden cache:", cache_path)
    print("Over CSV:", over_csv)
    print("Under CSV:", under_csv)
    print("Output:", out_csv)

    for i, (base, over, under, full_id, prob) in enumerate(
        zip(base_rows, over_rows, under_rows, cache["full_ids"], probs), 1
    ):
        assert base["full_id"] == over["full_id"] == under["full_id"] == full_id

        probe_id = int(prob.argmax())
        probe_pred = LABELS[probe_id]
        confidence = float(prob[probe_id])
        base_pred = base["pred_label"]
        action = choose_action(base_pred, probe_pred, confidence, args.tau)
        action_counts[action] += 1

        if action == "over":
            final_answer = over["steered_answer"]
            final_pred = over["steered_pred_label"]
            source_case = over["steered_case_type"]
        elif action == "under":
            final_answer = under["steered_answer"]
            final_pred = under["steered_pred_label"]
            source_case = under["steered_case_type"]
        else:
            final_answer = base["model_answer"]
            final_pred = base_pred
            source_case = base["case_type"]

        final_case = make_case(base["true_label"], final_pred)
        if final_case != source_case:
            source_case = final_case

        rr = dict(base)
        rr["probe_pred_label"] = probe_pred
        rr["probe_confidence"] = confidence
        rr["probe_prob_A"] = float(prob[0])
        rr["probe_prob_B"] = float(prob[1])
        rr["probe_prob_C"] = float(prob[2])
        rr["probe_tau"] = args.tau
        rr["base_pred_level"] = RANK.get(base_pred, "")
        rr["probe_pred_level"] = RANK.get(probe_pred, "")
        rr["chosen_action"] = action
        rr["alpha_over"] = args.alpha_over
        rr["alpha_under"] = args.alpha_under
        rr["final_answer"] = final_answer
        rr["final_pred_label"] = final_pred
        rr["final_case_type"] = source_case
        results.append(rr)

        print(
            f"[{i}/{len(base_rows)}] {base['full_id']} true={base['true_label']} "
            f"base={base_pred} probe={probe_pred} conf={confidence:.3f} "
            f"action={action} final={final_pred} {base['case_type']}->{source_case}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nAction counts:", action_counts)
    print_metrics("Base", metrics(results, "pred_label", "case_type"))
    print_metrics("Probe controller", metrics(results, "final_pred_label", "final_case_type"))
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
