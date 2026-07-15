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

OVER_THRESHOLD = 0.04
UNDER_THRESHOLD = 0.10

OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics(rows, pred_col, case_col):
    pred_counts = Counter(r[pred_col] for r in rows)
    case_counts = Counter(r[case_col] for r in rows)

    correct = sum(r[case_col] in CORRECT for r in rows)
    over = sum(r[case_col] in OVER for r in rows)
    under = sum(r[case_col] in UNDER for r in rows)

    return {
        "pred_counts": pred_counts,
        "case_counts": case_counts,
        "correct": correct,
        "over": over,
        "under": under,
    }


def print_metrics(name, m, n):
    print()
    print(name)
    print("Pred counts:", m["pred_counts"])
    print("Case counts:", m["case_counts"])
    print(f"Correct: {m['correct']}/{n} = {m['correct']/n:.3f}")
    print(f"Over-disclosure: {m['over']}/{n} = {m['over']/n:.3f}")
    print(f"Under-disclosure: {m['under']}/{n} = {m['under']/n:.3f}")


def choose_action(strategy, over_gate, under_gate, over_score, under_score):
    if strategy == "over_first":
        if over_gate:
            return "over"
        if under_gate:
            return "under"
        return "base"

    if strategy == "under_first":
        if under_gate:
            return "under"
        if over_gate:
            return "over"
        return "base"

    if strategy == "conflict_base":
        if over_gate and not under_gate:
            return "over"
        if under_gate and not over_gate:
            return "under"
        return "base"

    if strategy == "conflict_score_margin":
        if over_gate and not under_gate:
            return "over"
        if under_gate and not over_gate:
            return "under"
        if over_gate and under_gate:
            over_margin = over_score - OVER_THRESHOLD
            under_margin = under_score - UNDER_THRESHOLD
            if over_margin >= under_margin:
                return "over"
            return "under"
        return "base"

    raise ValueError(strategy)


def paths_for_split(split):
    n = {"val": 238, "test": 243}[split]
    base = ROOT / (
        f"outputs/03_other_vlms/llama32_11b_vision/00_base/{split}/"
        f"base_llama32_vision_{split}_{n}.csv"
    )
    over_steered = ROOT / (
        f"outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/02_over/{split}/"
        f"steered_llama32_vision_{split}_layer24_alpha1.0.csv"
    )
    under_steered = ROOT / (
        f"outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/03_under/{split}/"
        f"under_steered_llama32_vision_{split}_layer32_alpha0.5.csv"
    )
    over_scores = ROOT / (
        f"outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/02_over/{split}/"
        f"condition_scores_{split}_layer32.csv"
    )
    under_scores = ROOT / (
        f"outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/03_under/{split}/"
        f"under_condition_scores_{split}_layer32.csv"
    )
    out_dir = ROOT / f"outputs/03_other_vlms/llama32_11b_vision/02_mean_behavior_vectors/04_combined_router/{split}"
    return base, over_steered, under_steered, over_scores, under_scores, out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()

    (
        base_csv,
        over_steered_csv,
        under_steered_csv,
        over_score_csv,
        under_score_csv,
        out_dir,
    ) = paths_for_split(args.split)

    base_rows = read_csv(base_csv)
    over_rows = read_csv(over_steered_csv)
    under_rows = read_csv(under_steered_csv)
    over_score_rows = read_csv(over_score_csv)
    under_score_rows = read_csv(under_score_csv)

    assert len(base_rows) == len(over_rows) == len(under_rows) == len(over_score_rows) == len(under_score_rows)

    n = len(base_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Combined cached routing on {args.split.upper()}")
    print("n =", n)
    print("over threshold =", OVER_THRESHOLD)
    print("under threshold =", UNDER_THRESHOLD)

    print_metrics("Base", metrics(base_rows, "pred_label", "case_type"), n)
    print_metrics("Over unconditional", metrics(over_rows, "steered_pred_label", "steered_case_type"), n)
    print_metrics("Under unconditional", metrics(under_rows, "under_pred_label", "under_case_type"), n)

    strategies = [
        "over_first",
        "under_first",
        "conflict_base",
        "conflict_score_margin",
    ]

    summary_rows = []

    for strategy in strategies:
        final_rows = []
        action_counts = Counter()
        gate_counts = Counter()

        for b, o, u, os, us in zip(base_rows, over_rows, under_rows, over_score_rows, under_score_rows):
            assert b["full_id"] == o["full_id"] == u["full_id"] == os["full_id"] == us["full_id"]

            over_score = float(os["condition_score"])
            under_score = float(us["under_condition_score"])

            over_gate = over_score >= OVER_THRESHOLD
            under_gate = under_score >= UNDER_THRESHOLD

            gate_counts[(over_gate, under_gate)] += 1

            action = choose_action(strategy, over_gate, under_gate, over_score, under_score)
            action_counts[action] += 1

            rr = dict(b)
            rr["over_score"] = over_score
            rr["under_score"] = under_score
            rr["over_threshold"] = OVER_THRESHOLD
            rr["under_threshold"] = UNDER_THRESHOLD
            rr["over_gate"] = over_gate
            rr["under_gate"] = under_gate
            rr["combined_strategy"] = strategy
            rr["combined_action"] = action

            if action == "over":
                rr["combined_pred_label"] = o["steered_pred_label"]
                rr["combined_case_type"] = o["steered_case_type"]
            elif action == "under":
                rr["combined_pred_label"] = u["under_pred_label"]
                rr["combined_case_type"] = u["under_case_type"]
            else:
                rr["combined_pred_label"] = b["pred_label"]
                rr["combined_case_type"] = b["case_type"]

            final_rows.append(rr)

        m = metrics(final_rows, "combined_pred_label", "combined_case_type")

        print()
        print("=" * 80)
        print("Strategy:", strategy)
        print("Action counts:", action_counts)
        print("Gate counts:")
        for k, v in gate_counts.items():
            print(" ", k, v)

        print_metrics("Combined " + strategy, m, n)

        out_path = out_dir / f"combined_{args.split}_{strategy}_over24_under32.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)

        summary_rows.append({
            "strategy": strategy,
            "correct": m["correct"],
            "correct_rate": m["correct"] / n,
            "over": m["over"],
            "over_rate": m["over"] / n,
            "under": m["under"],
            "under_rate": m["under"] / n,
            "action_base": action_counts["base"],
            "action_over": action_counts["over"],
            "action_under": action_counts["under"],
            "gate_over_only": gate_counts[(True, False)],
            "gate_under_only": gate_counts[(False, True)],
            "gate_both": gate_counts[(True, True)],
            "gate_neither": gate_counts[(False, False)],
            "output_csv": str(out_path),
        })

    summary_path = out_dir / f"combined_{args.split}_summary_over24_under32.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "correct", "correct_rate",
                "over", "over_rate",
                "under", "under_rate",
                "action_base", "action_over", "action_under",
                "gate_over_only", "gate_under_only", "gate_both", "gate_neither",
                "output_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 80)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
