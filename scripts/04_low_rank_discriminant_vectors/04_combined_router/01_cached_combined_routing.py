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

OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics(rows, pred_col, case_col):
    n = len(rows)
    correct = sum(r[case_col] in CORRECT for r in rows)
    over = sum(r[case_col] in OVER for r in rows)
    under = sum(r[case_col] in UNDER for r in rows)
    return {
        "pred_counts": Counter(r[pred_col] for r in rows),
        "case_counts": Counter(r[case_col] for r in rows),
        "correct": correct,
        "correct_rate": correct / n,
        "over": over,
        "over_rate": over / n,
        "under": under,
        "under_rate": under / n,
    }


def print_metrics(name, m, n):
    print()
    print(name)
    print("Pred counts:", m["pred_counts"])
    print("Case counts:", m["case_counts"])
    print(f"Correct: {m['correct']}/{n} = {m['correct_rate']:.3f}")
    print(f"Over-disclosure: {m['over']}/{n} = {m['over_rate']:.3f}")
    print(f"Under-disclosure: {m['under']}/{n} = {m['under_rate']:.3f}")


def score_value(row, preferred_key, fallback_key):
    if preferred_key in row:
        return float(row[preferred_key])
    return float(row[fallback_key])


def choose_action(strategy, over_gate, under_gate, over_score, under_score, over_threshold, under_threshold):
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
            over_margin = over_score - over_threshold
            under_margin = under_score - under_threshold
            return "over" if over_margin >= under_margin else "under"
        return "base"

    raise ValueError(strategy)


def default_path(split, kind):
    n = 238 if split == "val" else 243
    base = f"outputs/04_low_rank_discriminant_vectors/00_base/{split}/base_qwen_vl_{split}_{n}.csv"
    paths = {
        "base": base,
        "over": f"outputs/04_low_rank_discriminant_vectors/02_over/{split}/steered_qwen_vl_{split}_layer32_alpha0.5.csv",
        "under": f"outputs/04_low_rank_discriminant_vectors/03_under/{split}/under_steered_qwen_vl_{split}_layer32_alpha0.5.csv",
        "over_score": f"outputs/04_low_rank_discriminant_vectors/02_over/{split}/condition_scores_{split}_layer32.csv",
        "under_score": f"outputs/04_low_rank_discriminant_vectors/03_under/{split}/under_condition_scores_{split}_layer32.csv",
        "out_dir": f"outputs/04_low_rank_discriminant_vectors/04_combined_router/{split}",
    }
    return paths[kind]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--base-csv")
    parser.add_argument("--over-csv")
    parser.add_argument("--under-csv")
    parser.add_argument("--over-score-csv")
    parser.add_argument("--under-score-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--over-threshold", type=float, default=-0.12)
    parser.add_argument("--under-threshold", type=float, default=-0.08)
    return parser.parse_args()


def main():
    args = parse_args()

    base_csv = ROOT / (args.base_csv or default_path(args.split, "base"))
    over_csv = ROOT / (args.over_csv or default_path(args.split, "over"))
    under_csv = ROOT / (args.under_csv or default_path(args.split, "under"))
    over_score_csv = ROOT / (args.over_score_csv or default_path(args.split, "over_score"))
    under_score_csv = ROOT / (args.under_score_csv or default_path(args.split, "under_score"))
    out_dir = ROOT / (args.output_dir or default_path(args.split, "out_dir"))

    base_rows = read_csv(base_csv)
    over_rows = read_csv(over_csv)
    under_rows = read_csv(under_csv)
    over_score_rows = read_csv(over_score_csv)
    under_score_rows = read_csv(under_score_csv)

    assert len(base_rows) == len(over_rows) == len(under_rows) == len(over_score_rows) == len(under_score_rows)

    n = len(base_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Combined cached routing on {args.split.upper()}")
    print("n =", n)
    print("base CSV:", base_csv)
    print("over CSV:", over_csv)
    print("under CSV:", under_csv)
    print("over score CSV:", over_score_csv)
    print("under score CSV:", under_score_csv)
    print("over threshold =", args.over_threshold)
    print("under threshold =", args.under_threshold)

    print_metrics("Base", metrics(base_rows, "pred_label", "case_type"), n)
    print_metrics("Over unconditional", metrics(over_rows, "steered_pred_label", "steered_case_type"), n)
    print_metrics("Under unconditional", metrics(under_rows, "under_pred_label", "under_case_type"), n)

    strategies = ["over_first", "under_first", "conflict_base", "conflict_score_margin"]
    summary_rows = []

    for strategy in strategies:
        final_rows = []
        action_counts = Counter()
        gate_counts = Counter()

        for b, o, u, os, us in zip(base_rows, over_rows, under_rows, over_score_rows, under_score_rows):
            assert b["full_id"] == o["full_id"] == u["full_id"] == os["full_id"] == us["full_id"]

            over_score = score_value(os, "condition_score", "over_condition_score")
            under_score = score_value(us, "under_condition_score", "condition_score")
            over_gate = over_score >= args.over_threshold
            under_gate = under_score >= args.under_threshold

            gate_counts[(over_gate, under_gate)] += 1
            action = choose_action(
                strategy,
                over_gate,
                under_gate,
                over_score,
                under_score,
                args.over_threshold,
                args.under_threshold,
            )
            action_counts[action] += 1

            rr = dict(b)
            rr["over_score"] = over_score
            rr["under_score"] = under_score
            rr["over_threshold"] = args.over_threshold
            rr["under_threshold"] = args.under_threshold
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

        out_path = out_dir / f"combined_{args.split}_{strategy}_layer32_alpha0.5_over{args.over_threshold}_under{args.under_threshold}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)

        summary_rows.append({
            "strategy": strategy,
            "correct": m["correct"],
            "correct_rate": m["correct_rate"],
            "over": m["over"],
            "over_rate": m["over_rate"],
            "under": m["under"],
            "under_rate": m["under_rate"],
            "action_base": action_counts["base"],
            "action_over": action_counts["over"],
            "action_under": action_counts["under"],
            "gate_false_false": gate_counts[(False, False)],
            "gate_true_false": gate_counts[(True, False)],
            "gate_false_true": gate_counts[(False, True)],
            "gate_true_true": gate_counts[(True, True)],
            "output_csv": str(out_path),
        })

    summary_path = out_dir / f"combined_{args.split}_summary_layer32_alpha0.5_over{args.over_threshold}_under{args.under_threshold}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "strategy",
            "correct",
            "correct_rate",
            "over",
            "over_rate",
            "under",
            "under_rate",
            "action_base",
            "action_over",
            "action_under",
            "gate_false_false",
            "gate_true_false",
            "gate_false_true",
            "gate_true_true",
            "output_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 80)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
