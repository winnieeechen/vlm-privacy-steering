import csv
from pathlib import Path
from collections import Counter

def find_project_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")

ROOT = find_project_root()

BASE_CSV = ROOT / "outputs/02_formal_full1200/02_over/base_qwen_vl_val_238.csv"

OVER_STEERED_CSV = ROOT / "outputs/02_formal_full1200/02_over/steered_qwen_vl_val_layer32_alpha0.5.csv"
UNDER_STEERED_CSV = ROOT / "outputs/02_formal_full1200/03_under/steered_under_qwen_vl_val_layer32_alpha0.5.csv"

OVER_SCORE_CSV = ROOT / "outputs/02_formal_full1200/02_over/condition_scores_val_layer32.csv"
UNDER_SCORE_CSV = ROOT / "outputs/02_formal_full1200/03_under/under_condition_scores_val_layer32.csv"

OUT_DIR = ROOT / "outputs/02_formal_full1200/04_combined_router/val"

OVER_THRESHOLD = -0.03
UNDER_THRESHOLD = 0.04

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def metrics(rows, pred_col, case_col):
    level = {"A": 0, "B": 1, "C": 2}

    pred_counts = Counter(r[pred_col] for r in rows)
    case_counts = Counter(r[case_col] for r in rows)

    correct = over = under = 0

    for r in rows:
        true_label, pred_label = r[case_col].split("_to_")
        if level[pred_label] == level[true_label]:
            correct += 1
        elif level[pred_label] > level[true_label]:
            over += 1
        else:
            under += 1

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
            else:
                return "under"
        return "base"

    raise ValueError(strategy)

def main():
    base_rows = read_csv(BASE_CSV)
    over_rows = read_csv(OVER_STEERED_CSV)
    under_rows = read_csv(UNDER_STEERED_CSV)
    over_score_rows = read_csv(OVER_SCORE_CSV)
    under_score_rows = read_csv(UNDER_SCORE_CSV)

    assert len(base_rows) == len(over_rows) == len(under_rows) == len(over_score_rows) == len(under_score_rows)

    n = len(base_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Combined cached routing on VAL")
    print("n =", n)
    print("over threshold =", OVER_THRESHOLD)
    print("under threshold =", UNDER_THRESHOLD)

    print_metrics("Base", metrics(base_rows, "pred_label", "case_type"), n)
    print_metrics("Over unconditional", metrics(over_rows, "steered_pred_label", "steered_case_type"), n)
    print_metrics("Under unconditional", metrics(under_rows, "steered_pred_label", "steered_case_type"), n)

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
            over_score = float(os["condition_score"])
            under_score = float(us["condition_score"])

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
                rr["combined_pred_label"] = u["steered_pred_label"]
                rr["combined_case_type"] = u["steered_case_type"]
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

        out_path = OUT_DIR / f"combined_val_{strategy}_layer32_alpha0.5.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)

        summary_rows.append({
            "strategy": strategy,
            "correct": m["correct"],
            "over": m["over"],
            "under": m["under"],
            "action_base": action_counts["base"],
            "action_over": action_counts["over"],
            "action_under": action_counts["under"],
            "output_csv": str(out_path),
        })

    summary_path = OUT_DIR / "combined_val_summary_layer32_alpha0.5.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "correct",
                "over",
                "under",
                "action_base",
                "action_over",
                "action_under",
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
