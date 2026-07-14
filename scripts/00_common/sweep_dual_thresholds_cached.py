#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with open(project_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(cases):
    return (
        sum(case in CORRECT for case in cases),
        sum(case in OVER for case in cases),
        sum(case in UNDER for case in cases),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep dual condition thresholds using four cached action CSVs."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--over-csv", required=True)
    parser.add_argument("--under-csv", required=True)
    parser.add_argument("--both-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--over-case-key", default="steered_case_type")
    parser.add_argument("--under-case-key", default="under_case_type")
    parser.add_argument("--both-case-key", default="dual_additive_case_type")
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--min-gate-rate", type=float, default=0.1)
    parser.add_argument("--max-gate-rate", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    base = read_csv(args.base_csv)
    over = read_csv(args.over_csv)
    under = read_csv(args.under_csv)
    both = read_csv(args.both_csv)
    scores = read_csv(args.score_csv)
    if not (len(base) == len(over) == len(under) == len(both) == len(scores)):
        raise ValueError("All cached CSVs must contain the same number of rows")
    for rows in (over, under, both, scores):
        for base_row, row in zip(base, rows):
            if base_row["full_id"] != row["full_id"]:
                raise ValueError("CSV row order/full_id mismatch")

    over_scores = [float(row["condition_score"]) for row in scores]
    under_scores = [float(row["under_condition_score"]) for row in scores]
    over_thresholds = sorted({quantile(over_scores, q) for q in args.quantiles})
    under_thresholds = sorted({quantile(under_scores, q) for q in args.quantiles})

    results = []
    n = len(base)
    for over_threshold in over_thresholds:
        for under_threshold in under_thresholds:
            cases = []
            over_count = 0
            under_count = 0
            both_count = 0
            neither_count = 0
            for b, o, u, d, score in zip(base, over, under, both, scores):
                over_gate = float(score["condition_score"]) >= over_threshold
                under_gate = float(score["under_condition_score"]) >= under_threshold
                over_count += over_gate
                under_count += under_gate
                if over_gate and under_gate:
                    both_count += 1
                    cases.append(d[args.both_case_key])
                elif over_gate:
                    cases.append(o[args.over_case_key])
                elif under_gate:
                    cases.append(u[args.under_case_key])
                else:
                    neither_count += 1
                    cases.append(b["case_type"])
            correct, over_errors, under_errors = metrics(cases)
            over_rate = over_count / n
            under_rate = under_count / n
            eligible = (
                args.min_gate_rate <= over_rate <= args.max_gate_rate
                and args.min_gate_rate <= under_rate <= args.max_gate_rate
            )
            results.append({
                "over_threshold": over_threshold,
                "under_threshold": under_threshold,
                "over_gate": over_count,
                "over_gate_rate": over_rate,
                "under_gate": under_count,
                "under_gate_rate": under_rate,
                "both_gate": both_count,
                "neither_gate": neither_count,
                "correct": correct,
                "correct_rate": correct / n,
                "over": over_errors,
                "over_rate": over_errors / n,
                "under": under_errors,
                "under_rate": under_errors / n,
                "eligible": eligible,
                "selected": False,
            })

    eligible = [row for row in results if row["eligible"]]
    if not eligible:
        raise RuntimeError("No threshold pair satisfies the gate-rate limits")
    best = max(
        eligible,
        key=lambda row: (
            row["correct"],
            -row["over"],
            -row["under"],
            -row["over_gate_rate"] - row["under_gate_rate"],
        ),
    )
    best["selected"] = True

    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("Selected validation thresholds")
    print(f"over_threshold  = {best['over_threshold']:.8f}")
    print(f"under_threshold = {best['under_threshold']:.8f}")
    print(
        f"correct / over / under = "
        f"{best['correct']} / {best['over']} / {best['under']}"
    )
    print(
        f"over gate / under gate / both / neither = "
        f"{best['over_gate']} / {best['under_gate']} / "
        f"{best['both_gate']} / {best['neither_gate']}"
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
