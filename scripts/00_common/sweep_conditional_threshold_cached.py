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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep one conditional gate using cached base and steered outputs."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--steered-csv", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--score-key", required=True)
    parser.add_argument("--steered-case-key", required=True)
    parser.add_argument("--error-priority", choices=["over", "under"], required=True)
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
    steered = read_csv(args.steered_csv)
    scores = read_csv(args.score_csv)
    if not (len(base) == len(steered) == len(scores)):
        raise ValueError("Base, steered, and score CSV lengths do not match")
    for base_row, steered_row, score_row in zip(base, steered, scores):
        if not (
            base_row["full_id"]
            == steered_row["full_id"]
            == score_row["full_id"]
        ):
            raise ValueError("CSV row order/full_id mismatch")

    score_values = [float(row[args.score_key]) for row in scores]
    thresholds = sorted({quantile(score_values, q) for q in args.quantiles})
    results = []
    n = len(base)
    for threshold in thresholds:
        cases = []
        gate_true = 0
        for base_row, steered_row, score_row in zip(base, steered, scores):
            use_steer = float(score_row[args.score_key]) >= threshold
            gate_true += use_steer
            cases.append(
                steered_row[args.steered_case_key]
                if use_steer
                else base_row["case_type"]
            )
        correct = sum(case in CORRECT for case in cases)
        over = sum(case in OVER for case in cases)
        under = sum(case in UNDER for case in cases)
        gate_rate = gate_true / n
        results.append({
            "threshold": threshold,
            "gate_true": gate_true,
            "gate_rate": gate_rate,
            "correct": correct,
            "correct_rate": correct / n,
            "over": over,
            "over_rate": over / n,
            "under": under,
            "under_rate": under / n,
            "eligible": args.min_gate_rate <= gate_rate <= args.max_gate_rate,
            "selected": False,
        })

    eligible = [row for row in results if row["eligible"]]
    if not eligible:
        raise RuntimeError("No threshold satisfies the gate-rate limits")
    secondary = args.error_priority
    tertiary = "under" if secondary == "over" else "over"
    best = max(
        eligible,
        key=lambda row: (
            row["correct"],
            -row[secondary],
            -row[tertiary],
            -row["gate_rate"],
        ),
    )
    best["selected"] = True

    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("Selected validation threshold")
    print(f"threshold = {best['threshold']:.8f}")
    print(f"gate = {best['gate_true']}/{n} = {best['gate_rate']:.3f}")
    print(
        f"correct / over / under = "
        f"{best['correct']} / {best['over']} / {best['under']}"
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
