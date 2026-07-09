#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
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


def metrics(rows):
    n = len(rows)
    correct = sum(r["final_case_type"] in CORRECT for r in rows)
    over = sum(r["final_case_type"] in OVER for r in rows)
    under = sum(r["final_case_type"] in UNDER for r in rows)
    return correct, correct / n, over, over / n, under, under / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--alpha-over", type=float, default=0.5)
    parser.add_argument("--alpha-under", type=float, default=0.5)
    parser.add_argument(
        "--taus",
        default="0.00,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument(
        "--output-summary",
        default=(
            "outputs/02_formal_full1200/06_target_aware_probe_guided_steering/"
            "val/probe_controller_val_sweep_layer32_alpha0.5.csv"
        ),
    )
    args = parser.parse_args()

    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    runner = ROOT / "scripts/02_formal_full1200/06_target_aware_probe_guided_steering/03_run_probe_controller.py"
    summary_rows = []

    for tau in taus:
        out_csv = (
            ROOT
            / "outputs/02_formal_full1200/06_target_aware_probe_guided_steering/val"
            / f"probe_controller_val_layer{args.layer}_tau{tau}_over{args.alpha_over}_under{args.alpha_under}.csv"
        )
        cmd = [
            sys.executable,
            str(runner),
            "--split",
            "val",
            "--layer",
            str(args.layer),
            "--tau",
            str(tau),
            "--alpha-over",
            str(args.alpha_over),
            "--alpha-under",
            str(args.alpha_under),
            "--output-csv",
            str(out_csv.relative_to(ROOT)),
        ]
        subprocess.run(cmd, check=True)

        rows = read_csv(out_csv)
        correct, correct_rate, over, over_rate, under, under_rate = metrics(rows)
        action_counts = {
            action: sum(r["chosen_action"] == action for r in rows)
            for action in ["base", "over", "under"]
        }
        summary_rows.append({
            "tau": tau,
            "alpha_over": args.alpha_over,
            "alpha_under": args.alpha_under,
            "correct": correct,
            "correct_rate": correct_rate,
            "over": over,
            "over_rate": over_rate,
            "under": under,
            "under_rate": under_rate,
            "action_base": action_counts["base"],
            "action_over": action_counts["over"],
            "action_under": action_counts["under"],
            "output_csv": str(out_csv.relative_to(ROOT)),
        })

    out_summary = ROOT / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSweep summary:")
    for row in summary_rows:
        print(
            f"tau={row['tau']:.2f} correct={row['correct']} "
            f"over={row['over']} under={row['under']} "
            f"actions base/over/under="
            f"{row['action_base']}/{row['action_over']}/{row['action_under']}"
        )
    print("\nSaved:", out_summary)


if __name__ == "__main__":
    main()
