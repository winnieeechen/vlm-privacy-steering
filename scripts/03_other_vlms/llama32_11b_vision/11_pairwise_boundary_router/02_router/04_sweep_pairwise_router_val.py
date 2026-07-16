#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()
MODEL_ROOT = ROOT / "outputs" / "03_other_vlms" / "llama32_11b_vision"
OUT_DIR = MODEL_ROOT / "11_pairwise_boundary_router" / "router" / "val"
RUNNER = (
    ROOT
    / "scripts"
    / "03_other_vlms"
    / "llama32_11b_vision"
    / "11_pairwise_boundary_router"
    / "02_router"
    / "03_run_pairwise_router_steering.py"
)

CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}


def output_path(layer, alpha_a, alpha_c, axis):
    return OUT_DIR / (
        f"routed_llama32_vision_val_layer{layer}_"
        f"aa{alpha_a}_ac{alpha_c}_{axis}.csv"
    )


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(path):
    rows = read_csv(path)
    n = len(rows)
    correct = sum(row["steered_case_type"] in CORRECT for row in rows)
    over = sum(row["steered_case_type"] in OVER for row in rows)
    under = sum(row["steered_case_type"] in UNDER for row in rows)
    route_correct = sum(row["route_target"] == row["true_label"] for row in rows)
    targets = Counter(row["route_target"] for row in rows)
    preds = Counter(row["steered_pred_label"] for row in rows)
    match = re.search(r"layer(\d+)_aa([0-9.]+)_ac([0-9.]+)_([^/]+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse sweep name: {path}")
    layer, alpha_a, alpha_c, axis = match.groups()
    return {
        "path": path,
        "layer": int(layer),
        "alpha_a": float(alpha_a),
        "alpha_c": float(alpha_c),
        "axis": axis,
        "n": n,
        "correct": correct,
        "correct_rate": correct / n,
        "over": over,
        "over_rate": over / n,
        "under": under,
        "under_rate": under / n,
        "route_correct": route_correct,
        "route_acc": route_correct / n,
        "targets": dict(targets),
        "preds": dict(preds),
    }


def summarize(pattern="routed_llama32_vision_val_layer*_aa*_ac*_*.csv"):
    files = sorted(OUT_DIR.glob(pattern))
    if not files:
        print("No sweep CSVs found in:", OUT_DIR)
        return None
    rows = [metrics(path) for path in files]
    rows.sort(key=lambda row: (row["correct"], -row["over"], -row["under"], -row["alpha_a"], -row["alpha_c"]), reverse=True)

    print("\nSorted by correct desc, over asc, under asc:")
    print("layer aa ac axis correct over under route_acc targets preds file")
    for row in rows:
        print(
            f"{row['layer']:<5d} {row['alpha_a']:<4g} {row['alpha_c']:<4g} "
            f"{row['axis']:<8s} "
            f"{row['correct']:3d}/{row['n']} {row['correct_rate']:.3f} "
            f"over={row['over']:3d} under={row['under']:3d} "
            f"route={row['route_acc']:.3f} "
            f"targets={row['targets']} preds={row['preds']} "
            f"{row['path'].name}"
        )

    best = rows[0]
    print("\nBest:")
    print(
        f"layer={best['layer']} alpha_a={best['alpha_a']} alpha_c={best['alpha_c']} "
        f"axis={best['axis']} correct={best['correct']}/{best['n']} "
        f"over={best['over']} under={best['under']}"
    )
    print("CSV:", best["path"])
    return best


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search layer/alpha settings for Llama-3.2 pairwise router on val."
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[20, 24, 28, 32, 36])
    parser.add_argument("--alpha-a", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--alpha-c", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--axis", choices=["bisector", "ab", "bc"], default="bisector")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.summary_only:
        summarize()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(args.layers) * len(args.alpha_a) * len(args.alpha_c)
    step = 0
    for layer in args.layers:
        for alpha_a in args.alpha_a:
            for alpha_c in args.alpha_c:
                step += 1
                output = output_path(layer, alpha_a, alpha_c, args.axis)
                if args.skip_existing and output.exists():
                    print(f"[{step}/{total}] skip existing {output}")
                    continue
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--split",
                    "val",
                    "--layer",
                    str(layer),
                    "--method",
                    args.method,
                    "--axis",
                    args.axis,
                    "--alpha-a",
                    str(alpha_a),
                    "--alpha-c",
                    str(alpha_c),
                    "--output-csv",
                    str(output),
                ]
                if args.resume:
                    command.append("--resume")
                if args.max_rows:
                    command.extend(["--max-rows", str(args.max_rows)])
                print(f"\n[{step}/{total}] Command:", " ".join(command))
                subprocess.run(command, check=True)

    summarize()


if __name__ == "__main__":
    main()

