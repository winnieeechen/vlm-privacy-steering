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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-csv",
        default="outputs/04_low_rank_discriminant_vectors/00_base/val/base_qwen_vl_val_238.csv",
    )
    parser.add_argument(
        "--under-csv",
        default="outputs/04_low_rank_discriminant_vectors/03_under/val/under_steered_qwen_vl_val_layer32_alpha0.5.csv",
    )
    parser.add_argument(
        "--score-csv",
        default="outputs/04_low_rank_discriminant_vectors/03_under/val/under_condition_scores_val_layer32.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/04_low_rank_discriminant_vectors/03_under/val/under_conditional_sweep_val_layer32_alpha0.5.csv",
    )
    args = parser.parse_args()

    base_rows = read_csv(ROOT / args.base_csv)
    under_rows = read_csv(ROOT / args.under_csv)
    score_rows = read_csv(ROOT / args.score_csv)
    assert len(base_rows) == len(under_rows) == len(score_rows)

    thresholds = [
        -0.080, -0.060, -0.040, -0.020,
        -0.010, 0.000, 0.010, 0.020,
        0.040, 0.060, 0.080, 0.100,
        0.120, 0.140,
    ]

    print_metrics("Base", metrics(base_rows, "pred_label", "case_type"))
    print_metrics("Unconditional under", metrics(under_rows, "under_pred_label", "under_case_type"))

    sweep_results = []
    best = None

    print("\nConditional under sweep:")
    print("threshold, gate_true, correct, over, under")

    for thr in thresholds:
        rows = []
        gate_true = 0

        for b, u, sc in zip(base_rows, under_rows, score_rows):
            assert b["full_id"] == u["full_id"] == sc["full_id"]
            score = float(sc["under_condition_score"])
            use_under = score >= thr

            rr = dict(b)
            rr["under_condition_score"] = score
            rr["threshold"] = thr
            rr["use_under"] = str(use_under)

            if use_under:
                gate_true += 1
                rr["conditional_under_pred_label"] = u["under_pred_label"]
                rr["conditional_under_case_type"] = u["under_case_type"]
            else:
                rr["conditional_under_pred_label"] = b["pred_label"]
                rr["conditional_under_case_type"] = b["case_type"]

            rows.append(rr)

        m = metrics(rows, "conditional_under_pred_label", "conditional_under_case_type")
        print(
            f"{thr:.4f}, {gate_true}, "
            f"{m['correct']}/{m['n']}={m['correct_rate']:.3f}, "
            f"{m['over']}/{m['n']}={m['over_rate']:.3f}, "
            f"{m['under']}/{m['n']}={m['under_rate']:.3f}"
        )

        result = {
            "threshold": thr,
            "gate_true": gate_true,
            "correct": m["correct"],
            "correct_rate": m["correct_rate"],
            "over": m["over"],
            "over_rate": m["over_rate"],
            "under": m["under"],
            "under_rate": m["under_rate"],
            "pred_counts": dict(m["pred_counts"]),
            "case_counts": dict(m["case_counts"]),
        }
        sweep_results.append(result)

        score_for_select = (
            -3.0 * m["over_rate"]
            -2.0 * m["under_rate"]
            +2.0 * m["correct_rate"]
        )
        if best is None or score_for_select > best[0]:
            best = (score_for_select, thr, m, gate_true)

    out_path = ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "threshold", "gate_true",
            "correct", "correct_rate",
            "over", "over_rate",
            "under", "under_rate",
            "pred_counts", "case_counts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    print("\nBest by simple selection rule:")
    _, best_thr, best_m, best_gate = best
    print("threshold:", best_thr)
    print("gate_true:", best_gate)
    print_metrics("Best conditional under", best_m)
    print("\nSaved sweep:", out_path)


if __name__ == "__main__":
    main()
