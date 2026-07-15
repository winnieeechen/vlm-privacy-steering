#!/usr/bin/env python3
"""Run the YH pairwise-boundary router.

This is not an over/under gate. It routes each example to a target disclosure
level A/B/C using the prompt-token condition vectors, then applies the local
answer-token behavior direction needed for that target:

    target A -> +v_AB
    target B -> no steering
    target C -> -v_BC

The default layer is 28 for the current experiment.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import torch


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent
    raise RuntimeError("Cannot find project root")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "scripts" / "11_pairwise_boundary_vectors" / "00_common"))

from pairwise_boundary_extraction import (  # noqa: E402
    MODEL_NAME,
    QUESTION,
    load_model_dependencies,
    resolve_image_path,
)


LABELS = ["A", "B", "C"]
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}

DEFAULT_BASE = {
    "train": "outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv",
    "val": "outputs/02_formal_full1200/00_base/val/base_qwen_vl_val_238.csv",
    "test": "outputs/02_formal_full1200/00_base/test/base_qwen_vl_test_243.csv",
}


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path):
    with open(project_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def parse_label(text):
    match = re.search(r"\b([ABC])\b", text.strip())
    return match.group(1) if match else "UNKNOWN"


def make_case(true_label, pred_label):
    if pred_label not in {"A", "B", "C"}:
        return f"{true_label}_to_UNKNOWN"
    return f"{true_label}_to_{pred_label}"


def summarize(rows, pred_key, case_key):
    n = len(rows)
    correct = sum(row[case_key] in CORRECT for row in rows)
    over = sum(row[case_key] in OVER for row in rows)
    under = sum(row[case_key] in UNDER for row in rows)
    print("Pred counts:", Counter(row[pred_key] for row in rows))
    print("Case counts:", Counter(row[case_key] for row in rows))
    print(f"Correct: {correct}/{n} = {correct / n:.3f}")
    print(f"Over-disclosure: {over}/{n} = {over / n:.3f}")
    print(f"Under-disclosure: {under}/{n} = {under / n:.3f}")


def ordinal_threshold_accuracy(scores, labels):
    candidates = torch.unique(scores)
    y = torch.tensor([LABELS.index(label) for label in labels])
    best_acc = -1.0
    best = (None, None)
    for i, t1 in enumerate(candidates):
        for t2 in candidates[: i + 1]:
            pred = torch.where(scores >= t1, 0, torch.where(scores >= t2, 1, 2))
            acc = (pred == y).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best = (float(t1), float(t2))
    return best_acc, best


def route_target(score, t1, t2):
    if score >= t1:
        return "A"
    if score >= t2:
        return "B"
    return "C"


def load_condition_axis(layer, method, axis):
    cond_dir = ROOT / "outputs" / "11_pairwise_boundary_vectors" / "02_over" / "vectors"
    c_ab = torch.load(cond_dir / "condition_vectors_A_minus_B.pt", map_location="cpu")["method_vectors"][method][layer].float()
    c_bc = torch.load(cond_dir / "condition_vectors_B_minus_C.pt", map_location="cpu")["method_vectors"][method][layer].float()
    if axis == "ab":
        return unit(c_ab)
    if axis == "bc":
        return unit(c_bc)
    return unit(unit(c_ab) + unit(c_bc))


def load_behavior_vectors(layer, method):
    beh_dir = ROOT / "outputs" / "11_pairwise_boundary_vectors" / "02_over" / "vectors"
    v_ab = torch.load(beh_dir / "behavior_vectors_A_minus_B.pt", map_location="cpu")["method_vectors"][method][layer].float()
    v_bc = torch.load(beh_dir / "behavior_vectors_B_minus_C.pt", map_location="cpu")["method_vectors"][method][layer].float()
    return v_ab, v_bc


def load_train_cache(cache_path, train_csv, layer):
    rows = read_rows(train_csv)
    cache = torch.load(project_path(cache_path), map_location="cpu")
    full_ids = [row["full_id"] for row in rows]
    if cache["full_ids"] != full_ids:
        raise RuntimeError("Train condition cache rows do not match the train base CSV")
    acts = cache["activations"][:, layer, :].float()
    labels = [row["true_label"] for row in rows]
    return acts, labels


def fit_thresholds(layer, method, axis, cache_path, train_csv):
    u = load_condition_axis(layer, method, axis)
    acts, labels = load_train_cache(cache_path, train_csv, layer)
    mu = acts.mean(dim=0)
    scores = (acts - mu) @ u
    acc, (t1, t2) = ordinal_threshold_accuracy(scores, labels)
    return u, mu, t1, t2, acc


def build_inputs(processor, process_vision_info, image_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": QUESTION},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def make_dynamic_hook(state):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steer = state["steer_vec"].to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + steer
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


def report(csv_path):
    rows = read_rows(csv_path)
    print(f"{'full_id':24s} {'true':4s} {'base':4s} {'target':6s} {'steered':7s}")
    for row in rows:
        changed = " <-- changed" if row["steered_pred_label"] != row["pred_label"] else ""
        print(
            f"{row['full_id']:24s} {row['true_label']:4s} {row['pred_label']:4s} "
            f"{row['route_target']:6s} {row['steered_pred_label']:7s}{changed}"
        )
    routing_correct = sum(row["route_target"] == row["true_label"] for row in rows)
    print(f"\nRouting accuracy: {routing_correct}/{len(rows)} = {routing_correct / len(rows):.3f}")
    print("\nBase summary:")
    summarize(rows, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize(rows, "steered_pred_label", "steered_case_type")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--base-csv")
    parser.add_argument("--train-csv", default=DEFAULT_BASE["train"])
    parser.add_argument(
        "--condition-cache",
        default="outputs/04_low_rank_discriminant_vectors/02_over/cache/condition_train_last_token_activations.pt",
        help="Prompt last-token activation cache for fitting router thresholds.",
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--axis", choices=["bisector", "ab", "bc"], default="bisector")
    parser.add_argument("--t1", type=float)
    parser.add_argument("--t2", type=float)
    parser.add_argument("--alpha-a", type=float, default=0.5)
    parser.add_argument("--alpha-c", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.report:
        report(args.report)
        return

    base_csv = args.base_csv or DEFAULT_BASE[args.split]
    out_csv = project_path(
        args.output_csv
        or (
            f"outputs/11_pairwise_boundary_vectors/04_pairwise_router/{args.split}/"
            f"routed_{args.split}_layer{args.layer}_aa{args.alpha_a}_ac{args.alpha_c}_{args.axis}.csv"
        )
    )

    u, mu, fitted_t1, fitted_t2, train_acc = fit_thresholds(
        args.layer,
        args.method,
        args.axis,
        args.condition_cache,
        args.train_csv,
    )
    t1 = fitted_t1 if args.t1 is None else args.t1
    t2 = fitted_t2 if args.t2 is None else args.t2
    print(f"Router thresholds: t1={t1:.4f} t2={t2:.4f} (train routing acc {train_acc:.3f})")

    v_ab, v_bc = load_behavior_vectors(args.layer, args.method)
    print("Layer:", args.layer, "method:", args.method, "axis:", args.axis)
    print("alpha_a:", args.alpha_a, "|v_AB|:", round(float(v_ab.norm()), 4))
    print("alpha_c:", args.alpha_c, "|v_BC|:", round(float(v_bc.norm()), 4))
    print("Base CSV:", project_path(base_csv))
    print("Output:", out_csv)

    rows = read_rows(base_csv)
    if args.max_rows:
        rows = rows[: args.max_rows]

    AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info = load_model_dependencies()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    zero = torch.zeros_like(v_ab)
    steer_by_target = {
        "A": args.alpha_a * v_ab,
        "B": zero,
        "C": -args.alpha_c * v_bc,
    }
    state = {"steer_vec": zero}
    handle = model.model.language_model.layers[args.layer].register_forward_hook(make_dynamic_hook(state))

    results = []
    target_counts = Counter()
    for i, row in enumerate(rows, 1):
        inputs = build_inputs(processor, process_vision_info, resolve_image_path(row)).to(model.device)

        state["steer_vec"] = zero
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        h = outputs.hidden_states[args.layer + 1][0, -1, :].float().cpu()
        score = float((h - mu) @ u)
        target = route_target(score, t1, t2)
        target_counts[target] += 1

        state["steer_vec"] = steer_by_target[target]
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = processor.batch_decode(
            [generated_ids[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        pred = parse_label(answer)
        case = make_case(row["true_label"], pred)

        result = dict(row)
        result["condition_score"] = score
        result["route_target"] = target
        result["router_t1"] = t1
        result["router_t2"] = t2
        result["steered_answer"] = answer
        result["steered_pred_label"] = pred
        result["steered_case_type"] = case
        result["steer_layer"] = args.layer
        result["alpha_a"] = args.alpha_a
        result["alpha_c"] = args.alpha_c
        result["router_axis"] = args.axis
        results.append(result)

        print(
            f"[{i}/{len(rows)}] {row['full_id']:20s} true={row['true_label']} "
            f"base={row['pred_label']} target={target} score={score:+.3f} "
            f"steered={pred} {row['case_type']}->{case}"
        )

    handle.remove()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    routing_correct = sum(row["route_target"] == row["true_label"] for row in results)
    print("\nTarget counts:", dict(target_counts))
    print(f"Routing accuracy: {routing_correct}/{len(results)} = {routing_correct / len(results):.3f}")
    print("\nBase summary:")
    summarize(results, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize(results, "steered_pred_label", "steered_case_type")
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()

