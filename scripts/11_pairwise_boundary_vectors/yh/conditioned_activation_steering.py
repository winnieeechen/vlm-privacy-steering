#!/usr/bin/env python3
"""Condition-routed activation steering: does it move answers toward true_label?

Per example, two passes over one loaded model:
1. Scoring pass: forward the prompt, project the layer-L last-prompt-token
   hidden state onto the shared sensitivity axis u (centered with the train
   mean), and route to a target label with two thresholds:
       s >= t1 -> A,  t2 <= s < t1 -> B,  s < t2 -> C
   Thresholds default to the best in-sample fit on the train activation cache.
2. Steering pass: generate with a forward hook on layer L adding
       target A: +alpha_a * v_AB   (push answers from B toward A)
       target B:  no steering
       target C: -alpha_c * v_BC   (push answers from B toward C)
   where v_AB / v_BC are the answer-token behavior vectors.

Outputs a CSV with per-row scores/targets/preds and prints base vs steered
accuracy summaries.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from condition_projection import (  # noqa: E402
    CACHE as TRAIN_CACHE,
    CONDITION_VECTORS,
    LABELS,
    ordinal_threshold_accuracy,
    unit,
)
from main_behavior import (  # noqa: E402
    MODEL_NAME,
    QUESTION,
    ROOT,
    load_model_dependencies,
    resolve_image_path,
)

BEHAVIOR_VECTORS = {
    "A_minus_B": ROOT / "outputs" / "behavior_vectors" / "vectors" / "behavior_vectors_A_minus_B.pt",
    "B_minus_C": ROOT / "outputs" / "behavior_vectors" / "vectors" / "behavior_vectors_B_minus_C.pt",
}

BASE_CSVS = {
    "train": ROOT / "outputs" / "02_formal_full1200" / "00_base" / "base_qwen_vl_train_717.csv",
    "val": ROOT / "outputs" / "02_formal_full1200" / "00_base" / "val" / "base_qwen_vl_val_238.csv",
    "test": ROOT / "outputs" / "02_formal_full1200" / "00_base" / "test" / "base_qwen_vl_test_243.csv",
}


def parse_label(text):
    m = re.search(r"\b([ABC])\b", text.strip())
    return m.group(1) if m else "UNKNOWN"


def make_case(true_label, pred_label):
    if pred_label not in {"A", "B", "C"}:
        return f"{true_label}_to_UNKNOWN"
    return f"{true_label}_to_{pred_label}"


def build_axis(layer, method, axis):
    v_ab = torch.load(CONDITION_VECTORS["A_minus_B"], map_location="cpu")["method_vectors"][method][layer].float()
    v_bc = torch.load(CONDITION_VECTORS["B_minus_C"], map_location="cpu")["method_vectors"][method][layer].float()
    if axis == "ab":
        return unit(v_ab)
    if axis == "bc":
        return unit(v_bc)
    return unit(unit(v_ab) + unit(v_bc))


def fit_thresholds(layer, u):
    """Fit (t1, t2) on the train activation cache; returns thresholds, train mean, accuracy."""
    cache = torch.load(TRAIN_CACHE, map_location="cpu")
    acts = cache["activations"][:, layer, :].float()
    mu = acts.mean(dim=0)
    s = (acts - mu) @ u
    acc, (t1, t2) = ordinal_threshold_accuracy(s, cache["labels"])
    return t1, t2, mu, acc


def route_target(s, t1, t2):
    if s >= t1:
        return "A"
    if s >= t2:
        return "B"
    return "C"


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


def summarize(rows, pred_key, case_key):
    over = {"A_to_B", "A_to_C", "B_to_C"}
    under = {"B_to_A", "C_to_A", "C_to_B"}
    correct = {"A_to_A", "B_to_B", "C_to_C"}

    n = len(rows)
    n_correct = sum(r[case_key] in correct for r in rows)
    n_over = sum(r[case_key] in over for r in rows)
    n_under = sum(r[case_key] in under for r in rows)

    print("Pred counts:", Counter(r[pred_key] for r in rows))
    print("Case counts:", Counter(r[case_key] for r in rows))
    print(f"Correct: {n_correct}/{n} = {n_correct/n:.3f}")
    print(f"Over-disclosure: {n_over}/{n} = {n_over/n:.3f}")
    print(f"Under-disclosure: {n_under}/{n} = {n_under/n:.3f}")


def report(csv_path):
    """Print per-row true/base/target/steered from an existing results CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{'full_id':24s} {'true':4s} {'base':4s} {'target':6s} {'steered':7s}")
    for r in rows:
        changed = " <-- changed" if r["steered_pred_label"] != r["pred_label"] else ""
        print(
            f"{r['full_id']:24s} {r['true_label']:4s} {r['pred_label']:4s} "
            f"{r['route_target']:6s} {r['steered_pred_label']:7s}{changed}"
        )

    print("\nTransitions by route target (base_pred -> steered_pred):")
    trans = Counter(
        (r["route_target"], r["pred_label"], r["steered_pred_label"]) for r in rows
    )
    for (t, b, s), n in sorted(trans.items()):
        flag = "  <-- changed" if b != s else ""
        print(f"  target={t}  {b} -> {s}  x{n}{flag}")

    routing_correct = sum(r["route_target"] == r["true_label"] for r in rows)
    print(f"\nRouting accuracy (target == true): {routing_correct}/{len(rows)} = {routing_correct/len(rows):.3f}")
    print("\nBase summary:")
    summarize(rows, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize(rows, "steered_pred_label", "steered_case_type")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Read an existing results CSV and print true/base/target/steered; no model run.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--base-csv", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--axis", choices=["bisector", "ab", "bc"], default="bisector")
    parser.add_argument("--t1", type=float, default=None, help="Default: fit on train cache.")
    parser.add_argument("--t2", type=float, default=None)
    parser.add_argument("--alpha-a", type=float, default=0.5, help="Strength for +v_AB when target=A.")
    parser.add_argument("--alpha-c", type=float, default=0.5, help="Strength for -v_BC when target=C.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report is not None:
        report(args.report)
        return

    base_csv = args.base_csv or BASE_CSVS[args.split]
    out_csv = args.output_csv or (
        ROOT / "outputs" / "conditioned_steering" / args.split /
        f"steered_{args.split}_layer{args.layer}_aa{args.alpha_a}_ac{args.alpha_c}.csv"
    )

    # Conditioning vector
    u = build_axis(args.layer, args.method, args.axis)

    if args.t1 is None or args.t2 is None:
        t1, t2, mu, train_acc = fit_thresholds(args.layer, u)
        print(f"Fitted thresholds on train cache: t1={t1:.3f} t2={t2:.3f} (train routing acc {train_acc:.3f})")
    else:
        t1, t2 = args.t1, args.t2
        cache = torch.load(TRAIN_CACHE, map_location="cpu")
        mu = cache["activations"][:, args.layer, :].float().mean(dim=0)

    # Behavior vectors
    v_ab = torch.load(BEHAVIOR_VECTORS["A_minus_B"], map_location="cpu")["method_vectors"][args.method][args.layer].float()
    v_bc = torch.load(BEHAVIOR_VECTORS["B_minus_C"], map_location="cpu")["method_vectors"][args.method][args.layer].float()

    with open(base_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows:
        rows = rows[: args.max_rows]

    print("Split:", args.split, "rows:", len(rows))
    print("Layer:", args.layer, "method:", args.method, "axis:", args.axis)
    print("alpha_a:", args.alpha_a, "|v_AB|:", round(float(v_ab.norm()), 2))
    print("alpha_c:", args.alpha_c, "|v_BC|:", round(float(v_bc.norm()), 2), "(built from only 7 pred-C rows)")
    print("Output:", out_csv)

    AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info = load_model_dependencies()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    zero = torch.zeros_like(v_ab)
    steer_by_target = {"A": args.alpha_a * v_ab, "B": zero, "C": -args.alpha_c * v_bc}
    state = {"steer_vec": zero}
    handle = model.model.language_model.layers[args.layer].register_forward_hook(make_dynamic_hook(state))

    results = []
    target_counts = Counter()

    for i, row in enumerate(rows, 1):
        inputs = build_inputs(processor, process_vision_info, resolve_image_path(row)).to(model.device)

        # Scoring pass (hook active but steering zero).
        state["steer_vec"] = zero
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        h = outputs.hidden_states[args.layer + 1][0, -1, :].float().cpu()
        s = float((h - mu) @ u)
        target = route_target(s, t1, t2)
        target_counts[target] += 1

        # Steering pass.
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

        rr = dict(row)
        rr["condition_score"] = s
        rr["route_target"] = target
        rr["steered_answer"] = answer
        rr["steered_pred_label"] = pred
        rr["steered_case_type"] = case
        results.append(rr)

        print(
            f"[{i}/{len(rows)}] {row['full_id']:20s} "
            f"true={row['true_label']} base_pred={row['pred_label']} "
            f"target={target} (s={s:+.2f}) steered={pred} "
            f"{row['case_type']}->{case}"
        )

    handle.remove()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    routing_correct = sum(r["route_target"] == r["true_label"] for r in results)
    print("\nTarget counts:", dict(target_counts))
    print(f"Routing accuracy (target == true): {routing_correct}/{len(results)} = {routing_correct/len(results):.3f}")

    print("\nBase summary:")
    summarize(results, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize(results, "steered_pred_label", "steered_case_type")
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
