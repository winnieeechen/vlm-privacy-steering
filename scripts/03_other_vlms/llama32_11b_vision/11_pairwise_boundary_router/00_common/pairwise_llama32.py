#!/usr/bin/env python3
import argparse
import csv
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
LLAMA_COMMON = ROOT / "scripts" / "03_other_vlms" / "llama32_11b_vision" / "00_common"
LOW_RANK_COMMON = ROOT / "scripts" / "04_low_rank_discriminant_vectors" / "00_common"
for path in (LLAMA_COMMON, LOW_RANK_COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llama32_vision_common import (  # noqa: E402
    LABELS,
    MODEL_NAME,
    build_prompt_inputs,
    build_teacher_forced_inputs,
    language_layers,
    load_model_and_processor,
    make_case,
    make_dynamic_hook,
    parse_label,
    project_path,
    read_csv,
    resolve_image_path,
    run_generation,
    summarize,
    write_csv,
)
from low_rank_vector_extraction import build_vectors  # noqa: E402


LABEL_PAIRS = [("A", "B"), ("B", "C")]
MODEL_ROOT = "outputs/03_other_vlms/llama32_11b_vision"
OUT_ROOT = f"{MODEL_ROOT}/11_pairwise_boundary_router"
TRAIN_CSV = f"{MODEL_ROOT}/00_base/train/base_llama32_vision_train_717.csv"
BASE_CSVS = {
    "train": TRAIN_CSV,
    "val": f"{MODEL_ROOT}/00_base/val/base_llama32_vision_val_238.csv",
    "test": f"{MODEL_ROOT}/00_base/test/base_llama32_vision_test_243.csv",
}
CORRECT = {"A_to_A", "B_to_B", "C_to_C"}
OVER = {"A_to_B", "A_to_C", "B_to_C"}
UNDER = {"B_to_A", "C_to_A", "C_to_B"}


def unit(vector):
    return vector / vector.norm().clamp_min(1e-8)


def normalize_layers(vector):
    return vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)


def print_label_distribution(rows, column):
    counts = Counter(row[column] for row in rows)
    total = sum(counts.values())
    print(f"{column} distribution ({total} rows):")
    for label in sorted(counts):
        count = counts[label]
        print(f"  {label}: {count} ({count / total:.1%})")
    return dict(counts)


def get_answer_mean_vectors(model, processor, row, answer_max_tokens):
    inputs, answer_start = build_teacher_forced_inputs(
        processor,
        row,
        row["model_answer"],
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("Model output did not include hidden_states")

    vectors = []
    for hidden in outputs.hidden_states[1:]:
        answer_end = hidden.shape[1]
        if answer_max_tokens is not None:
            answer_end = min(answer_start + answer_max_tokens, answer_end)
        if answer_end <= answer_start:
            raise RuntimeError(f"No answer tokens for {row.get('full_id')}")
        vectors.append(hidden[0, answer_start:answer_end, :].mean(dim=0).detach().float().cpu())
    return torch.stack(vectors, dim=0)


def get_prompt_last_vectors(model, processor, row):
    inputs = build_prompt_inputs(processor, row).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("Model output did not include hidden_states")
    return torch.stack(
        [hidden[0, -1, :].detach().float().cpu() for hidden in outputs.hidden_states[1:]],
        dim=0,
    )


def save_cache(path, args, rows, activations, labels, complete):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "llama32_pairwise_boundary_activation_cache",
        "model_name": args.model_name,
        "source_csv": str(project_path(args.base_csv)),
        "complete": complete,
        "full_ids": [row["full_id"] for row in rows],
        "rows": rows,
        "label_column": args.label_column,
        "labels": labels,
        "activation_kind": args.activation_kind,
        "answer_max_tokens": args.answer_max_tokens,
        "activations": torch.stack(activations, dim=0).float(),
    }
    torch.save(payload, path)
    print(f"Saved {'complete' if complete else 'partial'} cache:", path)
    return payload


def collect_or_load_activations(args):
    rows = [row for row in read_csv(args.base_csv) if row.get(args.label_column) in LABELS]
    if args.max_rows:
        rows = rows[:args.max_rows]
    cache_path = project_path(args.activation_cache)
    expected_ids = [row["full_id"] for row in rows]

    activations = []
    labels = []
    if cache_path.exists() and not args.refresh_cache:
        cached = torch.load(cache_path, map_location="cpu")
        cached_ids = cached.get("full_ids", [])
        if expected_ids[:len(cached_ids)] != cached_ids:
            raise ValueError("Cache rows do not match requested base CSV")
        if cached.get("activation_kind") != args.activation_kind:
            raise ValueError("Cache activation kind does not match request")
        if cached.get("answer_max_tokens") != args.answer_max_tokens:
            raise ValueError("Cache answer_max_tokens does not match request")
        activations = list(cached["activations"])
        labels = list(cached.get("labels", []))
        if cached.get("complete") and len(cached_ids) == len(expected_ids):
            print("Loaded complete activation cache:", cache_path)
            return cached["activations"].float(), [row[args.label_column] for row in rows], rows
        print(f"Resuming activation cache at {len(activations)}/{len(rows)}")

    print("Rows:", len(rows))
    print("Activation kind:", args.activation_kind)
    print_label_distribution(rows, args.label_column)
    model, processor = load_model_and_processor(args.model_name)
    start = len(activations)
    for index, row in enumerate(rows[start:], start + 1):
        if args.activation_kind == "answer_mean":
            vector = get_answer_mean_vectors(model, processor, row, args.answer_max_tokens)
        elif args.activation_kind == "prompt_last":
            vector = get_prompt_last_vectors(model, processor, row)
        else:
            raise ValueError(f"Unknown activation_kind: {args.activation_kind}")
        activations.append(vector)
        labels.append(row[args.label_column])
        print(
            f"[{index}/{len(rows)}] {row['full_id']} "
            f"true={row['true_label']} pred={row.get('pred_label', '')} "
            f"case={row.get('case_type', '')}"
        )
        if args.save_every and index % args.save_every == 0:
            save_cache(cache_path, args, rows[:index], activations, labels, complete=False)
    payload = save_cache(cache_path, args, rows, activations, labels, complete=True)
    return payload["activations"].float(), [row[args.label_column] for row in rows], rows


def build_pair_payload(args, activations, binary_labels, group, label_counts, vector_key):
    labels_t = torch.tensor(binary_labels, dtype=torch.long)
    pos_mean, neg_mean, variants = build_vectors(
        activations=activations,
        labels=labels_t,
        pca_rank=args.pca_rank,
        residual_rank=args.residual_rank,
        fisher_rank=args.fisher_rank,
        ridge=args.ridge,
    )
    selected = variants[args.method]
    payload = {
        "method": "llama32_pairwise_boundary_vector",
        "method_short_name": "Pairwise Boundary Router",
        "model_name": args.model_name,
        "source_csv": str(project_path(args.base_csv)),
        "activation_cache": str(project_path(args.activation_cache)),
        "activation_kind": args.activation_kind,
        "group": group,
        "num_layers": selected.shape[0],
        "hidden_size": selected.shape[1],
        "positive_count": int((labels_t == 1).sum()),
        "negative_count": int((labels_t == -1).sum()),
        "label_counts": label_counts,
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "selected_method": args.method,
        "method_vectors": variants,
        "method_layer_norms": {name: vector.norm(dim=1) for name, vector in variants.items()},
        "layer_norms": selected.norm(dim=1),
        "hyperparameters": {
            "answer_max_tokens": args.answer_max_tokens,
            "pca_rank": args.pca_rank,
            "residual_rank": args.residual_rank,
            "fisher_rank": args.fisher_rank,
            "ridge": args.ridge,
        },
        vector_key: selected,
        f"{vector_key}_unit": normalize_layers(selected),
    }
    return payload


def save_payload(payload, output):
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print("Saved:", output)
    print("group:", payload["group"])
    print("positive_count:", payload["positive_count"])
    print("negative_count:", payload["negative_count"])
    print("layer28_norm:", float(payload["layer_norms"][28]))


def extraction_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-csv", default=TRAIN_CSV)
    parser.add_argument("--activation-cache")
    parser.add_argument("--activation-kind", choices=["answer_mean", "prompt_last"])
    parser.add_argument("--label-column", choices=["pred_label", "true_label"])
    parser.add_argument("--output-dir")
    parser.add_argument("--vector-key", choices=["behavior_vector", "condition_vector"])
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--method", choices=["mean_diff", "pca_projected", "pca_residual", "fisher_pca", "ensemble"], default="mean_diff")
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--fisher-rank", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser


def run_pair_extraction(args):
    activations, labels, rows = collect_or_load_activations(args)
    label_counts = dict(Counter(labels))
    output_dir = project_path(args.output_dir)
    for pos_label, neg_label in LABEL_PAIRS:
        indices = []
        binary = []
        for index, label in enumerate(labels):
            if label == pos_label:
                indices.append(index)
                binary.append(1)
            elif label == neg_label:
                indices.append(index)
                binary.append(-1)
        group = f"{pos_label}_minus_{neg_label}"
        payload = build_pair_payload(
            args=args,
            activations=activations[torch.tensor(indices, dtype=torch.long)],
            binary_labels=binary,
            group=group,
            label_counts=label_counts,
            vector_key=args.vector_key,
        )
        prefix = "behavior" if args.vector_key == "behavior_vector" else "condition"
        save_payload(payload, output_dir / f"{prefix}_vectors_{group}.pt")


def load_condition_axis(layer, method, axis):
    vector_dir = ROOT / OUT_ROOT / "vectors"
    c_ab = torch.load(vector_dir / "condition_vectors_A_minus_B.pt", map_location="cpu")["method_vectors"][method][layer].float()
    c_bc = torch.load(vector_dir / "condition_vectors_B_minus_C.pt", map_location="cpu")["method_vectors"][method][layer].float()
    if axis == "ab":
        return unit(c_ab)
    if axis == "bc":
        return unit(c_bc)
    return unit(unit(c_ab) + unit(c_bc))


def load_behavior_vectors(layer, method):
    vector_dir = ROOT / OUT_ROOT / "vectors"
    v_ab = torch.load(vector_dir / "behavior_vectors_A_minus_B.pt", map_location="cpu")["method_vectors"][method][layer].float()
    v_bc = torch.load(vector_dir / "behavior_vectors_B_minus_C.pt", map_location="cpu")["method_vectors"][method][layer].float()
    return v_ab, v_bc


def ordinal_threshold_accuracy(scores, labels):
    candidates = torch.unique(scores)
    y = torch.tensor([LABELS.index(label) for label in labels])
    best_acc, best = -1.0, (None, None)
    for i, t1 in enumerate(candidates):
        for t2 in candidates[: i + 1]:
            pred = torch.where(scores >= t1, 0, torch.where(scores >= t2, 1, 2))
            acc = (pred == y).float().mean().item()
            if acc > best_acc:
                best_acc, best = acc, (float(t1), float(t2))
    return best_acc, best


def fit_router(layer, method, axis, condition_cache, train_csv):
    u = load_condition_axis(layer, method, axis)
    cache = torch.load(project_path(condition_cache), map_location="cpu")
    rows = read_csv(train_csv)
    if cache["full_ids"] != [row["full_id"] for row in rows]:
        raise ValueError("Condition cache rows do not match train CSV")
    acts = cache["activations"][:, layer, :].float()
    labels = [row["true_label"] for row in rows]
    mu = acts.mean(dim=0)
    scores = (acts - mu) @ u
    acc, (t1, t2) = ordinal_threshold_accuracy(scores, labels)
    return u, mu, t1, t2, acc


def route_target(score, t1, t2):
    if score >= t1:
        return "A"
    if score >= t2:
        return "B"
    return "C"


def summarize_prediction_rows(rows, pred_key, case_key):
    print("Pred counts:", Counter(row[pred_key] for row in rows))
    summarize(rows, case_key)


def report(csv_path):
    rows = read_csv(csv_path)
    routing = sum(row["route_target"] == row["true_label"] for row in rows)
    print(f"Routing accuracy: {routing}/{len(rows)} = {routing / len(rows):.3f}")
    print("Target counts:", Counter(row["route_target"] for row in rows))
    print("\nBase summary:")
    summarize_prediction_rows(rows, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize_prediction_rows(rows, "steered_pred_label", "steered_case_type")


def run_router(args):
    if args.report:
        report(args.report)
        return
    base_csv = args.base_csv or BASE_CSVS[args.split]
    output = project_path(
        args.output_csv
        or (
            f"{OUT_ROOT}/router/{args.split}/"
            f"routed_llama32_vision_{args.split}_layer{args.layer}_"
            f"aa{args.alpha_a}_ac{args.alpha_c}_{args.axis}.csv"
        )
    )
    existing = read_csv(output) if args.resume and output.exists() else []
    done_ids = {row["full_id"] for row in existing}

    u, mu, fitted_t1, fitted_t2, train_acc = fit_router(
        args.layer,
        args.method,
        args.axis,
        args.condition_cache,
        args.train_csv,
    )
    t1 = fitted_t1 if args.t1 is None else args.t1
    t2 = fitted_t2 if args.t2 is None else args.t2
    v_ab, v_bc = load_behavior_vectors(args.layer, args.method)

    rows = read_csv(base_csv)
    if args.max_rows:
        rows = rows[:args.max_rows]
    print("Rows:", len(rows))
    print("Already complete:", len(done_ids))
    print("Layer:", args.layer, "method:", args.method, "axis:", args.axis)
    print(f"Thresholds: t1={t1:.4f} t2={t2:.4f} train_acc={train_acc:.3f}")
    print("alpha_a:", args.alpha_a, "|v_AB|:", float(v_ab.norm()))
    print("alpha_c:", args.alpha_c, "|v_BC|:", float(v_bc.norm()))
    print("Output:", output)

    model, processor = load_model_and_processor()
    zero = torch.zeros_like(v_ab)
    steer_by_target = {
        "A": args.alpha_a * v_ab,
        "B": zero,
        "C": -args.alpha_c * v_bc,
    }
    state = {"vector": zero}
    layers = language_layers(model)
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"Layer {args.layer} out of range 0..{len(layers) - 1}")
    handle = layers[args.layer].register_forward_hook(make_dynamic_hook(state))

    results = existing
    try:
        for index, row in enumerate(rows, 1):
            if row["full_id"] in done_ids:
                print(f"[{index}/{len(rows)}] skip {row['full_id']}")
                continue

            prompt_inputs = build_prompt_inputs(processor, row).to(model.device)
            state["vector"] = zero
            with torch.no_grad():
                outputs = model(**prompt_inputs, output_hidden_states=True, use_cache=False)
            h = outputs.hidden_states[args.layer + 1][0, -1, :].float().cpu()
            score = float((h - mu) @ u)
            target = route_target(score, t1, t2)

            state["vector"] = steer_by_target[target]
            answer, pred = run_generation(model, processor, row)
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
            write_csv(output, results)
            print(
                f"[{index}/{len(rows)}] {row['full_id']} true={row['true_label']} "
                f"base={row.get('pred_label')} target={target} score={score:+.3f} "
                f"steered={pred} {row.get('case_type')}->{case}"
            )
    finally:
        handle.remove()

    routing = sum(row["route_target"] == row["true_label"] for row in results)
    print("\nRouting accuracy:", f"{routing}/{len(results)} = {routing / len(results):.3f}")
    print("Target counts:", Counter(row["route_target"] for row in results))
    print("\nBase summary:")
    summarize_prediction_rows(results, "pred_label", "case_type")
    print("\nSteered summary:")
    summarize_prediction_rows(results, "steered_pred_label", "steered_case_type")
    print("\nSaved:", output)


def router_parser():
    parser = argparse.ArgumentParser(description="Run Llama-3.2 pairwise A/B/C router steering.")
    parser.add_argument("--report")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--base-csv")
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--condition-cache", default=f"{OUT_ROOT}/cache/prompt_last_activations.pt")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--axis", choices=["bisector", "ab", "bc"], default="bisector")
    parser.add_argument("--t1", type=float)
    parser.add_argument("--t2", type=float)
    parser.add_argument("--alpha-a", type=float, default=1.0)
    parser.add_argument("--alpha-c", type=float, default=1.0)
    parser.add_argument("--output-csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    return parser
