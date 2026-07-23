#!/usr/bin/env python3
"""Run attentive activation steering with adaptive A and C strengths."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import torch

import method14_common as common

ROOT = common.ROOT
LABELS = common.LABELS


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path):
    with open(project_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize(value):
    return value / value.norm().clamp_min(1e-8)


def load_global_vector(vector_dir, pair, layer):
    payload = torch.load(
        vector_dir / f"behavior_vectors_{pair}.pt", map_location="cpu"
    )
    return payload["method_vectors"]["mean_diff"][layer].float()


def route(hidden, assets):
    feature = normalize(hidden - assets["route_center"])
    similarities = assets["route_features"] @ feature
    indices = similarities.topk(assets["route_neighbors"]).indices
    votes = torch.nn.functional.one_hot(
        assets["route_labels"][indices], num_classes=3
    ).float().sum(0)
    probabilities = votes / votes.sum()
    return LABELS[int(probabilities.argmax())], feature, probabilities


def attentive_centroid(feature, label_index, assets, neighbors, temperature):
    indices = torch.where(assets["memory_labels"] == label_index)[0]
    similarities = assets["memory_condition"][indices] @ feature
    top = similarities.topk(min(neighbors, len(indices)))
    weights = torch.softmax(top.values / temperature, dim=0)
    selected = assets["memory_behavior"][indices[top.indices]]
    return (weights.unsqueeze(1) * selected).sum(0)


def query_vector(feature, target, assets, global_vector, args):
    if target == "A":
        local = attentive_centroid(feature, 0, assets, args.expert_neighbors, args.expert_temperature)
        local -= attentive_centroid(feature, 1, assets, args.expert_neighbors, args.expert_temperature)
    elif target == "C":
        local = attentive_centroid(feature, 2, assets, args.expert_neighbors, args.expert_temperature)
        local -= attentive_centroid(feature, 1, assets, args.expert_neighbors, args.expert_temperature)
    else:
        return torch.zeros_like(global_vector), 0.0
    cosine = float(torch.dot(normalize(local), normalize(global_vector)))
    if cosine < 0:
        local = -local
        cosine = -cosine
    local = local * (global_vector.norm() / local.norm().clamp_min(1e-8))
    # A poorly aligned local prototype is weak evidence. Let it contribute only
    # in proportion to its agreement with the reliable global direction.
    effective_blend = args.prototype_blend * cosine
    vector = (1 - effective_blend) * global_vector + effective_blend * local
    vector = vector * (global_vector.norm() / vector.norm().clamp_min(1e-8))
    return vector, cosine, effective_blend


def validate_resume(rows, args, assets):
    if not rows:
        return
    expected_floats = {
        "min_confidence_a": args.min_confidence_a,
        "min_confidence_c": args.min_confidence_c,
        "alpha_a_low": args.alpha_a_low,
        "alpha_a_high": args.alpha_a_high,
        "alpha_c_low": args.alpha_c_low,
        "alpha_c_high": args.alpha_c_high,
        "prototype_blend": args.prototype_blend,
        "expert_temperature": args.expert_temperature,
        "steered_generation_temperature": args.temperature,
    }
    first = rows[0]
    for key, expected in expected_floats.items():
        if key not in first or not math.isclose(float(first[key]), expected, abs_tol=1e-8):
            raise RuntimeError(
                f"Cannot resume {args.output_csv}: {key} does not match "
                f"the current configuration ({first.get(key)!r} != {expected!r})"
            )
    expected_ints = {
        "route_layer": int(assets["route_layer"]),
        "steer_layer": int(assets["steer_layer"]),
        "expert_neighbors": args.expert_neighbors,
    }
    for key, expected in expected_ints.items():
        if int(first.get(key, -1)) != expected:
            raise RuntimeError(
                f"Cannot resume {args.output_csv}: {key} does not match "
                f"the current configuration ({first.get(key)!r} != {expected!r})"
            )
    if first.get("judge_model") != args.judge_model:
        raise RuntimeError(
            f"Cannot resume {args.output_csv}: judge_model does not match "
            f"({first.get('judge_model')!r} != {args.judge_model!r})"
        )
    expected_reuse = str(bool(args.reuse_base_for_b))
    if first.get("reuse_base_for_b") != expected_reuse:
        raise RuntimeError(
            f"Cannot resume {args.output_csv}: reuse_base_for_b does not match "
            f"({first.get('reuse_base_for_b')!r} != {expected_reuse!r})"
        )


def reusable_result(old, target, alpha, args, route_layer, steer_layer):
    if not old or not old.get("Q7-gen", "").strip() or not old.get("Q7-label", "").strip():
        return False
    if old.get("route_target") != target:
        return False
    expected_floats = {
        "effective_alpha": alpha,
        "prototype_blend": args.prototype_blend,
        "expert_temperature": args.expert_temperature,
        "steered_generation_temperature": args.temperature,
    }
    # Thresholds matter only through the resulting target and effective alpha,
    # both of which are checked above. This permits exact reuse of unaffected
    # rows while sweeping confidence gates.
    # Only a C branch depends on the C strength schedule metadata.
    if target == "C":
        expected_floats.update(
            alpha_c_low=args.alpha_c_low,
            alpha_c_high=args.alpha_c_high,
        )
    for key, expected in expected_floats.items():
        try:
            if not math.isclose(float(old[key]), expected, abs_tol=1e-8):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    expected_ints = {
        "route_layer": route_layer,
        "steer_layer": steer_layer,
        "expert_neighbors": args.expert_neighbors,
    }
    for key, expected in expected_ints.items():
        try:
            if int(float(old[key])) != expected:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return (
        old.get("judge_model") == args.judge_model
        and old.get("reuse_base_for_b") == str(bool(args.reuse_base_for_b))
    )


def effective_alpha(target, confidence, args):
    if target == "A":
        progress = max(
            0.0,
            min(
                1.0,
                (confidence - args.min_confidence_a)
                / (1 - args.min_confidence_a),
            ),
        )
        return args.alpha_a_low + progress * (args.alpha_a_high - args.alpha_a_low)
    if target == "C":
        progress = max(
            0.0,
            min(
                1.0,
                (confidence - args.min_confidence_c)
                / (1 - args.min_confidence_c),
            ),
        )
        return args.alpha_c_low + progress * (
            args.alpha_c_high - args.alpha_c_low
        )
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--base-csv",
        default="outputs/14_free/base/base_freeform_qwen_val_gpt4o_mini.csv",
    )
    parser.add_argument(
        "--assets", default="outputs/14_free/assets/training_free_assets.pt"
    )
    parser.add_argument(
        "--vector-dir", default="outputs/14_free/assets/vectors"
    )
    parser.add_argument("--min-confidence-a", type=float, default=0.45)
    parser.add_argument("--min-confidence-c", type=float, default=0.55)
    parser.add_argument("--alpha-a-low", type=float, default=1.0)
    parser.add_argument("--alpha-a-high", type=float, default=1.5)
    parser.add_argument("--alpha-c-low", type=float, default=3.0)
    parser.add_argument("--alpha-c-high", type=float, default=4.0)
    parser.add_argument("--prototype-blend", type=float, default=0.25)
    parser.add_argument("--expert-neighbors", type=int, default=24)
    parser.add_argument("--expert-temperature", type=float, default=0.05)
    parser.add_argument("--judge-provider", choices=["auto", "local"], default="local")
    parser.add_argument("--judge-base-url", default="http://hl279-cmp-01.egr.duke.edu:4000")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--output-csv", default="outputs/14_free/val/14_free_val.csv"
    )
    parser.add_argument(
        "--reuse-compatible-csv",
        help="Reuse already generated/judged rows only when their effective configuration matches.",
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--generation-only", action="store_true")
    phase.add_argument("--judge-only", action="store_true")
    parser.add_argument(
        "--reuse-base-for-b",
        action="store_true",
        help="Reuse cached no-steering B responses. Do not use for a strict full run.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.prototype_blend <= 1:
        raise ValueError("--prototype-blend must be in [0, 1]")
    if not 0 <= args.min_confidence_a < 1:
        raise ValueError("--min-confidence-a must be in [0, 1)")
    if not 0 <= args.min_confidence_c < 1:
        raise ValueError("--min-confidence-c must be in [0, 1)")
    if args.alpha_a_low > args.alpha_a_high:
        raise ValueError("--alpha-a-low must not exceed --alpha-a-high")
    if args.alpha_c_low > args.alpha_c_high:
        raise ValueError("--alpha-c-low must not exceed --alpha-c-high")
    if args.expert_neighbors <= 0:
        raise ValueError("--expert-neighbors must be positive")
    if args.expert_temperature <= 0:
        raise ValueError("--expert-temperature must be positive")
    common.load_dotenv(ROOT / ".env")
    common.load_dotenv(ROOT / "external/VLM-GeoPrivacyBench/.env")
    system_prompt, user_prompt, extract_granularity = common.load_external_benchmark()
    assets = torch.load(project_path(args.assets), map_location="cpu")
    route_layer = int(assets["route_layer"])
    steer_layer = int(assets["steer_layer"])
    vector_dir = project_path(args.vector_dir)
    global_vectors = {
        "A": load_global_vector(vector_dir, "A_minus_B", steer_layer),
        "C": -load_global_vector(vector_dir, "B_minus_C", steer_layer),
    }
    zero = torch.zeros_like(global_vectors["A"])

    output = project_path(args.output_csv)
    results = read_rows(output) if output.exists() and (args.resume or args.judge_only) else []
    validate_resume(results, args, assets)
    if args.judge_only:
        if not results:
            raise RuntimeError("--judge-only requires an existing output")
        client = common.make_judge_client(args.judge_provider, args.judge_base_url)
        results = common.judge_saved_answers(results, output, client, extract_granularity, args.judge_model)
        common.summarize(results, "Q7-label", "steered_free_case_type")
        return

    rows = read_rows(args.base_csv)
    if args.max_rows:
        rows = rows[: args.max_rows]
    for row in rows:
        if row.get("question") != user_prompt or row.get("judge_model") != args.judge_model:
            raise RuntimeError(f"Base prompt/judge mismatch for {row['full_id']}")
    completed = {row["full_id"] for row in results}
    reusable = (
        {row["full_id"]: row for row in read_rows(args.reuse_compatible_csv)}
        if args.reuse_compatible_csv
        else {}
    )
    judge_client = None if args.generation_only else common.make_judge_client(
        args.judge_provider, args.judge_base_url
    )

    AutoProcessor, QwenModel, process_vision_info = common.load_model_dependencies()
    model = QwenModel.from_pretrained(assets["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(assets["model_name"])
    model.eval()
    state = {"steer_vec": zero}
    handle = model.model.language_model.layers[steer_layer].register_forward_hook(common.make_dynamic_hook(state))
    route_counts = Counter(row.get("route_target", "") for row in results)
    try:
        for index, row in enumerate(rows, 1):
            if row["full_id"] in completed:
                continue
            inputs = common.build_free_form_inputs(
                processor, process_vision_info, common.resolve_image_path(row), system_prompt, user_prompt
            ).to(model.device)
            state["steer_vec"] = zero
            with torch.no_grad():
                prompt_output = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
            hidden = prompt_output.hidden_states[route_layer + 1][0, -1].float().cpu()
            target, feature, probabilities = route(hidden, assets)
            confidence = float(probabilities.max())
            threshold = args.min_confidence_a if target == "A" else args.min_confidence_c if target == "C" else 0.0
            if confidence < threshold:
                target = "B"
            route_counts[target] += 1
            alpha = effective_alpha(target, confidence, args)
            local_cosine = 0.0
            effective_prototype_blend = 0.0
            old = reusable.get(row["full_id"])
            if reusable_result(old, target, alpha, args, route_layer, steer_layer):
                result = dict(old)
                # The dual schedule records low/high rather than a fixed-C field.
                result.pop("alpha_c", None)
                result.update({
                    "alpha_a_low": args.alpha_a_low,
                    "alpha_a_high": args.alpha_a_high,
                    "alpha_c_low": args.alpha_c_low,
                    "alpha_c_high": args.alpha_c_high,
                    "min_confidence_a": args.min_confidence_a,
                    "min_confidence_c": args.min_confidence_c,
                    "reuse_source": "compatible_candidate",
                    "reused_from": str(project_path(args.reuse_compatible_csv)),
                    "assets": str(project_path(args.assets)),
                    "vector_dir": str(vector_dir),
                })
                results.append(result)
                common.write_rows(output, results)
                print(
                    f"[{index}/{len(rows)}] {row['full_id']:20s} true={row['true_label']} "
                    f"target={target} conf={confidence:.2f} alpha={alpha:.2f} "
                    "final=" + result["Q7-label"] + " via=compatible"
                )
                continue
            if target == "B" and args.reuse_base_for_b:
                answer = row["model_answer"]
                label = common.normalize_judge_label(row["pred_label"])
                judge_raw = row.get("judge_raw", label)
                reuse_source = "base"
            else:
                if target in global_vectors:
                    vector, local_cosine, effective_prototype_blend = query_vector(
                        feature, target, assets, global_vectors[target], args
                    )
                    state["steer_vec"] = alpha * vector
                sample_seed = int(row.get("generation_seed") or args.seed + index - 1)
                answer = common.generate_answer(model, processor, inputs, args, sample_seed)
                reuse_source = "generated"
                if args.generation_only:
                    label = judge_raw = ""
                else:
                    label, judge_raw = common.judge_answer(answer, judge_client, extract_granularity, args.judge_model)
            result = dict(row)
            result.update({
                "base_q7_gen": row["model_answer"],
                "base_q7_label": common.normalize_judge_label(row["pred_label"]),
                "route_target": target,
                "route_confidence": confidence,
                "route_prob_a": float(probabilities[0]),
                "route_prob_b": float(probabilities[1]),
                "route_prob_c": float(probabilities[2]),
                "effective_alpha": alpha,
                "local_global_cosine": local_cosine,
                "effective_prototype_blend": effective_prototype_blend,
                "Q7-gen": answer,
                "Q7-label": label,
                "steered_judge_raw": judge_raw,
                "steered_free_case_type": common.make_case(row["true_label"], label) if label else "",
                "reuse_source": reuse_source,
                "route_layer": route_layer,
                "steer_layer": steer_layer,
                "alpha_a_low": args.alpha_a_low,
                "alpha_a_high": args.alpha_a_high,
                "alpha_c_low": args.alpha_c_low,
                "alpha_c_high": args.alpha_c_high,
                "min_confidence_a": args.min_confidence_a,
                "min_confidence_c": args.min_confidence_c,
                "prototype_blend": args.prototype_blend,
                "expert_neighbors": args.expert_neighbors,
                "expert_temperature": args.expert_temperature,
                "steered_generation_temperature": args.temperature,
                "reuse_base_for_b": bool(args.reuse_base_for_b),
                "reused_from": "",
                "assets": str(project_path(args.assets)),
                "vector_dir": str(vector_dir),
                "judge_model": args.judge_model,
            })
            results.append(result)
            common.write_rows(output, results)
            print(
                f"[{index}/{len(rows)}] {row['full_id']:20s} true={row['true_label']} "
                f"target={target} conf={confidence:.2f} alpha={alpha:.2f} "
                f"local_cos={local_cosine:.2f} final={label or 'pending'}"
            )
    finally:
        handle.remove()
    print("Route counts:", dict(route_counts))
    if args.generation_only:
        print("Generation complete; rerun with --judge-only.")
    else:
        common.summarize(results, "Q7-label", "steered_free_case_type")
    print("Saved:", output)


if __name__ == "__main__":
    main()
