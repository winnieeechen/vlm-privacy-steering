#!/usr/bin/env python3
"""Extract train activations and build all assets required by method 14."""

import argparse
from collections import Counter
from pathlib import Path

import torch

import method14_common as common


LABELS = common.LABELS
ROOT = common.ROOT


def normalize_rows(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)


def free_form_messages(image_path, system_prompt, user_prompt):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def layer_vectors(
    model, processor, process_vision_info, image_path,
    system_prompt, user_prompt, model_answer=None, answer_max_tokens=None,
):
    messages = free_form_messages(image_path, system_prompt, user_prompt)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    def encode(text):
        return processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

    if model_answer is None:
        inputs = encode(prompt_text)
        answer_start = answer_end = None
    else:
        answer_start = encode(prompt_text)["input_ids"].shape[1]
        inputs = encode(prompt_text + model_answer)
        answer_end = inputs["input_ids"].shape[1]
        if answer_end <= answer_start:
            raise RuntimeError(f"Empty answer token span for {image_path}")
        if answer_max_tokens is not None:
            answer_end = min(answer_end, answer_start + answer_max_tokens)

    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model(
            **inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1
        )
    hidden_states = output.hidden_states[1:]
    if model_answer is None:
        return torch.stack([
            hidden[0, -1].detach().float().cpu() for hidden in hidden_states
        ])
    return torch.stack([
        hidden[0, answer_start:answer_end].mean(0).detach().float().cpu()
        for hidden in hidden_states
    ])


def validate_cache(payload, rows, kind, system_prompt, user_prompt, answer_max_tokens):
    expected = {
        "full_ids": [row["full_id"] for row in rows],
        "kind": kind,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "answer_max_tokens": answer_max_tokens if kind == "behavior" else None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Cache mismatch for {key}. Re-run with --refresh-cache."
            )


def load_cache(path, rows, kind, system_prompt, user_prompt, answer_max_tokens):
    payload = torch.load(path, map_location="cpu")
    validate_cache(
        payload, rows, kind, system_prompt, user_prompt, answer_max_tokens
    )
    print(f"Loaded {kind} cache: {path}")
    return payload


def save_cache(
    path, model_name, rows, labels, activations, kind,
    system_prompt, user_prompt, answer_max_tokens,
):
    payload = {
        "model_name": model_name,
        "full_ids": [row["full_id"] for row in rows],
        "labels": labels,
        "label_column": "pred_label" if kind == "behavior" else "true_label",
        "kind": kind,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "answer_max_tokens": answer_max_tokens if kind == "behavior" else None,
        "token_span": (
            "mean over teacher-forced natural-language answer tokens"
            if kind == "behavior" else "last image-plus-prompt token"
        ),
        "activations": activations,
        "shape": tuple(activations.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved {kind} cache: {path} {tuple(activations.shape)}")
    return payload


def extract_missing_caches(args, condition_rows, behavior_rows, system_prompt, user_prompt):
    condition_path = common.project_path(args.condition_cache)
    behavior_path = common.project_path(args.behavior_cache)
    condition = None
    behavior = None
    if condition_path.exists() and not args.refresh_cache:
        condition = load_cache(
            condition_path, condition_rows, "condition", system_prompt,
            user_prompt, args.answer_max_tokens,
        )
    if behavior_path.exists() and not args.refresh_cache:
        behavior = load_cache(
            behavior_path, behavior_rows, "behavior", system_prompt,
            user_prompt, args.answer_max_tokens,
        )
    if condition is not None and behavior is not None:
        return condition, behavior

    AutoProcessor, QwenModel, process_vision_info = common.load_model_dependencies()
    model = QwenModel.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    if condition is None:
        activations = []
        for index, row in enumerate(condition_rows, 1):
            activations.append(layer_vectors(
                model, processor, process_vision_info,
                common.resolve_image_path(row), system_prompt, user_prompt,
            ))
            print(f"[condition {index}/{len(condition_rows)}] {row['full_id']}")
        condition = save_cache(
            condition_path, args.model_name, condition_rows,
            [row["true_label"] for row in condition_rows],
            torch.stack(activations).float(), "condition",
            system_prompt, user_prompt, args.answer_max_tokens,
        )

    if behavior is None:
        activations = []
        for index, row in enumerate(behavior_rows, 1):
            activations.append(layer_vectors(
                model, processor, process_vision_info,
                common.resolve_image_path(row), system_prompt, user_prompt,
                model_answer=row["model_answer"],
                answer_max_tokens=args.answer_max_tokens,
            ))
            print(f"[behavior {index}/{len(behavior_rows)}] {row['full_id']}")
        behavior = save_cache(
            behavior_path, args.model_name, behavior_rows,
            [row["pred_label"] for row in behavior_rows],
            torch.stack(activations).float(), "behavior",
            system_prompt, user_prompt, args.answer_max_tokens,
        )
    return condition, behavior


def save_behavior_vectors(behavior, output_dir):
    activations = behavior["activations"].float()
    labels = behavior["labels"]
    means = {
        label: activations[[i for i, value in enumerate(labels) if value == label]].mean(0)
        for label in LABELS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for positive, negative in (("A", "B"), ("B", "C")):
        pair = f"{positive}_minus_{negative}"
        vector = means[positive] - means[negative]
        payload = {
            "method": "method14_answer_mean_difference",
            "model_name": behavior["model_name"],
            "group": pair,
            "definition": (
                f"mean teacher-forced answer state for {positive} minus {negative}"
            ),
            "method_vectors": {"mean_diff": vector},
            "layer_norms": vector.norm(dim=1),
            "label_counts": dict(Counter(labels)),
        }
        path = output_dir / f"behavior_vectors_{pair}.pt"
        torch.save(payload, path)
        print(f"Saved behavior vector: {path}")


def build_assets(args, condition, behavior):
    route_raw = condition["activations"][:, args.route_layer].float()
    route_center = route_raw.mean(0)
    route_features = normalize_rows(route_raw - route_center)
    route_labels = torch.tensor([LABELS.index(label) for label in condition["labels"]])
    condition_index = {
        full_id: index for index, full_id in enumerate(condition["full_ids"])
    }

    memory_condition = []
    memory_behavior = []
    memory_labels = []
    memory_ids = []
    for index, (full_id, label) in enumerate(
        zip(behavior["full_ids"], behavior["labels"])
    ):
        if full_id not in condition_index or label not in LABELS:
            continue
        memory_condition.append(route_features[condition_index[full_id]])
        memory_behavior.append(behavior["activations"][index, args.steer_layer].float())
        memory_labels.append(LABELS.index(label))
        memory_ids.append(full_id)

    payload = {
        "format_version": 2,
        "method": "method14_dual_adaptive_attentive_activation_steering",
        "model_name": condition["model_name"],
        "condition_cache": str(common.project_path(args.condition_cache)),
        "behavior_cache": str(common.project_path(args.behavior_cache)),
        "route_layer": args.route_layer,
        "steer_layer": args.steer_layer,
        "route_neighbors": args.route_neighbors,
        "route_center": route_center,
        "route_features": route_features,
        "route_labels": route_labels,
        "memory_condition": torch.stack(memory_condition),
        "memory_behavior": torch.stack(memory_behavior),
        "memory_labels": torch.tensor(memory_labels),
        "memory_ids": memory_ids,
        "labels": LABELS,
    }
    output = common.project_path(args.assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    counts = Counter(LABELS[index] for index in memory_labels)
    print(
        f"Saved assets: {output}; route=L{args.route_layer}/k{args.route_neighbors}, "
        f"steer=L{args.steer_layer}, memory={dict(counts)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default="outputs/14_free/base/base_freeform_qwen_train_gpt4o_mini.csv",
    )
    parser.add_argument("--model-name", default=common.MODEL_NAME)
    parser.add_argument(
        "--condition-cache",
        default="outputs/14_free/assets/cache/condition_prompt_last.pt",
    )
    parser.add_argument(
        "--behavior-cache",
        default="outputs/14_free/assets/cache/behavior_answer_mean.pt",
    )
    parser.add_argument(
        "--vector-dir", default="outputs/14_free/assets/vectors"
    )
    parser.add_argument(
        "--assets", default="outputs/14_free/assets/training_free_assets.pt"
    )
    parser.add_argument("--route-layer", type=int, default=29)
    parser.add_argument("--steer-layer", type=int, default=28)
    parser.add_argument("--route-neighbors", type=int, default=11)
    parser.add_argument("--answer-max-tokens", type=int)
    parser.add_argument("--require-judge-model", default="gpt-4o-mini")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.route_neighbors <= 0:
        raise ValueError("--route-neighbors must be positive")
    if args.route_layer < 0 or args.steer_layer < 0:
        raise ValueError("--route-layer and --steer-layer must be non-negative")
    common.load_dotenv(ROOT / ".env")
    common.load_dotenv(ROOT / "external/VLM-GeoPrivacyBench/.env")
    system_prompt, user_prompt, _ = common.load_external_benchmark()
    rows = common.read_rows(args.train_csv)
    if not rows:
        raise RuntimeError("Training CSV is empty")
    prompt_mismatches = [
        row["full_id"] for row in rows if row.get("question") != user_prompt
    ]
    if prompt_mismatches:
        raise RuntimeError(
            "Training CSV does not use the benchmark free-form prompt; "
            f"first mismatch: {prompt_mismatches[0]}"
        )
    if args.require_judge_model:
        judges = {row.get("judge_model", "") for row in rows}
        if judges != {args.require_judge_model}:
            raise RuntimeError(
                f"Expected judge_model={args.require_judge_model!r}, found {sorted(judges)}"
            )
    condition_rows = [row for row in rows if row.get("true_label") in LABELS]
    behavior_rows = [row for row in rows if row.get("pred_label") in LABELS]
    empty_answers = [
        row["full_id"] for row in behavior_rows if not row.get("model_answer", "").strip()
    ]
    if empty_answers:
        raise RuntimeError(f"Empty train response for {empty_answers[0]}")
    print(
        f"Train rows: condition={len(condition_rows)}, behavior={len(behavior_rows)}, "
        f"behavior labels={dict(Counter(row['pred_label'] for row in behavior_rows))}"
    )
    condition, behavior = extract_missing_caches(
        args, condition_rows, behavior_rows, system_prompt, user_prompt
    )
    save_behavior_vectors(behavior, common.project_path(args.vector_dir))
    build_assets(args, condition, behavior)


if __name__ == "__main__":
    main()
