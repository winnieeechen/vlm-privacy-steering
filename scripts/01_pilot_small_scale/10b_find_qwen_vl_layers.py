#!/usr/bin/env python3
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Top-level children:")
for name, module in model.named_children():
    print(name, "->", type(module))

print("\nChildren under model.model:")
for name, module in model.model.named_children():
    print(name, "->", type(module))

print("\nPossible layer/module names containing 'layer' or 'block':")
count = 0
for name, module in model.named_modules():
    lname = name.lower()
    if "layer" in lname or "block" in lname:
        print(name, "->", type(module))
        count += 1
        if count >= 80:
            break

print("\nDone.")
