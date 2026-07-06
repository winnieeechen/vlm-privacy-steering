#!/usr/bin/env python3
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lm = model.model.language_model

print("Language model class:")
print(type(lm))

print("\nChildren under model.model.language_model:")
for name, module in lm.named_children():
    print(name, "->", type(module))

print("\nTrying common layer paths:")

candidates = [
    "layers",
    "model.layers",
    "decoder.layers",
]

for path in candidates:
    obj = lm
    ok = True
    for part in path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            ok = False
            break

    if ok:
        print(f"FOUND language_model.{path}")
        print("type:", type(obj))
        try:
            print("length:", len(obj))
            print("first layer:", type(obj[0]))
        except Exception as e:
            print("cannot get length:", e)

print("\nNamed modules containing language_model and layers:")
count = 0
for name, module in model.named_modules():
    if "language_model" in name and ("layers" in name or "layer" in name):
        print(name, "->", type(module))
        count += 1
        if count >= 80:
            break

print("\nDone.")
