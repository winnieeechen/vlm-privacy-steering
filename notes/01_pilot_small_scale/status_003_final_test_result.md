# Final Test Result: Conditional Privacy Activation Steering

## Dataset

HF-supported subset of GeoPrivacyBench:
- Total downloaded images: 649
- Train: 385
- Val: 131
- Test: 133

## Model

Qwen/Qwen2.5-VL-3B-Instruct

## Vector construction

Condition vector:
- Positive: true label A/B
- Negative: true label C

Behavior vector:
- Positive: A_to_A, B_to_B
- Negative: A_to_B, A_to_C, B_to_C

## Selected hyperparameters

Chosen on validation set:
- layer = 32
- alpha = 0.5
- threshold = -0.09

## Test results

Base:
- Correct: 48/133 = 0.361
- Over-disclosure: 28/133 = 0.211
- Under-disclosure: 57/133 = 0.429

Unconditional steering:
- Correct: 55/133 = 0.414
- Over-disclosure: 17/133 = 0.128
- Under-disclosure: 61/133 = 0.459

Conditional steering:
- Correct: 56/133 = 0.421
- Over-disclosure: 17/133 = 0.128
- Under-disclosure: 60/133 = 0.451

## Main conclusion

Conditional activation steering improves the model's privacy behavior on the held-out test set.

Compared with the base model:
- Correct predictions increase from 48 to 56.
- Over-disclosure decreases from 28 to 17.
- Under-disclosure only slightly increases from 57 to 60.

This suggests that the learned privacy behavior vector can reduce over-disclosure, and the condition vector can gate the intervention to preserve utility better than unconditional steering.
