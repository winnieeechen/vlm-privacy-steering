# Dual-Router Test Result

## Fixed setting selected on validation set

- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Layer: 32
- alpha_over: 0.5
- alpha_under: 0.3
- privacy_threshold: -0.09
- utility_threshold: 0.08

## Test results

Base:
- Correct: 48/133 = 0.361
- Over-disclosure: 28/133 = 0.211
- Under-disclosure: 57/133 = 0.429

Over-only steering:
- Correct: 55/133 = 0.414
- Over-disclosure: 17/133 = 0.128
- Under-disclosure: 61/133 = 0.459

Under-only steering:
- Correct: 41/133 = 0.308
- Over-disclosure: 37/133 = 0.278
- Under-disclosure: 55/133 = 0.414

Final dual-router:
- Correct: 57/133 = 0.429
- Over-disclosure: 17/133 = 0.128
- Under-disclosure: 59/133 = 0.444

## Interpretation

The under-disclosure vector alone reduces under-disclosure but substantially increases over-disclosure. This shows that the vector captures a more informative / more disclosing direction, but should not be applied globally.

The final dual-router improves over the over-only steering baseline:
- Correct predictions increase from 55 to 57.
- Under-disclosure decreases from 61 to 59.
- Over-disclosure stays unchanged at 17.

This supports the dual-vector router design: the privacy vector reduces over-disclosure, while the utility vector can partially recover under-disclosure when gated carefully.
