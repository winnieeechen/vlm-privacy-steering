# Status 002: Small-scale conditional steering result

Date: 2026-07-01

## Model

Qwen/Qwen2.5-VL-3B-Instruct

## Dataset

Small downloaded subset from VLM-GeoPrivacyBench.

Total: 40 images

Label distribution:

- A: 20
- B: 10
- C: 10

## Base result

| Metric | Result |
|---|---:|
| Correct | 13/40 = 0.325 |
| Over-disclosure | 9/40 = 0.225 |
| Under-disclosure | 18/40 = 0.450 |

Base case counts:

- A_to_A: 11
- A_to_B: 9
- B_to_A: 8
- B_to_B: 2
- C_to_B: 10

## Vectors

Condition vector:

- positive: true_label A/B
- negative: true_label C
- saved at: outputs/vectors/condition_vectors_qwen_vl_40.pt

Behavior vector:

- positive: A_to_A, B_to_B
- negative: A_to_B, A_to_C, B_to_C
- saved at: outputs/vectors/behavior_vectors_qwen_vl_40.pt

## Selected small-scale setting

- layer: 32
- alpha: 0.5
- threshold: -0.042

Output file:

outputs/conditional_steered_qwen_vl_40_layer32_alpha0.5_thr-0.042.csv

## Conditional steering result

| Metric | Base | Conditional Steering |
|---|---:|---:|
| Correct | 13/40 = 0.325 | 16/40 = 0.400 |
| Over-disclosure | 9/40 = 0.225 | 5/40 = 0.125 |
| Under-disclosure | 18/40 = 0.450 | 19/40 = 0.475 |

Conditional case counts:

- A_to_A: 15
- A_to_B: 5
- B_to_A: 9
- B_to_B: 1
- C_to_B: 9
- C_to_A: 1

Gate counts:

- True: 29
- False: 11

## Interpretation

The over-disclosure behavior vector successfully reduces over-disclosure.

However, the vector also pushes some samples toward more conservative answers, causing a small increase in under-disclosure.

This supports the need for a future under-disclosure / utility vector.
A two-vector system may be better than a single over-disclosure vector.

## Current conclusion

The pipeline is working:

1. Download benchmark images.
2. Run base VLM inference.
3. Build condition and behavior cases.
4. Extract hidden-state vectors.
5. Apply behavior steering with hooks.
6. Use condition vector as a gate.
7. Reduce over-disclosure on a small-scale subset.

Next step: scale up the dataset and rebuild vectors using train/validation/test splits.
