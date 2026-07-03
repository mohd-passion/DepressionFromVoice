# Phase 2C DAIC-to-Android Domain-Shift Diagnosis

Run directory: `outputs\phase2_android\phase2c_domain_shift_runs\20260703_125614`

## Integrity

- Participants: 116 ({'HC': 52, 'PT': 64})
- Fold leakage status: passed
- WavLM rerun performed: False
- Reading speech processed: False

## Protocol Metrics

| Protocol | ROC-AUC | PR-AUC | Balanced Acc. | Macro F1 | Sens. | Spec. | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_frozen_daic_at_0_5 | 0.6412 | 0.6809 | 0.5000 | 0.3095 | 0.0000 | 1.0000 | 0.4056 | 0.500000 |
| A_frozen_daic_at_frozen_threshold | 0.6412 | 0.6809 | 0.5000 | 0.3095 | 0.0000 | 1.0000 | 0.4056 | 0.555840 |
| B_threshold_only | 0.6412 | 0.6809 | 0.5733 | 0.5688 | 0.5312 | 0.6154 | 0.4056 | fold-specific |
| C_frozen_score_calibration | 0.5694 | 0.6001 | 0.5883 | 0.5857 | 0.7344 | 0.4423 | 0.2474 | 0.500000 |
| D_android_lr_at_0_5 | 0.8675 | 0.8892 | 0.8137 | 0.8098 | 0.7812 | 0.8462 | 0.1695 | 0.500000 |
| D_android_lr_threshold_adapted | 0.8675 | 0.8892 | 0.8275 | 0.8263 | 0.8281 | 0.8269 | 0.1695 | fold-specific |

## Domain-Shift Finding

Dominant failure source: `classifier_boundary_shift`.

Phase 2C separates the failure modes by keeping the frozen DAIC scores fixed for threshold and calibration adaptation, then comparing them with a within-Android logistic-regression boundary on the same frozen layer-9 participant embeddings. The conclusion is descriptive research evidence only, not a clinical claim.

No clinical-validity claim is made.
