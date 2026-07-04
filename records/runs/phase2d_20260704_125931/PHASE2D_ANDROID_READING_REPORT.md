# Phase 2D Android Reading-Speech Benchmark

Run directory: `outputs\phase2_android\phase2d_reading_runs\20260704_125931`

## Scope

Android reading speech was processed with the locked WavLM layer-9 representation. No interview audio was reprocessed, no Phase 2C model was rerun, and no WavLM fine-tuning or new classifier family was used.

## Reading Inputs

- Eligible reading participants: `112`
- Class counts: `HC=54`, `PT=58`
- Reading recordings: `112`
- Reading windows: `955`
- Extraction status: `passed`

## Protocol Metrics

| Protocol | ROC-AUC | PR-AUC | Balanced Acc. | Macro F1 | Sens. | Spec. | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_zero_shot_daic_at_0_5 | 0.5795 | 0.5997 | 0.5000 | 0.3253 | 0.0000 | 1.0000 | 0.3351 |
| A_zero_shot_daic_at_frozen_threshold | 0.5795 | 0.5997 | 0.5000 | 0.3253 | 0.0000 | 1.0000 | 0.3351 |
| B_android_lr_at_0_5 | 0.8796 | 0.8929 | 0.8033 | 0.8033 | 0.8103 | 0.7963 | 0.1438 |
| B_android_lr_threshold_adapted | 0.8796 | 0.8929 | 0.7768 | 0.7766 | 0.7759 | 0.7778 | 0.1438 |

## Headline Comparison

- Frozen DAIC zero-shot reading ROC-AUC: `0.5795`
- Android-trained reading ROC-AUC: `0.8796`
- Android-trained interview ROC-AUC from Phase 2C: `0.8675`
- Reading-signal classification: `reading_signal_strong`

## Interpretation

Reading and interview performance were similar enough that useful vocal cues may persist across speech tasks under this representation, though the comparison remains descriptive.

No clinical-validity claim is made.
