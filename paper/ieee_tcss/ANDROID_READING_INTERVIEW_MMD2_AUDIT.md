# Android Reading/Interview MMD2 Audit

Stage: Android reading/interview MMD2 analysis for IEEE TCSS

Branch: `paper/android-reading-interview-mmd2-v1`

## Source Records

| Record | Path |
| --- | --- |
| Reading participant manifest | `outputs/phase2_android/phase2d_reading_runs/20260704_125931/android_reading_participant_embedding_manifest.csv` |
| Interview participant manifest | `outputs/phase2_android/phase2b_zero_shot_interview_runs/20260622_162548/android_interview_participant_embedding_manifest.csv` |
| Phase 2D task-shift report | `outputs/phase2_android/phase2d_reading_runs/20260704_125931/phase2d_task_shift_report.json` |
| Phase 2C DAIC-to-Android domain-shift report | `outputs/phase2_android/phase2c_domain_shift_runs/20260703_125614/daic_android_domain_shift_report.json` |

## Recovered DAIC-to-Android MMD2 Method

| Method Detail | Value |
| --- | --- |
| Existing DAIC-to-Android MMD2 | 0.609127676214 |
| Kernel | rbf |
| Estimator | biased |
| Bandwidth rule | median_nonzero_squared_pairwise_distance_on_combined_samples |
| Scaling for MMD2 | none_for_mmd2_raw_participant_embeddings |
| Feature level | participant_level_arithmetic_mean_wavlm_layer_09_embeddings |
| Embedding dimension | 768 |

## Computed Android Reading/Interview MMD2

| Result | Reading n | Interview n | Dimension | MMD2 | Sigma2 | Gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary paired common participants | 110 | 110 | 768 | 0.286414548756 | 1.35144609691 | 0.369974060484 |
| Sensitivity all available participants | 112 | 116 | 768 | 0.285303706284 | 1.35388086412 | 0.369308713381 |

## Comparability Assessment

The primary paired-participant Android reading/interview MMD2 is directly comparable in estimator, kernel, bandwidth rule, feature dimensionality, participant-level aggregation and raw embedding scale to the DAIC-to-Android MMD2 value. It is not a like-for-like corpus-transfer result because the DAIC-to-Android value compares DAIC train participants with Android interview participants, whereas this value compares reading and interview task distributions inside Android for participants with both task recordings.

Primary value for paper use: `0.2864`.

## Limitations

- The value is a new aggregate analysis computed from existing feature records, not a new classifier result.
- It does not change any DAIC-to-Android MMD2 value, classifier metric, label, dataset count or table value.
- Participant-level embeddings are read locally, but the outputs written here are aggregate audit records only.
- The paired-participant estimate is preferred for Figure 5 because existing Android task-shift metrics were computed on common participants.

## Scope Controls

- No audio preprocessing was run.
- No WavLM extraction was run.
- No classifier training or evaluation was run.
- No experiment registry record was modified.
- No references, author details, clinical claims, deployment claims or speaker-disentanglement claims were changed.
