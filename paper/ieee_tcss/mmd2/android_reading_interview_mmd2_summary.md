# Android Reading/Interview MMD2 Summary

Computed at: `2026-07-30T11:50:01.184844+00:00`

## Primary Result

| Item | Value |
| --- | --- |
| Comparison | Android reading versus Android interview |
| Sample selection | Paired participants with both tasks |
| Reading participants | 110 |
| Interview participants | 110 |
| Embedding dimension | 768 |
| Kernel | rbf |
| Estimator | biased |
| Scaling for MMD2 | none_for_mmd2_raw_participant_embeddings |
| Sigma squared | 1.35144609691 |
| Gamma | 0.369974060484 |
| MMD2 | 0.286414548756 |
| Centroid cosine distance | 0.044216632843 |

The primary MMD2 value is methodologically compatible with the DAIC-to-Android MMD2 computation because it uses the same participant-level frozen WavLM layer-9 embeddings, the same raw embedding scale, the same biased RBF MMD2 estimator and the same median squared-distance bandwidth rule. It is not the same experimental comparison: DAIC-to-Android compares corpora, whereas this estimate compares paired Android reading and interview tasks.

## Sensitivity Result

| Item | Value |
| --- | --- |
| Sample selection | All available reading and interview participants |
| Reading participants | 112 |
| Interview participants | 116 |
| MMD2 | 0.285303706284 |

The all-available estimate is provided as a sensitivity check. The paired-participant estimate is preferred for Figure 5 because the existing Android reading/interview task-shift metrics in Phase 2D were computed over common participants with both task recordings.
