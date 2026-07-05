# Supplementary Material Plan

No supplementary files are generated in Stage 3A. This plan identifies candidate supplementary content.

| Item | Purpose | Source | Main paper or supplement | Privacy risk | Participant-level data excluded? |
| --- | --- | --- | --- | --- | --- |
| Phase 1 layer-wise results | Show all WavLM layer comparisons without crowding main text. | `records/runs/phase1b_20260622_123725/layerwise_nested_cv_summary.csv` | supplement | low | yes |
| Aggregation ablation | Document mean, median and mean-plus-std PCA comparison. | `records/runs/phase1c_20260622_135800/aggregation_nested_cv_summary.csv` | supplement | low | yes |
| Phase 1 timing and fusion ablations | Explain why timing-only was retained for DAIC but not transferred. | `records/runs/phase1d_20260622_142132/timing_fusion_nested_cv_summary.csv`; `records/runs/phase1e_20260622_145240/late_fusion_nested_cv_summary.csv` | supplement | low | yes |
| Selected C values | Support reproducibility of logistic-regression choices. | `phase2c_selected_parameters.csv`; `phase2d_selected_parameters.csv`; `phase2e_selected_parameters_summary.csv` | supplement | low | yes |
| Selected thresholds | Clarify fixed, adapted and fold-specific thresholds. | `selected_threshold.json`; Phase 2 selected-parameter files | supplement | medium | yes |
| Calibration diagnostics | Support distinction between discrimination and probability scale. | `phase2c_protocol_comparison.csv`; `CALIBRATION_METHODS_TABLE.md` | supplement | low | yes |
| Bootstrap intervals | Provide interval detail beyond main tables. | `android_interview_bootstrap_intervals.csv`; Phase 2D/2E interval records | supplement | low | yes |
| Domain-shift feature summaries | Support feature-shift figure and discussion. | `PHASE2C_WITHIN_ANDROID_REPORT.json`; `PHASE2D_ANDROID_READING_REPORT.json` | supplement | medium | yes |
| Classifier-boundary fold metrics | Support optional boundary similarity figure. | `phase2e_classifier_boundary_analysis.csv` | supplement | medium | yes |
| Claim and evidence governance | Document audit trail for manuscript claims. | `PAPER_EVIDENCE_MAP.md`; `FINAL_NUMERICAL_AUDIT.csv`; `FINAL_RESULT_SCOPE_AUDIT.md` | supplement | low | yes |
| Reproducibility workflow | Explain compact records, registry validation and frozen tag. | `outputs/experiment_registry/EXPERIMENT_LOG.md`; `paper/validate_manuscript.py` | supplement | low | yes |

## Exclusions

Raw audio, embeddings, fitted models, participant predictions and participant-level artifacts should remain excluded from supplementary material unless a separate ethics, privacy and dataset-license review authorizes release.
