# Final Table Audit

| Table number | Purpose | Source data | Numerical validation | Caption status | Required footnotes | Submission readiness |
| --- | --- | --- | --- | --- | --- | --- |
| Table 1 | Dataset and task summary | `paper/PAPER_EVIDENCE_MAP.md`; Phase 1A, Phase 2A, Phase 2B, Phase 2D and Phase 2E compact records | Counts match evidence map. | Title is clear. | Note that DAIC test labels are provisional local labels; Android labels are HC/PT group labels. | ready_for_author_review |
| Table 2 | DAIC development results | Phase 1A-1F compact records and evidence map | Values match evidence map for manuscript-selected train/provisional metrics. | Title is clear. | Note that Phase 1B/1C rows use train-only nested-CV selection metrics; Phase 1F is not official AVEC hidden-test performance. | ready_for_author_review |
| Table 3 | Cross-corpus Android interview results | `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv` and Phase 2B records | Values match evidence map and official reports. | Title is clear. | State that Protocols B/C use Android labels and are not pure zero-shot. | ready_for_author_review |
| Table 4 | Android reading results | `records/runs/phase2d_20260704_125931/phase2d_protocol_comparison.csv`; `phase2d_task_comparison.csv` | Values match evidence map. | Title is clear. | State that Android-trained reading is within-corpus nested CV, not external validation. | ready_for_author_review |
| Table 5 | Participant-coherent cross-task results | `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`; `phase2e_transfer_gap_analysis.csv`; registry intervals | Values match evidence map. | Title is clear. | State that all rows are within Androids Corpus and pooled gains have intervals including zero. | ready_for_author_review |
| Table 6 | Result-scope and claim matrix | `docs/PAPER_RESULTS_INTERPRETATION_MATRIX.md`; `paper/FINAL_RESULT_SCOPE_AUDIT.md` | Scope flags match Stage 3A audit. | Title is clear. | State that validation scope is methodological, not clinical. | ready_for_author_review |

## Table-Wide Style Notes

- Decimal precision is consistently four decimals for primary performance metrics.
- Counts should remain integer values with comma formatting only in prose/table rendering.
- Table captions should define HC, PT, ROC-AUC, PR-AUC and Brier score if the journal requires stand-alone tables.
