# BSPC Figure Visual Review

Status: `ready_for_author_review`

Review date: 2026-07-06

Scope: Stage 3F review of the Stage 3E generated BSPC figures. This review used the generated PNG review copies, SVG/vector files, `paper/BSPC_FIGURE_CAPTIONS.md`, `paper/BSPC_FIGURE_DATA_AUDIT.csv` and `paper/figures/source/bspc_figure_source_values.csv`.

No figure revisions were required in Stage 3F. All figures remain author-review copies and must still be checked against final BSPC artwork requirements before submission.

## Summary

| Figure | Output files | Purpose | Panels included | Main/supplement decision | Revision required | Final status |
|---|---|---|---|---|---|---|
| Figure 1 | `figure1_study_design_protocol_overview.svg`; `figure1_study_design_protocol_overview.png` | Study design and evaluation-protocol overview. | Schematic pipeline and protocol branches. | main | no | ready_for_author_review |
| Figure 2 | `figure2_daic_phase1_key_results.svg`; `figure2_daic_phase1_key_results.png` | DAIC Phase 1F WavLM versus timing result with local-test caveat. | Single ROC-AUC bar chart. | main | no | ready_for_author_review |
| Figure 3 | `figure3_daic_android_transfer_adaptation.svg`; `figure3_daic_android_transfer_adaptation.png` | DAIC-to-Android zero-shot/adaptation comparison. | Grouped ROC-AUC and macro F1 bars. | main | no | ready_for_author_review |
| Figure 4 | `figure4_android_cross_task_results.svg`; `figure4_android_cross_task_results.png` | Android reading/interview and participant-coherent cross-task ROC-AUC comparison. | Phase 2D task panel and Phase 2E protocol panel. | main | no | ready_for_author_review |
| Figure 5 | `figure5_domain_shift_boundary_summary.svg`; `figure5_domain_shift_boundary_summary.png` | Domain-shift and classifier-boundary summary. | Feature-distribution shift panel and classifier-boundary/score-similarity panel. | main | no | ready_for_author_review |

## Figure 1 Review

- Purpose: communicates the shared frozen WavLM layer-9 pipeline and the separation between DAIC development, DAIC-to-Android transfer, Android reading evaluation and cross-task Android analysis.
- Readability: readable in the PNG review copy; final journal-size check remains required.
- Label clarity: protocol labels are direct and avoid implying clinical deployment.
- Axis clarity: not applicable because the figure is schematic.
- Legend clarity: not applicable; color is supported by direct labels.
- Caption match: matches the caption in `paper/BSPC_FIGURE_CAPTIONS.md`.
- Metric traceability: schematic only; audit row marked `not_applicable`.
- Clinical-overclaim risk: low; scope guard explicitly says research evaluation only.
- Privacy risk: low; no participant identifiers or participant-level data.
- Suitable for main manuscript: yes.
- Suitable for supplement: no; this is a main orientation figure.
- Revision required: no.
- Final status: `ready_for_author_review`.

## Figure 2 Review

- Purpose: shows why the free-speech WavLM candidate was retained despite the guided-interview timing candidate having stronger provisional DAIC local-test ROC-AUC.
- Readability: labels and values are readable in the PNG review copy.
- Label clarity: WavLM and timing candidates are distinct; the local-test caveat is visible.
- Axis clarity: y-axis is ROC-AUC/metric value from 0 to 1.
- Legend clarity: direct bar labels make a separate legend unnecessary.
- Caption match: caption states the provisional local labeled DAIC test caveat.
- Metric traceability: both ROC-AUC values trace to `outputs/experiment_registry/experiment_summary.csv`.
- Clinical-overclaim risk: low; caption prohibits treating values as official hidden AVEC or clinical validation.
- Privacy risk: low; aggregate metrics only.
- Suitable for main manuscript: yes.
- Suitable for supplement: no.
- Revision required: no.
- Final status: `ready_for_author_review`.

## Figure 3 Review

- Purpose: separates pure frozen DAIC zero-shot scoring from target-label-used threshold adaptation, frozen-score calibration and Android-trained classifier-boundary learning.
- Readability: grouped bars, labels and notes are readable in the PNG review copy.
- Label clarity: zero-shot and target-adapted protocols are visually and textually distinct.
- Axis clarity: y-axis reports metric value from 0 to 1; legend defines ROC-AUC and macro F1.
- Legend clarity: clear, with color-separated metrics.
- Caption match: caption states that Protocols B/C are not pure zero-shot and Protocol D is not external validation.
- Metric traceability: all plotted values trace to `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`.
- Clinical-overclaim risk: low; figure notes and caption prevent external-validation overstatement.
- Privacy risk: low; aggregate protocol metrics only.
- Suitable for main manuscript: yes.
- Suitable for supplement: no.
- Revision required: no.
- Final status: `ready_for_author_review`.

## Figure 4 Review

- Purpose: compares Android reading/interview within-task results and participant-coherent cross-task/pooled-task ROC-AUC values.
- Readability: labels and values are readable in the PNG review copy.
- Label clarity: matched, cross-task and pooled protocols are named directly.
- Axis clarity: y-axis reports ROC-AUC from 0 to 1.
- Legend clarity: no legend is required because each protocol is directly labeled.
- Caption match: caption notes that pooled point estimates should not be treated as definitive superiority.
- Metric traceability: values trace to `phase2d_task_comparison.csv`, `phase2e_protocol_comparison.csv` and `phase2e_transfer_gap_analysis.csv`.
- Clinical-overclaim risk: low; within-corpus and cross-task scope is retained.
- Privacy risk: low; aggregate task/protocol metrics only.
- Suitable for main manuscript: yes.
- Suitable for supplement: no.
- Revision required: no.
- Final status: `ready_for_author_review`.

## Figure 5 Review

- Purpose: summarizes DAIC-to-Android versus Android reading/interview feature shift and Phase 2E classifier-boundary similarity.
- Readability: two-panel layout is readable in the PNG review copy; final journal-size check should confirm small labels.
- Label clarity: shifted-feature share, MMD^2 availability and boundary metrics are labelled directly.
- Axis clarity: feature-shift panel uses labelled horizontal bars; classifier-boundary panel uses mean value from 0 to 1.
- Legend clarity: direct labels and metric names make a legend unnecessary.
- Caption match: caption notes that Android reading/interview MMD^2 was not available and was not fabricated.
- Metric traceability: values trace to compact Phase 2C, Phase 2D and Phase 2E reports.
- Clinical-overclaim risk: low; caption prohibits causal feature-importance and single-mechanism claims.
- Privacy risk: low; aggregate shift and fold-summary metrics only.
- Suitable for main manuscript: yes.
- Suitable for supplement: no, unless BSPC figure limits require consolidation.
- Revision required: no.
- Final status: `ready_for_author_review`.

## Cross-Figure Checks

- Generated figures exist as PNG review copies and SVG/vector versions.
- Figure numbering matches `paper/BSPC_MANUSCRIPT_DRAFT.md`.
- Figure captions exist for Figures 1-5.
- `paper/BSPC_FIGURE_DATA_AUDIT.csv` contains 38 rows and traces each plotted numeric value.
- No raw audio, participant identifiers, embeddings, fitted models or participant predictions are referenced or included.
- No confidence intervals were fabricated. Where uncertainty intervals were not available in compact figure sources, figures rely on point estimates and captions retain the interpretation caveats.
- No generated figure presents within-corpus results as external validation.
