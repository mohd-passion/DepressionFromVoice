# BSPC Figure Production Plan

Stage 3E generated the five main BSPC figure candidates from compact, non-sensitive aggregate records. Stage 3F visually reviewed the generated PNG copies and SVG files. The figures are not final submitted artwork until the authors review content, captions and journal-format requirements.

Generated files:

- `paper/figures/scripts/generate_bspc_figures.py`
- `paper/figures/README.md`
- `paper/figures/source/bspc_figure_source_values.csv`
- `paper/figures/output/figure_generation_log.txt`
- `paper/BSPC_FIGURE_CAPTIONS.md`
- `paper/BSPC_FIGURE_DATA_AUDIT.csv`
- `paper/BSPC_FIGURE_VISUAL_REVIEW.md`

| Figure | Classification | Required source data | Output files | Caption source | Author review needed | Privacy risk | Status |
|---|---|---|---|---|---|---|---|
| Figure 1: Study design/protocol overview | main | `paper/FIGURE_SPECIFICATIONS.md`; `paper/BSPC_FIGURE_PRODUCTION_PLAN.md`; `paper/PAPER_RESULTS_TABLES.md`; `paper/PAPER_EVIDENCE_MAP.md` | `figure1_study_design_protocol_overview.svg`; `figure1_study_design_protocol_overview.png` | `paper/BSPC_FIGURE_CAPTIONS.md` | yes | low; schematic only | ready_for_author_review |
| Figure 2: DAIC Phase 1 key results | main | `outputs/experiment_registry/experiment_summary.csv`; `paper/PAPER_RESULTS_TABLES.md`; `paper/PAPER_EVIDENCE_MAP.md` | `figure2_daic_phase1_key_results.svg`; `figure2_daic_phase1_key_results.png` | `paper/BSPC_FIGURE_CAPTIONS.md` | yes | low; aggregate metrics only | ready_for_author_review |
| Figure 3: DAIC-to-Android transfer and adaptation | main | `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv` | `figure3_daic_android_transfer_adaptation.svg`; `figure3_daic_android_transfer_adaptation.png` | `paper/BSPC_FIGURE_CAPTIONS.md` | yes | low; aggregate metrics only | ready_for_author_review |
| Figure 4: Android reading, interview and cross-task results | main | `records/runs/phase2d_20260704_125931/phase2d_task_comparison.csv`; `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`; `records/runs/phase2e_20260704_151323/phase2e_transfer_gap_analysis.csv` | `figure4_android_cross_task_results.svg`; `figure4_android_cross_task_results.png` | `paper/BSPC_FIGURE_CAPTIONS.md` | yes | low; aggregate metrics only | ready_for_author_review |
| Figure 5: Domain-shift and classifier-boundary summary | main | `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`; `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json`; `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json` | `figure5_domain_shift_boundary_summary.svg`; `figure5_domain_shift_boundary_summary.png` | `paper/BSPC_FIGURE_CAPTIONS.md` | yes | low; aggregate shift and boundary metrics only | ready_for_author_review |
| Figure S1: Classifier-boundary fold metrics | optional | `records/runs/phase2e_20260704_151323/phase2e_classifier_boundary_analysis.csv` | not generated | not generated | yes | low; fold-level aggregate metrics only | optional_not_generated |
| Figure S2: Participant-coherent fold design | optional | Phase 2E protocol/report schematic only | not generated | not generated | yes | low if schematic placeholders only | optional_not_generated |

## Source And Audit Controls

- Every plotted numeric value is listed in `paper/BSPC_FIGURE_DATA_AUDIT.csv`.
- The source-value manifest is `paper/figures/source/bspc_figure_source_values.csv`.
- Figure outputs are SVG and PNG only. EPS, PDF and final journal production artwork remain deferred.
- No raw audio, participant identifiers, embeddings, fitted models, prediction files or participant-level outputs were read or written.
- Figure 5 includes the official DAIC-to-Android MMD^2 value. Android reading/interview MMD^2 was not available in compact records and was marked explicitly as not available rather than fabricated.

## BSPC Suitability Notes

- Keep the five generated main figures unless the final BSPC formatting check requires consolidation.
- Optional supplementary figures remain possible but were not generated in Stage 3E or Stage 3F.
- Do not use these images as final submitted artwork until author review, journal artwork-format checks and any required resolution/vector-format conversion are complete.
