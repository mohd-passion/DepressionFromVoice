# BSPC Figure Captions

Status: figures_generated_awaiting_author_review

## Figure 1. Study design/protocol overview

Study design and protocol overview for the frozen WavLM layer-9 participant-level research pipeline. The schematic distinguishes DAIC-WOZ development, DAIC-to-Android interview transfer, Android reading evaluation and participant-coherent Android cross-task transfer. Source basis: `paper/FIGURE_SPECIFICATIONS.md`, `paper/BSPC_FIGURE_PRODUCTION_PLAN.md`, `paper/PAPER_RESULTS_TABLES.md` and `paper/PAPER_EVIDENCE_MAP.md`. Prohibited interpretation: do not imply that DAIC timing features were used in Android WavLM transfer or that the workflow is clinically deployable.

## Figure 2. DAIC Phase 1 key results

DAIC Phase 1F provisional local labeled test ROC-AUC for the frozen WavLM free-speech candidate and the guided-interview timing candidate. The WavLM candidate is lower on this DAIC-only local split but was retained for Android transfer because it is compatible with unconstrained speech; the timing candidate depends on guided-interview timing definitions. Source basis: `outputs/experiment_registry/experiment_summary.csv`. Prohibited interpretation: do not treat these as official hidden AVEC test results or as clinical validation.

## Figure 3. DAIC-to-Android transfer and adaptation

Android interview protocol comparison for frozen DAIC zero-shot scores, threshold-only adaptation, frozen-score calibration and Android-trained logistic regression. ROC-AUC and macro F1 are plotted from `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`. Prohibited interpretation: do not label threshold-only adaptation or calibration as pure zero-shot, and do not describe Android-trained logistic regression as external validation.

## Figure 4. Android reading, interview and cross-task results

Android reading/interview task results and participant-coherent Phase 2E cross-task/pooled-task ROC-AUC values. Sources are `records/runs/phase2d_20260704_125931/phase2d_task_comparison.csv`, `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv` and `records/runs/phase2e_20260704_151323/phase2e_transfer_gap_analysis.csv`. Prohibited interpretation: do not claim pooled training is definitively superior from point estimates alone.

## Figure 5. Domain-shift and classifier-boundary summary

Aggregate domain-shift and classifier-boundary summaries. Panel A compares DAIC-to-Android feature shift with Android reading/interview task shift using centroid cosine distance, mean absolute standardized shift, shifted-feature share and available MMD^2. Panel B shows mean Phase 2E classifier-boundary and score-similarity metrics. Sources are `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`, `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json` and `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json`. Android reading/interview MMD^2 was not available in compact records and is not fabricated. Prohibited interpretation: do not treat coefficients as causal feature importance or claim a single mechanism explains domain shift.
