# BSPC Supplementary Material Draft

Status: `draft_prepared_awaiting_author_review`

Target journal: Biomedical Signal Processing and Control

This supplementary draft references compact, non-sensitive paper artifacts only. It does not include raw participant-level predictions, participant identifiers, restricted dataset files, raw audio paths, embeddings, fitted models or model files.

## Supplementary Overview

The supplement should support reproducibility, result-scope clarity and interpretation safeguards without duplicating the full main manuscript. The final supplement can be shortened or reorganized after BSPC formatting review.

The complete experiment-rationale support package is available in `paper/STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md`, `paper/STUDY_EXPERIMENTS_RATIONALE_TABLE.csv`, `paper/EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md`, `paper/EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md`, `paper/EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md`, `paper/EXPERIMENT_DEPENDENCY_FLOW.md` and `paper/EXPERIMENT_SCOPE_BOUNDARIES.md`. These files are documentation support only and do not add experiments or modify results.

## Supplementary Tables

### Supplementary Table S1. Result-scope and claim matrix

Source: `paper/PAPER_RESULTS_TABLES.md`, Table 6.

Purpose: preserve the distinction between DAIC methodological development, provisional local DAIC testing, zero-shot external evaluation, target-domain adaptation, within-Android validation and cross-task transfer. This table may remain in the main manuscript if page limits allow; otherwise it should move here.

### Supplementary Table S2. Numerical audit and evidence-map traceability

Sources:

- `paper/FINAL_NUMERICAL_AUDIT.csv`
- `paper/PAPER_EVIDENCE_MAP.md`
- `paper/BSPC_FIGURE_DATA_AUDIT.csv`

Purpose: document that reported manuscript and figure values trace to frozen compact records and that metric values were not altered during Stage 3F.

### Supplementary Table S3. Clinical-readiness and claim-scope safeguards

Sources:

- `paper/CLINICAL_READINESS_MATRIX.md`
- `paper/FINAL_CLINICAL_CLAIM_AUDIT.md`
- `paper/CONTRIBUTION_CLAIM_AUDIT.md`

Purpose: keep research-only framing visible and prevent diagnostic, screening, fairness or deployment overclaims.

## Extended Method And Result Details

### Layer-wise and aggregation ablations

Sources:

- `paper/PAPER_RESULTS_TABLES.md`, Table 2.
- `outputs/experiment_registry/experiment_summary.csv`.
- Compact Phase 1 run records under `records/runs/phase1b_20260622_123725/` and `records/runs/phase1c_20260622_135800/`.

Include only aggregate layer and aggregation summaries. Do not include raw embeddings, predictions or fitted models.

### Timing, fusion and DAIC candidate selection

Sources:

- `paper/PAPER_RESULTS_TABLES.md`, Table 2.
- `records/runs/phase1d_20260622_142132/`.
- `records/runs/phase1e_20260622_145240/`.
- `records/runs/phase1f_20260622_151118/`.

Use this section to explain why the timing candidate was not used for Android transfer even though it had stronger provisional local DAIC ROC-AUC.

### Calibration and threshold details

Sources:

- `paper/CALIBRATION_METHODS_TABLE.md`.
- `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`.
- `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`.

Use this section to distinguish zero-shot scoring, threshold-only adaptation and frozen-score calibration. Do not describe target-label-used protocols as pure zero-shot.

### Domain-shift details

Sources:

- `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`.
- `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json`.
- `paper/BSPC_FIGURE_DATA_AUDIT.csv`.

Report only aggregate feature-shift statistics such as centroid cosine distance, mean absolute standardized shift, shifted feature count/share and available MMD^2. Do not include participant-level feature vectors.

### Cross-task and classifier-boundary details

Sources:

- `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`.
- `records/runs/phase2e_20260704_151323/phase2e_transfer_gap_analysis.csv`.
- `records/runs/phase2e_20260704_151323/phase2e_classifier_boundary_analysis.csv`.
- `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json`.

Use fold-level classifier-boundary metrics only if needed. Do not treat logistic-regression coefficients as causal feature importance.

## Optional Supplementary Figures

No supplementary figures were generated in Stage 3F. Optional future figures:

- Figure S1: classifier-boundary fold metrics from `phase2e_classifier_boundary_analysis.csv`.
- Figure S2: participant-coherent fold-design schematic using placeholders only.

If generated later, use only compact aggregate records and schematic placeholders.

## Reproducibility Notes

Source freeze:

- Frozen source tag: `paper-experiment-freeze-v1`.
- Registry validation status: 42 records, validation passed in Stage 3F.
- Figure-generation script: `paper/figures/scripts/generate_bspc_figures.py`.

The source datasets are not redistributed. Users must obtain DAIC-WOZ and Androids Corpus data under the original dataset terms before attempting restricted-data reproduction.

## Excluded Materials

Do not include:

- raw audio;
- participant identifiers;
- participant-level predictions;
- raw or derived embeddings;
- fitted models;
- restricted dataset files;
- private credentials or submission portal exports.
