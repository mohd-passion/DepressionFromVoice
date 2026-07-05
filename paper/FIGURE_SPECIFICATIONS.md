# Figure Specifications

No figures are generated in Stage 3A. These specifications are implementation-ready and journal-neutral.

## Figure 1 - End-to-End Experimental Pipeline

| Field | Specification |
| --- | --- |
| Scientific purpose | Show the shared path from audio to participant-level research prediction while distinguishing DAIC, Android interview, Android reading and paired-task evaluation. |
| Exact source files | `configs/phase2c_android_domain_shift.json`; `configs/phase2d_android_reading.json`; `configs/phase2e_cross_task_transfer.json`; `paper/MANUSCRIPT_DRAFT.md` Methods 3.1-3.9 |
| Required columns | Not applicable for diagram; use protocol names, preprocessing settings, WavLM layer, aggregation and classifier labels. |
| Data transformations | None; schematic only. |
| Chart type | Flow diagram. |
| X-axis | Not applicable. |
| Y-axis | Not applicable. |
| Groups | DAIC development; Android interview; Android reading; paired Android transfer. |
| Confidence intervals | Not applicable. |
| Annotations | Mark 16 kHz, 10-second windows, 5-second hop, frozen WavLM layer 9, temporal mean pooling, participant mean aggregation and logistic regression. |
| Caption | Frozen WavLM layer-9 pipeline used across DAIC-WOZ and Androids Corpus experiments. |
| Interpretation | Shows common representation pipeline and protocol separation. |
| Prohibited interpretation | Do not imply DAIC timing features were used in Android WavLM transfer. |
| Accessibility | Use high-contrast colors, direct labels and shape redundancy for protocol groups. |
| Preferred vector output format | SVG or PDF. |

## Figure 2 - Phase 1-2E Experimental Progression

| Field | Specification |
| --- | --- |
| Scientific purpose | Show how model choices were frozen before later evaluations. |
| Exact source files | `outputs/experiment_registry/EXPERIMENT_LOG.md`; `outputs/experiment_registry/experiment_registry.json`; `paper/PAPER_EVIDENCE_MAP.md` |
| Required columns | Phase, experiment ID, decision, selected/not selected status, evaluation scope. |
| Data transformations | Collapse detailed experiment records into phase-level milestones. |
| Chart type | Timeline or stepped protocol diagram. |
| X-axis | Phase sequence from 1A to 2E. |
| Y-axis | Decision/output annotation lane. |
| Groups | DAIC development; Android preparation; zero-shot evaluation; adaptation diagnosis; reading evaluation; cross-task transfer. |
| Confidence intervals | Not applicable. |
| Annotations | Mark frozen tag `paper-experiment-freeze-v1` and no Android use for WavLM layer selection. |
| Caption | Sequential experiment governance from DAIC-WOZ representation development through Android cross-task transfer. |
| Interpretation | Shows governance and freeze sequence. |
| Prohibited interpretation | Do not imply repeated test-set optimization. |
| Accessibility | Use readable phase labels and avoid color-only encoding. |
| Preferred vector output format | SVG or PDF. |

## Figure 3 - Zero-Shot Versus Adapted Android Interview Performance

| Field | Specification |
| --- | --- |
| Scientific purpose | Visualize frozen DAIC reference, threshold adaptation, score calibration and Android-trained boundary results. |
| Exact source files | `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`; `paper/PAPER_RESULTS_TABLES.md` Table 3 |
| Required columns | Protocol, ROC-AUC, macro F1, Brier, target labels used, classifier refit. |
| Data transformations | Round metrics to four decimals; optionally invert Brier panel interpretation by labeling lower as better. |
| Chart type | Grouped bar plot or faceted bar plot. |
| X-axis | Protocol. |
| Y-axis | Metric value. |
| Groups | Frozen zero-shot; threshold-only adaptation; frozen-score calibration; Android-trained logistic regression. |
| Confidence intervals | Not required unless added from official compact interval files. |
| Annotations | Label Protocols B/C as target-domain adaptation and Protocol D as within-Android. |
| Caption | DAIC-to-Android interview domain-shift diagnosis. |
| Interpretation | Frozen scores retained limited ranking while Android-trained boundaries improved within-corpus discrimination. |
| Prohibited interpretation | Do not label Protocols B/C as pure zero-shot or Protocol D as external validation. |
| Accessibility | Use distinct shapes or direct labels in addition to color. |
| Preferred vector output format | SVG or PDF. |

## Figure 4 - Reading, Interview and Cross-Task ROC-AUC Comparison

| Field | Specification |
| --- | --- |
| Scientific purpose | Compare matched-task, cross-task and pooled-task Android results under participant-coherent folds. |
| Exact source files | `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`; `records/runs/phase2e_20260704_151323/phase2e_transfer_gap_analysis.csv`; `outputs/experiment_registry/experiment_registry.json` |
| Required columns | Protocol, source task, target task, ROC-AUC, AUC CI lower, AUC CI upper, gain interval where relevant. |
| Data transformations | Round ROC-AUC and intervals to four decimals; order protocols as matched, cross-task, pooled. |
| Chart type | Point plot with 95 percent intervals. |
| X-axis | Protocol. |
| Y-axis | ROC-AUC. |
| Groups | Matched reading, matched interview, reading-to-interview, interview-to-reading, pooled-to-reading, pooled-to-interview. |
| Confidence intervals | Use AUC CIs where available and annotate gain intervals separately. |
| Annotations | Mark pooled gains as suggestive because gain intervals include zero. |
| Caption | Participant-coherent Android reading/interview transfer. |
| Interpretation | Cross-task transfer remains strong but below matched-task training; pooled point estimates are highest. |
| Prohibited interpretation | Do not claim pooled training is definitively superior. |
| Accessibility | Use readable labels and avoid overlapping interval bars. |
| Preferred vector output format | SVG or PDF. |

## Figure 5 - Domain-Shift Feature Summary

| Field | Specification |
| --- | --- |
| Scientific purpose | Compare DAIC-to-Android feature shift with Android reading-versus-interview shift. |
| Exact source files | `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`; `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json` |
| Required columns | Comparison, centroid cosine distance, mean absolute standardized shift, shifted feature count, total feature count. |
| Data transformations | Extract summary statistics; round to four decimals; compute shifted feature share only if needed. |
| Chart type | Horizontal bar plot or small multiples. |
| X-axis | Shift statistic value. |
| Y-axis | Comparison. |
| Groups | DAIC-to-Android interview; Android reading versus interview. |
| Confidence intervals | Not available; do not fabricate. |
| Annotations | Label 550/768 and 292/768 shifted features. |
| Caption | Feature-shift summaries for frozen WavLM layer-9 participant embeddings. |
| Interpretation | DAIC-to-Android shift is larger than within-Android task shift by the reported summaries. |
| Prohibited interpretation | Do not claim causality or identify a single shift mechanism. |
| Accessibility | Avoid dense small text; include numeric labels. |
| Preferred vector output format | SVG or PDF. |

## Optional Supplement Figure S1 - Classifier-Boundary Similarity Analysis

| Field | Specification |
| --- | --- |
| Scientific purpose | Show fold-level classifier-boundary similarity metrics. |
| Exact source files | `records/runs/phase2e_20260704_151323/phase2e_classifier_boundary_analysis.csv` |
| Required columns | Fold, coefficient cosine similarity, coefficient sign agreement, top-50 Jaccard overlap, matched-task score correlation, cross-target score correlation. |
| Data transformations | Use fold-level values; report mean in caption if shown. |
| Chart type | Dot plot or box plot across folds. |
| X-axis | Metric. |
| Y-axis | Value. |
| Groups | Boundary similarity and score-correlation metrics. |
| Confidence intervals | Not required. |
| Annotations | Add dashed line only if meaning is clearly defined. |
| Caption | Classifier-boundary similarity across participant-coherent folds. |
| Interpretation | Reading and interview classifiers share some structure but retain task-specific boundaries. |
| Prohibited interpretation | Do not treat coefficients as causal feature importance. |
| Accessibility | Use direct metric labels and clear y-axis scale from 0 to 1 where applicable. |
| Preferred vector output format | SVG or PDF. |

## Optional Supplement Figure S2 - Participant-Coherent Fold Design

| Field | Specification |
| --- | --- |
| Scientific purpose | Explain Phase 2E leakage prevention. |
| Exact source files | `docs/experiments/PHASE2E_CROSS_TASK_TRANSFER_PROTOCOL.md`; `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json` |
| Required columns | Participant ID placeholder, task, fold, train/test role; use schematic placeholders only. |
| Data transformations | Do not expose real participant identifiers. |
| Chart type | Schematic fold diagram. |
| X-axis | Fold assignment. |
| Y-axis | Participant/task rows. |
| Groups | Reading and interview examples from same participant. |
| Confidence intervals | Not applicable. |
| Annotations | Show reading and interview records assigned to same held-out fold. |
| Caption | Participant-coherent fold design for Android reading/interview transfer. |
| Interpretation | Demonstrates leakage prevention. |
| Prohibited interpretation | Do not imply original task-specific folds were already participant-coherent. |
| Accessibility | Use simple shapes and text labels rather than color alone. |
| Preferred vector output format | SVG or PDF. |
