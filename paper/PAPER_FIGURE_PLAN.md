# Paper Figure Plan

No figures are generated in this drafting task. These are proposed figure concepts only.

## Figure 1 - End-to-End Experimental Pipeline

- Purpose: Show the shared pipeline from audio to participant-level prediction.
- Source data: Protocol documents; `configs/phase2c_android_domain_shift.json`; `configs/phase2d_android_reading.json`; `configs/phase2e_cross_task_transfer.json`.
- Chart type: Flow diagram.
- Axes: Not applicable.
- Groups: DAIC development, Android interview, Android reading, paired Android transfer.
- Caption draft: "Frozen WavLM layer-9 pipeline used across DAIC-WOZ and Androids Corpus experiments. Audio was converted to 16 kHz windows, represented with frozen WavLM layer 9, temporally mean pooled, aggregated by participant mean and evaluated with leakage-controlled logistic-regression protocols."
- Potential interpretation risk: The figure must distinguish DAIC participant-only interview speech from Android reading/full-task processing; it should not imply timing features were used in Android WavLM protocols.

## Figure 2 - Phase 1-2E Experimental Progression

- Purpose: Provide a chronological map of how model decisions were frozen.
- Source data: `outputs/experiment_registry/EXPERIMENT_LOG.md`; `outputs/experiment_registry/experiment_registry.json`.
- Chart type: Timeline or stepped protocol diagram.
- Axes: Phase on horizontal axis; decision/output on vertical annotations.
- Groups: Phase 1 DAIC development, Phase 2 Android preparation, zero-shot evaluation, adaptation diagnosis, reading evaluation, cross-task transfer.
- Caption draft: "Sequential experiment governance from DAIC-WOZ representation development through Android cross-task transfer. Android results were not used to select WavLM layer or DAIC participant aggregation."
- Potential interpretation risk: Avoid showing within-Android trained models as external validation.

## Figure 3 - Zero-Shot Versus Adapted Android Interview Performance

- Purpose: Visualize the gap between frozen DAIC transfer, target-threshold adaptation, target calibration and Android-trained classifier boundaries.
- Source data: `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`.
- Chart type: Grouped bar plot.
- Axes: Protocol on x-axis; ROC-AUC, macro F1 and Brier score on y-axis or separate panels.
- Groups: Protocol A frozen zero-shot, Protocol B threshold-only, Protocol C calibration, Protocol D Android-trained logistic regression.
- Caption draft: "DAIC-to-Android interview domain-shift diagnosis. Frozen scores retained limited ranking information, while Android-trained classifier boundaries produced substantially higher within-corpus discrimination."
- Potential interpretation risk: Protocols B and C use Android labels and must not be labeled zero-shot.

## Figure 4 - Reading, Interview and Cross-Task ROC-AUC Comparison

- Purpose: Compare matched-task, cross-task and pooled-task Android results under participant-coherent folds.
- Source data: `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`; registry paper interpretation intervals.
- Chart type: Point plot with 95% intervals where available.
- Axes: Protocol on x-axis; ROC-AUC on y-axis.
- Groups: Matched reading, matched interview, reading-to-interview, interview-to-reading, pooled-to-reading, pooled-to-interview.
- Caption draft: "Participant-coherent Android reading/interview transfer. Cross-task transfer remained strong but lower than matched-task training; pooled-task training produced the highest point estimates."
- Potential interpretation risk: Pooled-task gains should be described as suggestive because paired intervals included zero.

## Figure 5 - Domain-Shift Feature Summary

- Purpose: Summarize feature-distribution shift across DAIC-to-Android and Android reading-to-interview comparisons.
- Source data: `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`; `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json`.
- Chart type: Bar plot or small multiples.
- Axes: Shift statistic on x-axis; comparison on y-axis.
- Groups: DAIC-to-Android interview, Android reading versus interview.
- Caption draft: "Feature-shift summaries for frozen WavLM layer-9 participant embeddings. DAIC-to-Android shift was larger than the within-Android reading/interview task shift."
- Potential interpretation risk: Shift statistics are descriptive and do not prove causality.

## Figure 6 - Classifier-Boundary Similarity Analysis

- Purpose: Show that reading and interview classifiers share some structure but retain task-specific boundaries.
- Source data: `records/runs/phase2e_20260704_151323/phase2e_classifier_boundary_analysis.csv`.
- Chart type: Box plot or dot plot across outer folds.
- Axes: Metric on x-axis; value on y-axis.
- Groups: Coefficient cosine similarity, sign agreement, top-50 Jaccard overlap, matched-task score correlation, cross-target score correlation.
- Caption draft: "Classifier-boundary similarity across participant-coherent folds. Moderate coefficient cosine similarity and low top-feature overlap indicate partially task-specific classifier boundaries."
- Potential interpretation risk: Boundary similarity is exploratory and should not be overinterpreted as feature importance.

## Figure 7 - Participant-Coherent Fold Design

- Purpose: Explain the Phase 2E leakage prevention design.
- Source data: `docs/experiments/PHASE2E_CROSS_TASK_TRANSFER_PROTOCOL.md`; `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json`.
- Chart type: Schematic fold diagram.
- Axes: Not applicable.
- Groups: Participant A reading/interview examples in same fold; training folds versus held-out fold.
- Caption draft: "Participant-coherent fold design for Android reading/interview transfer. Reading and interview records from the same participant were assigned to the same fold to prevent participant leakage."
- Potential interpretation risk: Do not imply the original Android task-specific folds were participant-coherent before Phase 2E correction.
