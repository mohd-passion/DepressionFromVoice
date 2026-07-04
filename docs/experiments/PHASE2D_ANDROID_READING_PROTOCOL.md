# Phase 2D Android Reading-Speech Protocol

## Research Objective

Phase 2D evaluates whether controlled Androids Corpus reading speech contains
depression-related information under the same frozen WavLM representation used
for the completed Android interview analysis.

Research questions:

- Does frozen WavLM layer 9 contain useful depression-related signal in reading speech?
- Does the frozen DAIC-WOZ classifier transfer zero-shot to Android reading speech?
- Does an Android-trained reading classifier outperform the frozen DAIC classifier?
- Is spontaneous interview speech more informative than controlled reading speech?
- Are task-specific classifier boundaries required?

## Official Input Runs

- Corrected Android preparation run:
  `outputs/phase2_android/phase2a_preparation_runs/20260622_154228`
- Completed Phase 2C interview comparison run:
  `outputs/phase2_android/phase2c_domain_shift_runs/20260703_125614`
- Superseded Phase 2A run excluded:
  `outputs/phase2_android/phase2a_preparation_runs/20260622_154008`

## Locked WavLM Configuration

- Encoder: `microsoft/wavlm-base-plus`
- Model revision: `4c66d4806a428f2e922ccfa1a962776e232d487b`
- Transformer layer: `layer_09`
- Sample rate: 16 kHz
- Window duration: 10 seconds
- Hop duration: 5 seconds
- Overlap: 50%
- Window pooling: temporal arithmetic mean
- Participant aggregation: arithmetic mean
- Embedding dimension: 768
- Encoder state: frozen, evaluation mode, no gradients
- Saved embedding dtype: float32

These settings are fixed from prior train-only DAIC and Android interview
protocols. Phase 2D does not reselect layers, pooling, aggregation, or model
family.

## Dataset Scope

Phase 2D uses Android reading recordings only. Interview audio, interview clips,
timing features, gender, age, education, and condition metadata are not model
features. Canonical participant identity is based on condition plus numeric ID.
Binary labels follow Android clinical grouping: HC maps to 0 and PT maps to 1.

## Official Reading Folds

The experiment uses the official Android reading-task five-fold assignments.
Each eligible reading participant must appear in exactly one reading fold. The
interview fold identity must not be used for reading model selection,
preprocessing, threshold selection, or evaluation.

## Protocol A: Frozen DAIC Zero-Shot Reading

Apply the exact frozen DAIC WavLM classifier to Android reading participant
embeddings:

- DAIC-trained `StandardScaler`
- DAIC-trained `LogisticRegression`
- `C=0.001`
- Thresholds reported: 0.5 and frozen DAIC dev threshold `0.555840134621`

No Android reading labels may be used before probability generation. The scaler,
classifier, probability scale, and thresholds are not refit, calibrated, or
tuned.

## Protocol B: Android-Trained Reading Logistic Regression

Train and evaluate only within Android reading speech:

- Representation: frozen layer-9 mean participant embedding
- Pipeline: `StandardScaler` plus balanced L2 `LogisticRegression`
- Candidate `C`: `0.0001`, `0.001`, `0.01`, `0.1`, `1`, `10`, `100`
- Outer protocol: official five reading folds
- Inner protocol: leave-one-training-fold-out across the four outer-training folds
- Primary inner metric: ROC-AUC
- Threshold: optional fold-specific threshold selected only from outer-training
  out-of-fold probabilities for descriptive reporting

Each participant receives exactly one outer-fold prediction.

## Interview-Versus-Reading Comparison

Compare the Phase 2D reading result with the frozen Phase 2C Android interview
Protocol D result from the official Phase 2C run. Phase 2D must not rerun,
modify, retrain, recalibrate, or reinterpret the Phase 2C interview models.

The task comparison reports participant counts, class counts, window counts,
selected parameter distributions, ROC-AUC, PR-AUC, balanced accuracy, macro F1,
sensitivity, specificity, Brier score, uncertainty intervals, and descriptive
embedding shift. Because participant sets and fold assignments may differ, the
comparison is descriptive unless a valid paired subset analysis is explicitly
reported.

## Leakage Protections

- Participant is the evaluation unit.
- Reading participants may appear in exactly one reading fold.
- Interview folds are not used for reading evaluation.
- Outer-test participants are excluded from scaler fitting, C selection,
  classifier fitting, and threshold selection.
- Reading labels are not used before frozen DAIC zero-shot probabilities are
  generated.
- Every eligible participant receives exactly one outer-fold prediction.
- Participant-level predictions, embeddings, raw audio, and sensitive manifests
  remain in ignored local outputs and are not copied to compact Git records.

## Primary Metrics

- ROC-AUC
- PR-AUC
- Balanced accuracy
- Macro F1
- Positive-class precision
- Sensitivity
- Specificity
- Negative predictive value
- Brier score
- TN, FP, FN, TP
- Stratified participant bootstrap percentile intervals

## Prohibited Changes

Phase 2D must not perform WavLM fine-tuning, layer reselection, alternate
pooling, alternate participant aggregation, PCA, SVM, MLP, LightGBM, attention
pooling, timing-feature modeling, reading/interview fusion, domain-adversarial
learning, pooled reading/interview training, Android interview retraining, app
development, tag creation, or release creation.

## Planned Registry IDs

- `phase2d_android_reading_zero_shot_daic`
- `phase2d_android_reading_logistic_regression`
- `phase2d_android_interview_vs_reading_comparison`

## Stop Condition

Stop after the Phase 2D reading benchmark and interview-versus-reading
comparison are executed, validated, recorded, committed, and pushed. Do not
begin Phase 2E or merge the Phase 2D branch automatically.
