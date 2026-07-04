# Phase 2E Android Cross-Task Transfer Protocol

## Research Questions

Phase 2E tests whether depression-related classifier boundaries learned from
one Android speech task transfer to the other task under a participant-safe
evaluation design.

Questions:

- Can a classifier trained only on controlled reading speech classify
  spontaneous interview speech?
- Can a classifier trained only on interview speech classify controlled reading
  speech?
- Are reading and interview classifier boundaries interchangeable?
- Does pooled reading-plus-interview training improve task-specific evaluation?
- Is strong within-task performance driven by task-specific decision
  boundaries?
- Does high reading-interview embedding similarity translate into classifier
  transfer?

## Official Input Runs

- Corrected Android preparation run:
  `outputs/phase2_android/phase2a_preparation_runs/20260622_154228`
- Official Phase 2C Android interview run:
  `outputs/phase2_android/phase2c_domain_shift_runs/20260703_125614`
- Official Phase 2D Android reading run:
  `outputs/phase2_android/phase2d_reading_runs/20260704_125931`
- Superseded Android preparation run excluded:
  `outputs/phase2_android/phase2a_preparation_runs/20260622_154008`

## Participant Eligibility

Eligible participants must have exactly one valid reading participant embedding,
exactly one valid interview participant embedding, one consistent binary
diagnosis label, valid task metadata, finite nonzero 768-dimensional vectors for
both tasks, and no duplicate canonical identity. Canonical identity is
condition plus numeric participant ID.

Participant IDs, participant-level predictions, and sensitive manifests must
remain in ignored local outputs and must not be copied into compact Git
records.

## Participant-Coherent Fold Construction

The original reading and interview fold assignments differ for many
participants, so Phase 2E creates a new deterministic participant-coherent
five-fold split over the paired participant set. The same fold assignment is
used for both tasks and all Phase 2E protocols.

Fold construction requirements:

- Grouping unit: participant.
- Stratification target: binary diagnosis label.
- Fixed random seed: `20260704`.
- Approximately balanced participant and class counts.
- Each participant appears in exactly one fold.
- The fold manifest is created once before modeling and is not altered after
  seeing results.

## Locked Representation

- Model: `microsoft/wavlm-base-plus`
- Revision: `4c66d4806a428f2e922ccfa1a962776e232d487b`
- Transformer layer: `layer_09`
- Window duration: 10 seconds
- Window overlap: 50 percent
- Window temporal pooling: arithmetic mean
- Participant aggregation: arithmetic mean
- Embedding dimension: 768
- Embedding dtype: float32

Phase 2E reuses existing participant embeddings only. It must not rerun WavLM
extraction, regenerate windows, reselect layers, change pooling, or alter the
participant aggregation method.

## Locked Classifier Family

All trainable Phase 2E protocols use:

- `StandardScaler`
- L2 `LogisticRegression`
- `class_weight="balanced"`
- `max_iter=10000`
- Fixed random seed
- Candidate `C`: `0.0001`, `0.001`, `0.01`, `0.1`, `1`, `10`, `100`
- Primary inner-selection metric: ROC-AUC

Classification metrics are reported at threshold `0.5` and at an optional
fold-specific threshold selected only from outer-training out-of-fold
probabilities.

## Protocol A: Reading-To-Interview Transfer

For each participant-coherent outer fold, train on reading embeddings from the
outer-training participants and evaluate once on interview embeddings from the
outer-test participants. Interview embeddings from training participants are not
used for model fitting in this protocol.

## Protocol B: Interview-To-Reading Transfer

For each participant-coherent outer fold, train on interview embeddings from the
outer-training participants and evaluate once on reading embeddings from the
outer-test participants. Reading embeddings from training participants are not
used for model fitting in this protocol.

## Protocol C: Pooled-Task Training

For each participant-coherent outer fold, include both the reading and
interview embeddings for every outer-training participant as separate training
observations with the same participant label. Outer-test participants are
excluded from training in both tasks. Evaluation is reported separately on
outer-test interview embeddings and outer-test reading embeddings.

## Matched Within-Task References

To support fold-matched interpretation, Phase 2E also evaluates:

- reading-to-reading on the participant-coherent folds;
- interview-to-interview on the participant-coherent folds.

These are Phase 2E matched references, not replacements for the official Phase
2C and Phase 2D results.

## Leakage Safeguards

- Participant is the grouping unit for all folds.
- All samples from an outer-test participant are excluded from outer training.
- All samples from an inner-validation participant are excluded from inner
  training.
- Scalers, C selection, threshold selection, and model fitting use only the
  allowed training participants for that fold.
- Reading-to-interview training uses reading embeddings only.
- Interview-to-reading training uses interview embeddings only.
- Pooled training uses both tasks only for training participants.
- Every paired participant receives exactly one out-of-fold prediction per
  evaluated protocol and target task.
- Filename-derived condition, participant ID, fold ID, age, gender, education,
  audio, embeddings, fitted models, and participant-level predictions are not
  committed as compact artifacts.

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
- Paired bootstrap differences for the planned protocol comparisons

## Planned Registry IDs

- `phase2e_matched_reading_within_task`
- `phase2e_matched_interview_within_task`
- `phase2e_reading_to_interview_transfer`
- `phase2e_interview_to_reading_transfer`
- `phase2e_pooled_to_reading`
- `phase2e_pooled_to_interview`
- `phase2e_cross_task_transfer_conclusion`

## Prohibited Changes

Phase 2E must not perform new WavLM extraction, audio preprocessing, new window
generation, WavLM fine-tuning, layer selection, alternate pooling, alternate
aggregation, SVM, LightGBM, neural networks, attention pooling, timing features,
feature concatenation across tasks, participant-level reading/interview fusion,
domain-adversarial learning, DAIC retraining, app development, tag creation, or
release creation.

## Stop Condition

Stop after Phase 2E is executed, validated, recorded, committed, and pushed.
Do not merge the branch, create a release, or begin Phase 2F.
