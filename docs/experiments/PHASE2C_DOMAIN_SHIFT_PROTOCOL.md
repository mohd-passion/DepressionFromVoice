# Phase 2C Domain-Shift Protocol

Planned registry ID: `P2C_ANDROID_DOMAIN_SHIFT_001`

## Research Question

Why did the frozen DAIC-WOZ WavLM candidate transfer only weakly to Android interview speech, and how much of the observed drop is explained by threshold shift, probability calibration shift, classifier-boundary shift, or representation limitations?

## Hypothesis

The frozen layer-9 WavLM representation contains some transferable depression-related ranking signal across DAIC-WOZ and Android interview speech, but the DAIC-trained classifier threshold and probability scale are mismatched to Android because of language, task, recording, and label-definition differences.

## Official Input Runs

- Phase 1F frozen DAIC final evaluation: `outputs/phase1_daic/phase1f_final_evaluation_runs/20260622_151118`
- Phase 2A corrected Android preparation: `outputs/phase2_android/phase2a_preparation_runs/20260622_154228`
- Phase 2B zero-shot Android interview validation: `outputs/phase2_android/phase2b_zero_shot_interview_runs/20260622_162548`
- Compact registry: `outputs/experiment_registry/experiment_registry.json`

## Frozen Representation

- Model: `microsoft/wavlm-base-plus`
- Model commit: `4c66d4806a428f2e922ccfa1a962776e232d487b`
- Transformer layer: `layer_09`
- Windowing: 10-second participant-only speech windows with 50% overlap
- Window pooling: temporal mean
- Participant aggregation: arithmetic mean across window embeddings
- Frozen DAIC classifier: `StandardScaler + LogisticRegression`
- Frozen DAIC classifier C: `0.001`
- Frozen DAIC threshold: `0.555840134621`

## Android Interview Folds

Use the official Android participant-independent five-fold assignment parsed during Phase 2A. Participant identity remains the grouping unit. Interview speech and reading speech must remain separate; Phase 2C uses Android interview speech only unless a later protocol explicitly authorizes another task.

## Protocol A: Frozen DAIC Zero-Shot Reference

Use the existing Phase 2B frozen DAIC-to-Android interview results as the immutable zero-shot reference. Do not regenerate predictions unless a reproducibility audit requires exact rerun under the same frozen inputs.

## Protocol B: Threshold-Only Adaptation

Evaluate whether Android performance is limited mainly by the DAIC decision threshold. Any threshold adaptation must be performed inside Android training folds only and evaluated on held-out Android folds without using held-out labels for threshold selection.

## Protocol C: Score Calibration

Evaluate calibration of the frozen DAIC scores using leakage-safe Android fold structure. Any calibration model must be trained only on the training portion of each Android fold and applied once to that fold's held-out participants.

## Protocol D: Android-Trained Logistic Regression

Train a within-Android logistic-regression baseline using the same frozen layer-9 participant mean representation. Use only Android training folds for scaler and classifier fitting, and evaluate on held-out Android participants according to the official fold protocol.

## Leakage Safeguards

- Participant is the grouping unit.
- No participant may appear in both training and held-out evaluation data within a fold.
- All preprocessing, scaling, calibration, classifier fitting, threshold selection, and hyperparameter selection must occur inside the training portion of each fold.
- Do not use DAIC test labels or Android held-out fold labels for model selection.
- Do not combine Android reading and interview speech.
- Do not use filename-derived condition, participant ID, fold ID, age, gender, education, or label metadata as acoustic-model input features.
- Do not alter frozen Phase 1 or Phase 2B results.

## Primary Metrics

- ROC-AUC
- PR-AUC
- Balanced accuracy
- Macro F1
- Sensitivity
- Specificity
- TN, FP, FN, TP

Threshold-dependent metrics must clearly identify whether the threshold is frozen from DAIC, selected inside Android training folds, or learned by a calibration procedure.

## Forbidden Changes

- No WavLM fine-tuning.
- No alternate WavLM layer selection.
- No alternate participant aggregation.
- No Android reading-speech processing in this protocol.
- No MLP, SVM, LightGBM, attention pooling, window-level classifier, or multi-layer fusion.
- No fabricated registry entries or result summaries.
- No test-fold leakage.

## Stop Condition

Stop after completing the planned Phase 2C domain-shift analysis and producing the required fold-level reports, compact records, and registry update. Do not proceed to new model families, Android reading speech, WavLM fine-tuning, or application development without a new protocol.
