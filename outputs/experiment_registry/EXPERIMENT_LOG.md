# DepressionFromVoice Experiment Log

_Generated automatically from `experiment_registry.json` by `update_registry.py`._

## 1. Current Research Status

- Phase 1 status: `completed_baseline_ready_for_external_validation`
- Phase 2 status: `phase2b_zero_shot_external_validation_completed`
- Current candidates are research candidates only; no universal best model is declared.
- Accuracy-hunting warning: Do not pursue desired accuracy by repeatedly evaluating the same test set. Use training and validation data for model development; freeze decisions before test evaluation; record every inspected test result; document decisions made after viewing test performance; judge improvements using cross-validation stability, confidence intervals and external validation.

## 2. Dataset Summary

| Dataset | Participants | Labels | Language | Notes |
|---|---:|---|---|---|
| DAIC-WOZ Phase 1 clean | 187 | PHQ8_Binary | English | 17,058 participant-only 10s windows; provisional local labeled test |
| Androids Corpus | 118 | clinical patient/control grouping | Italian | 112 reading, 116 interview, 874 clips; cross-task fold inconsistency affecting 85 participants |

## 3. Chronological Experiment Table

| Phase | Method | Train AUC | Dev AUC | Test/External AUC | Decision | Key Finding |
|---|---|---:|---:|---:|---|---|
| Phase 1A | Metadata-only sanity baseline | 0.551 | 0.573 | 0.444 | not selected | Metadata sanity baseline underperformed and duration was nearly identical to window count. |
| Phase 1A | Final-layer WavLM participant baseline | 0.512 | 0.569 | 0.519 | not selected | Final WavLM layer was weak on dev and provisional local test. |
| Phase 1B | WavLM layer_01 layer-wise analysis | 0.593 |  |  | not selected | layer_01 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_02 layer-wise analysis | 0.572 |  |  | not selected | layer_02 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_03 layer-wise analysis | 0.492 |  |  | not selected | layer_03 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_04 layer-wise analysis | 0.468 |  |  | not selected | layer_04 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_05 layer-wise analysis | 0.520 |  |  | not selected | layer_05 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_06 layer-wise analysis | 0.513 |  |  | not selected | layer_06 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_07 layer-wise analysis | 0.558 |  |  | not selected | layer_07 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_08 layer-wise analysis | 0.573 |  |  | not selected | layer_08 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_09 layer-wise analysis | 0.596 | 0.644 |  | selected | Intermediate WavLM layer_09 outperformed layer_12 in train-only nested CV and improved dev ROC-AUC. |
| Phase 1B | WavLM layer_10 layer-wise analysis | 0.551 |  |  | not selected | layer_10 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_11 layer-wise analysis | 0.542 |  |  | not selected | layer_11 was evaluated in train-only nested CV. |
| Phase 1B | WavLM layer_12 layer-wise analysis | 0.499 |  |  | not selected | layer_12 was evaluated in train-only nested CV. |
| Phase 1C | Layer-9 mean aggregation | 0.596 | 0.644 |  | selected | Aggregation was not the main bottleneck; mean remained selected. |
| Phase 1C | Layer-9 mean_std_pca aggregation | 0.568 |  |  | not selected | mean_std_pca aggregation did not improve over mean. |
| Phase 1C | Layer-9 median aggregation | 0.591 |  |  | not selected | median aggregation did not improve over mean. |
| Phase 1D | Timing fusion model: early_fusion | 0.575 |  |  | not selected | Early feature fusion underperformed. |
| Phase 1D | Timing fusion model: timing_only | 0.613 | 0.625 |  | selected | Compact speech-timing features added useful signal on DAIC guided interview. |
| Phase 1D | Timing fusion model: wavlm_only | 0.596 |  |  | not selected | WavLM-only reproduced Phase 1C result within tolerance. |
| Phase 1E | Late-fusion protocol: logistic_stacking | 0.541 |  |  | not selected | logistic_stacking evaluated under leakage-safe Phase 1E protocol. |
| Phase 1E | Late-fusion protocol: timing_only | 0.610 | 0.625 |  | selected | Fusion did not improve reliability; timing-only retained. |
| Phase 1E | Late-fusion protocol: wavlm_only | 0.596 |  |  | not selected | wavlm_only evaluated under leakage-safe Phase 1E protocol. |
| Phase 1E | Late-fusion protocol: weighted_late_fusion | 0.604 |  |  | not selected | weighted_late_fusion evaluated under leakage-safe Phase 1E protocol. |
| Phase 1F | Frozen DAIC guided-interview timing candidate |  |  | 0.677 | selected | Guided-interview timing candidate performed better on provisional local test, but depends on standardized interaction timing. |
| Phase 1F | Frozen DAIC WavLM free-speech candidate |  |  | 0.506 | selected | Free-speech-compatible candidate closed for external validation, but provisional local-test AUC was near chance. |
| Phase 2A | Androids Corpus preparation and compatibility audit |  |  |  | audit | Android interview WavLM preprocessing is reproducible; timing model direct transfer is invalid because response-latency features are unavailable. |
| Phase 2B | Frozen DAIC-to-Android zero-shot WavLM interview |  |  | 0.641 | not selected | Ranking transferred modestly, but probability scale and frozen threshold did not: all Android participants were predicted negative. |
| Phase 2C | Protocol A frozen DAIC zero-shot reference |  |  | 0.641 | not selected | classifier_boundary_shift |
| Phase 2C | Protocol B threshold-only adaptation |  |  | 0.641 | not selected | classifier_boundary_shift |
| Phase 2C | Protocol C frozen-score calibration |  |  | 0.569 | not selected | classifier_boundary_shift |
| Phase 2C | Protocol D Android-trained logistic regression |  |  | 0.867 | not selected | classifier_boundary_shift |
| Phase 2D | Android interview-versus-reading comparison |  |  | 0.880 | not selected | Reading and interview performance were similar enough that useful vocal cues may persist across speech tasks under this representation, though the comparison remains descriptive. |
| Phase 2D | Android-trained reading logistic regression |  |  | 0.880 | not selected | Android-trained reading model achieved 0.8796 ROC-AUC under official reading folds. |
| Phase 2D | Frozen DAIC zero-shot Android reading |  |  | 0.580 | not selected | Frozen DAIC reading transfer retained modest ranking but threshold transfer failed. |

## 4. Selected And Rejected Approaches

### Selected / Frozen Research Candidates

| Experiment | Why It Was Kept | Limitation |
|---|---|---|
| WavLM layer_09 layer-wise analysis | Highest mean outer ROC-AUC under train-only nested CV. |  |
| Layer-9 mean aggregation | Mean aggregation had the highest train nested-CV ROC-AUC and remained simplest. |  |
| Timing fusion model: timing_only | Highest train nested-CV ROC-AUC among timing/WavLM/fusion. | Requires guided-interview timing structure. |
| Late-fusion protocol: timing_only | Best train nested-CV ROC-AUC; fusion failed practical improvement rule. | Guided-interview-dependent. |
| Frozen DAIC guided-interview timing candidate | Frozen Phase 1 candidate. | test provenance is provisional local labeled test |
| Frozen DAIC WavLM free-speech candidate | Frozen Phase 1 candidate. | test provenance is provisional local labeled test |

### Rejected Or Not Advanced

| Experiment | Reason |
|---|---|
| Metadata-only sanity baseline | Weak provisional test ROC-AUC and metadata-only sanity role. |
| Final-layer WavLM participant baseline | Layer-wise Phase 1B found layer_09 more promising. |
| WavLM layer_01 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_02 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_03 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_04 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_05 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_06 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_07 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_08 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_10 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_11 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| WavLM layer_12 layer-wise analysis | Not selected by train-only layer-selection criterion. |
| Layer-9 mean_std_pca aggregation | Lower or practically negligible train nested-CV performance versus mean. |
| Layer-9 median aggregation | Lower or practically negligible train nested-CV performance versus mean. |
| Timing fusion model: early_fusion | Train ROC-AUC below timing-only and WavLM-only. |
| Timing fusion model: wavlm_only | Timing-only was stronger under Phase 1D selection. |
| Late-fusion protocol: logistic_stacking | Fusion did not improve over timing-only. |
| Late-fusion protocol: wavlm_only | Timing-only was selected. |
| Late-fusion protocol: weighted_late_fusion | Fusion did not improve over timing-only. |

## 5. Current Conclusions

- WavLM layer_09 is the frozen free-speech DAIC representation.
- Mean participant aggregation was retained; aggregation variants did not appear to be the main bottleneck.
- DAIC timing features are useful inside guided-interview DAIC conditions but do not directly transfer to Android because required response-latency features are unavailable.
- Phase 2B showed Android ranking signal for frozen DAIC WavLM, but the DAIC probability scale and threshold failed externally.

## 6. Important Limitations

- DAIC test labels are provisional local labeled test labels, not confirmed official AVEC test-label provenance.
- DAIC PHQ8_Binary and Android clinical patient/control status are related but not equivalent.
- Android cross-task folds are inconsistent for 85 participants; keep task protocols explicit.
- Repeated inspection of the same test/external data must not drive model development.

## 7. Next Planned Experiment

| Planned phase | Protocols | Purpose |
|---|---|---|
| Phase 2C | frozen DAIC zero-shot reference; threshold-only adaptation; probability calibration; Android-trained logistic regression using fixed layer-9 embeddings | distinguish calibration shift; distinguish classifier-boundary shift; distinguish representation limitation |
