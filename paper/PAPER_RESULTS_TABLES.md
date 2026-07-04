# Paper Results Tables

All values are drawn from frozen registry entries and compact official records. Primary metrics are rounded to four decimal places.

## Table 1 - Dataset and Task Summary

| Corpus | Task | Participants | Class counts | Recordings/windows | Language | Evaluation role |
|---|---:|---:|---|---|---|---|
| DAIC-WOZ Phase 1 clean | Interview participant speech | 187 | Train/dev/local test labels from PHQ8_Binary | 17,058 clean 10-second windows | English | Representation development and provisional local testing |
| Androids Corpus | Interview | 116 | HC=52, PT=64 | 116 full recordings; 4,249 valid windows in Phase 2B/2C | Italian | Zero-shot external evaluation and within-Android interview benchmark |
| Androids Corpus | Reading | 112 | HC=54, PT=58 | 112 recordings; 955 valid windows | Italian | Reading zero-shot and within-Android reading benchmark |
| Androids Corpus | Paired reading/interview | 110 | HC=52, PT=58 | Paired participant-level reading and interview embeddings | Italian | Participant-coherent cross-task and pooled-task analysis |

## Table 2 - DAIC Development Results

| Phase | Result | Scope | Primary metric | ROC-AUC | Notes |
|---|---|---|---|---:|---|
| 1A | Metadata-only sanity baseline | Provisional local labeled test | Test ROC-AUC | 0.4437 | Duration/window count nearly redundant; sanity baseline only |
| 1A | Final-layer WavLM baseline | Provisional local labeled test | Test ROC-AUC | 0.5195 | Baseline before layer-wise analysis |
| 1B | Layer 9 WavLM | Train-only nested CV | Mean outer ROC-AUC | 0.5956 | Selected layer using DAIC training data only |
| 1B | Layer 12 WavLM | Train-only nested CV | Mean outer ROC-AUC | 0.4988 | Final transformer layer comparison |
| 1C | Layer-9 mean aggregation | Train-only nested CV | Mean outer ROC-AUC | 0.5956 | Retained participant mean aggregation |
| 1C | Layer-9 median aggregation | Train-only nested CV | Mean outer ROC-AUC | 0.5905 | Not selected |
| 1C | Layer-9 mean plus std with PCA | Train-only nested CV | Mean outer ROC-AUC | 0.5683 | Not selected |
| 1D | Timing-only | Train-only nested CV | Mean outer ROC-AUC | 0.6126 | Stronger in DAIC guided-interview setting |
| 1D | WavLM-only | Train-only nested CV | Mean outer ROC-AUC | 0.5956 | Locked free-speech representation |
| 1D | Early fusion | Train-only nested CV | Mean outer ROC-AUC | 0.5747 | Did not improve |
| 1E | Weighted late fusion | Train-only nested CV | Mean outer ROC-AUC | 0.6044 | Did not exceed timing-only |
| 1E | Logistic stacking | Train-only nested CV | Mean outer ROC-AUC | 0.5413 | Not selected |
| 1F | Frozen WavLM Candidate A | Provisional local labeled test | Test ROC-AUC | 0.5065 | Free-speech-compatible candidate |
| 1F | Frozen timing Candidate B | Provisional local labeled test | Test ROC-AUC | 0.6775 | Guided-interview timing candidate; not Android-compatible |

## Table 3 - Cross-Corpus Android Interview Results

| Protocol | Training data | Target labels used? | Classifier refit? | ROC-AUC | PR-AUC | Brier | Balanced accuracy | Macro F1 | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Frozen DAIC zero-shot at 0.5 | DAIC-WOZ | No | No | 0.6412 | 0.6809 | 0.4056 | 0.5000 | 0.3095 | Limited ranking transfer; all-negative predictions |
| Frozen DAIC zero-shot at frozen threshold | DAIC-WOZ | No | No | 0.6412 | 0.6809 | 0.4056 | 0.5000 | 0.3095 | Same all-negative decision outcome |
| Threshold-only adaptation | DAIC-WOZ scores plus Android train-fold thresholds | Yes, thresholds only | No | 0.6412 | 0.6809 | 0.4056 | 0.5733 | 0.5688 | Target-domain threshold adaptation, not zero-shot |
| Frozen-score calibration | DAIC-WOZ scores plus Android train-fold calibration | Yes, calibration only | No source classifier refit | 0.5694 | 0.6001 | 0.2474 | 0.5883 | 0.5857 | Improved Brier score but not ranking |
| Android-trained interview LR | Android interview folds | Yes | Yes | 0.8675 | 0.8892 | 0.1695 | 0.8275 | 0.8263 | Strong within-Android discrimination |

## Table 4 - Android Reading Results

| Protocol | Training data | Participants | Windows | ROC-AUC | 95% CI | PR-AUC | Brier | Balanced accuracy | Macro F1 | Interpretation |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Frozen DAIC zero-shot reading | DAIC-WOZ | 112 | 955 | 0.5795 | Not reported | 0.5997 | 0.3351 | 0.5000 | 0.3253 | All participants predicted negative |
| Android-trained reading LR | Android reading folds | 112 | 955 | 0.8796 | [0.8113, 0.9377] | 0.8929 | 0.1438 | 0.7768 | 0.7766 | Within-Android reading benchmark |

## Table 5 - Participant-Coherent Cross-Task Results

| Protocol | Source task | Target task | Participants | ROC-AUC | 95% CI / comparison interval | PR-AUC | Brier | Balanced accuracy at 0.5 | Macro F1 at 0.5 | Interpretation |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| Matched reading | Reading | Reading | 110 | 0.9048 | [0.8435, 0.9562] | 0.9281 | 0.1274 | 0.8273 | 0.8269 | Within-task reference |
| Matched interview | Interview | Interview | 110 | 0.9065 | [0.8452, 0.9589] | 0.9305 | 0.1214 | 0.8561 | 0.8545 | Within-task reference |
| Reading-to-interview | Reading | Interview | 110 | 0.8475 | AUC CI [0.7699, 0.9158]; gap [-0.1316, 0.0116] | 0.8877 | 0.2825 | 0.6867 | 0.6741 | Strong but below matched interview |
| Interview-to-reading | Interview | Reading | 110 | 0.8501 | AUC CI [0.7699, 0.9204]; gap [-0.1436, 0.0325] | 0.8602 | 0.1571 | 0.8176 | 0.8176 | Strong but below matched reading |
| Pooled-to-reading | Reading plus interview | Reading | 110 | 0.9058 | AUC CI [0.8442, 0.9539]; gain [-0.0172, 0.1340] | 0.9226 | 0.1232 | 0.8004 | 0.7997 | Suggestive gain; CI includes zero |
| Pooled-to-interview | Reading plus interview | Interview | 110 | 0.9135 | AUC CI [0.8515, 0.9635]; gain [-0.0013, 0.1346] | 0.9250 | 0.1122 | 0.8561 | 0.8545 | Suggestive gain; CI includes zero |

## Table 6 - Result-Scope and Claim Matrix

| Result | Validation scope | Target-domain training? | External validation? | Allowed interpretation |
|---|---|---:|---:|---|
| DAIC layer and aggregation development | Methodological ablation | No external target domain | No | Selects layer 9 and mean aggregation for later frozen representation use |
| DAIC Phase 1F WavLM Candidate A | Provisional local labeled DAIC test | No Android training | No | DAIC local labeled research estimate only |
| DAIC Phase 1F timing Candidate B | Provisional local labeled DAIC test | No Android training | No | Timing model works only under compatible guided-interview timing definitions |
| Phase 2B frozen DAIC-to-Android interview | Cross-corpus zero-shot external evaluation | No | Yes | Limited cross-corpus ranking transfer; thresholded classifier failed |
| Phase 2C threshold-only adaptation | Cross-corpus target adaptation | Threshold only | Yes, adapted | Threshold shift contributed to failure; not pure zero-shot |
| Phase 2C score calibration | Cross-corpus target adaptation | Calibration only | Yes, adapted | Calibration improved probability scale but not ranking |
| Phase 2C Android-trained interview | Within-corpus cross-validation | Yes | No | Strong within-Android interview discrimination with frozen WavLM features |
| Phase 2D Android-trained reading | Within-corpus cross-validation | Yes | No | Strong within-Android reading discrimination with frozen WavLM features |
| Phase 2E cross-task transfer | Cross-task transfer within Android | Yes, source task | No | Reading and interview share useful representation information, with task-specific boundaries |
| Phase 2E pooled-task training | Pooled-task within Android | Yes | No | Highest point estimates; pooled gains suggestive because intervals included zero |
