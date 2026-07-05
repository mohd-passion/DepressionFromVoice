# Stage 2C Claim-Reference Map

This map documents the Stage 2C literature integration for cross-corpus depression detection, dataset shift, calibration, classifier-boundary shift and SSL transfer. It does not resolve burden/prevalence, ethics/privacy, clinical deployment or submission-material references.

| Claim ID | Manuscript location | Claim now supported | Citation keys | Support type | Limitations carried forward |
| --- | --- | --- | --- | --- | --- |
| S2C-SSL-001 | 1. Introduction | Self-supervised speech models provide reusable speech representations trained from large-scale unlabeled audio. | `baevski2020wav2vec2`; `chen2022wavlm`; `yang2021superb` | Primary SSL model and benchmark sources | Does not establish depression-specific validity, domain invariance or clinical readiness. |
| S2C-SSL-002 | 2.3 Self-supervised speech representations | wav2vec 2.0, WavLM and benchmarked SSL encoders can be used as feature extractors or fine-tuned for downstream speech tasks. | `baevski2020wav2vec2`; `chen2022wavlm`; `yang2021superb` | Primary model and benchmark sources | Does not specify the best layer or aggregation for this project. |
| S2C-SSL-003 | 2.3 and 5.7 | Depression-speech studies have explored wav2vec 2.0 or related SSL representations, but target-condition validation remains necessary. | `wu2023sslDepression`; `huang2024voicePretraining`; `baevski2020wav2vec2`; `yang2021superb` | Depression-speech SSL applications plus general SSL context | Supports application context, not DAIC-to-Android transfer success. |
| S2C-XCORP-001 | Abstract and 2.4 | Models trained on one corpus can face shifts in language, task, recording conditions, label source and prevalence when applied to another. | `morenoTorres2012datasetShift`; `alghowinem2016crossCultural`; `leal2025survey` | Dataset-shift taxonomy plus depression-speech transfer context | Does not identify a single causal source of this project's transfer failure. |
| S2C-XCORP-002 | 2.4 and 5.7 | Cross-cultural and cross-database depression-speech work motivates cautious cross-corpus protocol labeling. | `alghowinem2016crossCultural`; `seneviratne2021generalized`; `leal2025survey` | Primary cross-cultural/cross-database sources plus survey context | `seneviratne2021generalized` is an arXiv preprint. |
| S2C-SHIFT-001 | 1. Introduction and 2.6 | Zero-shot evaluation, target-domain threshold adaptation, target-domain calibration and target-domain classifier retraining answer different methodological questions. | `morenoTorres2012datasetShift`; `niculescuMizil2005probabilities`; `vanCalster2019calibration` | General ML dataset-shift and calibration methodology | General ML sources; not depression-speech-specific. |
| S2C-SHIFT-002 | 2.6 | Dataset shift can involve feature-distribution, class-prior and conditional-label changes. | `morenoTorres2012datasetShift` | Dataset-shift taxonomy | Manuscript uses taxonomy to frame, not to prove which shift type uniquely occurred. |
| S2C-CAL-001 | 2.6, 3.6, 3.8, 4.3 and 5.2 | Calibration is distinct from discrimination, and Brier score summarizes probability error. | `brier1950probabilityForecasts`; `niculescuMizil2005probabilities`; `guo2017calibration`; `vanCalster2019calibration` | Calibration and scoring-rule methodology | Does not mean a lower Brier score implies better ROC-AUC. |
| S2C-CAL-002 | 3.6 | Target-domain score calibration is not zero-shot external validation because Android labels are used to fit the probability mapping. | `niculescuMizil2005probabilities`; `vanCalster2019calibration` | Calibration-method support | Applies only to the frozen score mapping; WavLM encoder and source score generator were not refit. |
| S2C-THR-001 | 3.6 and 4.3 | Target-domain threshold adaptation preserves the frozen score ranking and does not alter individual predicted probabilities. | `morenoTorres2012datasetShift`; `vanCalster2019calibration` | Dataset-shift and calibration framing | Threshold adaptation can improve thresholded metrics without being pure zero-shot transfer. |
| S2C-BOUND-001 | 5.3 | Classifier-boundary shift is used descriptively for the observed improvement from a target-domain Android classifier on the same frozen WavLM features. | `morenoTorres2012datasetShift`; `vanCalster2019calibration` | Dataset-shift and calibration framing plus project evidence | The cited literature supports the framing; the boundary-shift result itself comes from this project's Protocol D comparison. |
| S2C-SCOPE-001 | 5.6 and 5.7 | Android-trained interview and reading models demonstrate within-Android discrimination, not DAIC-to-Android model transfer. | `alghowinem2016crossCultural`; `seneviratne2021generalized`; `leal2025survey`; `morenoTorres2012datasetShift` | Cross-corpus context plus dataset-shift framing | Does not make clinical, fairness or deployment claims. |

## Placeholders Resolved In Stage 2C

- `[REFERENCE NEEDED: self-supervised speech representation learning]`
- `[REFERENCE NEEDED: wav2vec 2.0 and related SSL speech models]`
- `[REFERENCE NEEDED: cross-corpus depression detection]`
- `[REFERENCE NEEDED: domain shift and calibration in machine learning]`
- `[REFERENCE NEEDED: domain adaptation and dataset shift taxonomy]`
- `[REFERENCE NEEDED: classifier boundary shift under dataset shift]`
- `[REFERENCE NEEDED: prior cross-corpus findings in depression speech]`
- `[REFERENCE NEEDED: SSL transfer in paralinguistic or affective speech]`

## Still Out Of Scope

- `[REFERENCE NEEDED: depression burden and prevalence]`
- `[REFERENCE NEEDED: ethical and clinical limits of speech-based mental-health models]`
