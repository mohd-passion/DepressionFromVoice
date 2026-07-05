# Paper Reference Needs

This file tracks every `[REFERENCE NEEDED: ...]` placeholder used in `paper/MANUSCRIPT_DRAFT.md`, including resolved placeholders whose audit trail remains useful. Do not replace placeholders with unverified citations.

## Depression Background

- `[REFERENCE NEEDED: depression and speech as behavioral signal]` - resolved in Stage 2B with `leal2025survey`; survey-level support only.
- `[REFERENCE NEEDED: depression burden and prevalence]` - unresolved; still needs an authoritative epidemiology or public-health source.

## Speech Biomarkers

- `[REFERENCE NEEDED: acoustic and prosodic correlates of depression]` - resolved in Stage 2B with `leal2025survey` and `tao2023featureCorrelation`; wording remains hedged because findings are task- and population-dependent.
- `[REFERENCE NEEDED: handcrafted acoustic features in depression detection]` - resolved in Stage 2B with `leal2025survey`, `valstar2016avec` and `tao2023featureCorrelation`.
- `[REFERENCE NEEDED: survey of speech-based depression detection]` - resolved in Stage 2B with `leal2025survey`.
- `[REFERENCE NEEDED: evaluation variability in depression speech studies]` - resolved only for speech-study heterogeneity in Stage 2B with `leal2025survey`; metric-specific citation needs remain open below.
- `[REFERENCE NEEDED: acoustic feature sets for depression speech]` - resolved in Stage 2B with `leal2025survey`, `valstar2016avec` and `tao2023featureCorrelation`.

## DAIC-WOZ

- `[REFERENCE NEEDED: DAIC-WOZ dataset description]` - resolved in Stage 2A with `gratch2014distress`, `valstar2016avec` and `kroenke2009phq8`.

## Androids Corpus

- `[REFERENCE NEEDED: Androids Corpus dataset description]` - resolved in Stage 2A with `tao2023androids`.

## Self-Supervised Speech Models

- `[REFERENCE NEEDED: self-supervised speech representation learning]` - resolved in Stage 2C with `baevski2020wav2vec2`, `chen2022wavlm` and `yang2021superb`; supports reusable SSL speech representations, not depression-specific clinical validity or domain invariance.
- `[REFERENCE NEEDED: wav2vec 2.0 and related SSL speech models]` - resolved in Stage 2C with `baevski2020wav2vec2`, `chen2022wavlm`, `yang2021superb`, `wu2023sslDepression` and `huang2024voicePretraining`; manuscript wording remains limited to feature-extractor/fine-tuning framing.
- `[REFERENCE NEEDED: SSL transfer in paralinguistic or affective speech]` - resolved in Stage 2C with `baevski2020wav2vec2`, `yang2021superb`, `wu2023sslDepression` and `huang2024voicePretraining`; support is methodological and application-level, not proof of target-condition validity.

## WavLM

- `[REFERENCE NEEDED: WavLM paper]` - resolved in Stage 2A with `chen2022wavlm`.

## Cross-Corpus Depression Detection

- `[REFERENCE NEEDED: cross-corpus depression detection]` - resolved in Stage 2C with `alghowinem2016crossCultural`, `seneviratne2021generalized` and `leal2025survey`; `seneviratne2021generalized` is retained as an arXiv preprint.
- `[REFERENCE NEEDED: prior cross-corpus findings in depression speech]` - resolved in Stage 2C with `alghowinem2016crossCultural`, `seneviratne2021generalized` and `leal2025survey`; support is used only for cautious comparison with transfer concerns.

## Calibration and Domain Shift

- `[REFERENCE NEEDED: domain shift and calibration in machine learning]` - resolved in Stage 2C with `morenoTorres2012datasetShift`, `niculescuMizil2005probabilities` and `vanCalster2019calibration`.
- `[REFERENCE NEEDED: domain adaptation and dataset shift taxonomy]` - resolved in Stage 2C with `morenoTorres2012datasetShift`; calibration and probability-score tools are additionally supported by `brier1950probabilityForecasts`, `niculescuMizil2005probabilities`, `guo2017calibration` and `vanCalster2019calibration`.
- `[REFERENCE NEEDED: classifier boundary shift under dataset shift]` - resolved in Stage 2C with `morenoTorres2012datasetShift` and `vanCalster2019calibration`; manuscript uses classifier-boundary shift descriptively for this project's Protocol D versus frozen DAIC comparison.

## Reading Versus Spontaneous Speech

- `[REFERENCE NEEDED: reading versus spontaneous speech in clinical speech analysis]` - resolved in Stage 2B for the limited task-design contrast, using `tao2023androids`, `tao2023featureCorrelation` and `leal2025survey`; it does not establish clinical interchangeability of task types.
- `[REFERENCE NEEDED: reading speech and spontaneous speech comparison]` - resolved in Stage 2B for Androids-centered reading/interview comparison, using `tao2023androids`, `tao2023featureCorrelation` and `leal2025survey`.

## Evaluation Metrics

- Brier score and calibration-versus-discrimination framing are partially resolved in Stage 2C with `brier1950probabilityForecasts`, `niculescuMizil2005probabilities`, `guo2017calibration` and `vanCalster2019calibration`.
- Add references for ROC-AUC, PR-AUC, balanced accuracy, macro F1 and bootstrap confidence intervals if the target venue expects metric citations.

## Clinical Validation and Ethical Limitations

- `[REFERENCE NEEDED: ethical and clinical limits of speech-based mental-health models]` - unresolved.
- Add references for prospective validation, clinical utility, dataset bias and fairness before submission.
