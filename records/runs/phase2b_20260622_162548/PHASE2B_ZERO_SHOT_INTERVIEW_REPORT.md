# Phase 2B Zero-Shot Android Interview Report

Run directory: `outputs\phase2_android\phase2b_zero_shot_interview_runs\20260622_162548`

## Scope
Frozen DAIC-WOZ Candidate A was applied once to Androids Corpus interview participant speech. No Android training, scaler fitting, threshold tuning, probability recalibration, WavLM layer selection, or reading-speech processing was performed.

## Participants And Windows
- Eligible interview participants: `116`
- Class counts: `HC=52`, `PT=64`
- Excluded participants: `0`
- Valid 10-second windows: `4249`
- Embedding quality status: `passed`

## Zero-Shot Metrics
- ROC-AUC: `0.641226`
- PR-AUC: `0.680853`
- Balanced accuracy at 0.5: `0.500000`
- Macro F1 at 0.5: `0.309524`
- Balanced accuracy at frozen DAIC threshold `0.555840134621`: `0.500000`
- Macro F1 at frozen DAIC threshold: `0.309524`

## Probability Shift
- Probability range: `0.066834` to `0.393785`
- Probability mean/std: `0.142655` / `0.052565`
- Predicted-positive proportion at frozen threshold: `0.000000`

## Interpretation
- Transfer category: `weak_or_failed_transfer`
- DAIC labels and Android clinical-group labels are related but not equivalent; this is not clinical validation.

## Outputs
- Window, embedding, participant, prediction, bootstrap, probability-shift, duration-correlation, subgroup, and comparison artifacts are in this run directory.
