# Phase 2A Androids Corpus Preparation Audit

Run directory: `outputs\phase2_android\phase2a_preparation_runs\20260622_154228`

## Inventory
- Valid participant total: `118`
- Healthy controls: `54`
- Depression patients: `64`
- Reading recordings: `112`
- Full interview recordings: `116`
- Interview clips: `874`

## Labels
DAIC-WOZ uses PHQ8_Binary from questionnaire thresholding. Android uses clinical group status: HC control versus PT depression patient. These are related but not identical targets.

## Fold And Leakage Audit
- Fold integrity status: `failed`
- Participant is the required grouping unit; any participant crossing folds is listed in `android_fold_quality_report.csv`.
- Reading, interview, and clips must remain grouped by canonical participant ID.

## Timing Compatibility
- Exactly compatible Phase 1 timing features: `17` of `28`
- Frozen Candidate B direct transfer valid: `False`
Android timing contains participant turn/clip timestamps but no interviewer rows, so response-latency features cannot be reproduced directly.

## WavLM Compatibility
- Candidate A preprocessing can be reproduced: `True`
- Major caveats: Italian language, task shift, corpus/recording-condition shift, and reading versus interview separation.

## Proposed Next Protocol
Protocol A: zero-shot DAIC WavLM to Android interview after participant-only speech reconstruction validation. Protocol B should keep reading speech separate as a task-shift robustness experiment. Protocol C timing transfer is not valid without feature redefinition.

No modeling, prediction, threshold tuning, WavLM embedding extraction, or Phase 1 model modification was performed.
