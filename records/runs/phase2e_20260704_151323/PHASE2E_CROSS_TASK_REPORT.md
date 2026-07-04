# Phase 2E Cross-Task Transfer Report

Status: `completed_phase2e_cross_task_transfer`
Run directory: `outputs\phase2_android\phase2e_cross_task_runs\20260704_151323`

## Paired Participants

- Eligible paired participants: 110
- Class counts: HC=52, PT=58
- Reading-only excluded: 2
- Interview-only excluded: 6

## Protocol Metrics

| Protocol | Threshold | ROC-AUC | PR-AUC | BA | Macro F1 | Sens | Spec | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| interview_to_reading | threshold_0_5 | 0.8501 | 0.8602 | 0.8176 | 0.8176 | 0.8276 | 0.8077 | 0.1571 |
| interview_to_reading | threshold_adapted | 0.8501 | 0.8602 | 0.8137 | 0.8152 | 0.8966 | 0.7308 | 0.1571 |
| matched_interview_within_task | threshold_0_5 | 0.9065 | 0.9305 | 0.8561 | 0.8545 | 0.8276 | 0.8846 | 0.1214 |
| matched_interview_within_task | threshold_adapted | 0.9065 | 0.9305 | 0.8166 | 0.8172 | 0.8448 | 0.7885 | 0.1214 |
| matched_reading_within_task | threshold_0_5 | 0.9048 | 0.9281 | 0.8273 | 0.8269 | 0.8276 | 0.8269 | 0.1274 |
| matched_reading_within_task | threshold_adapted | 0.9048 | 0.9281 | 0.7964 | 0.7976 | 0.8621 | 0.7308 | 0.1274 |
| pooled_to_interview | threshold_0_5 | 0.9135 | 0.9250 | 0.8561 | 0.8545 | 0.8276 | 0.8846 | 0.1122 |
| pooled_to_interview | threshold_adapted | 0.9135 | 0.9250 | 0.8263 | 0.8266 | 0.8448 | 0.8077 | 0.1122 |
| pooled_to_reading | threshold_0_5 | 0.9058 | 0.9226 | 0.8004 | 0.7997 | 0.7931 | 0.8077 | 0.1232 |
| pooled_to_reading | threshold_adapted | 0.9058 | 0.9226 | 0.7994 | 0.7994 | 0.8103 | 0.7885 | 0.1232 |
| reading_to_interview | threshold_0_5 | 0.8475 | 0.8877 | 0.6867 | 0.6741 | 0.9310 | 0.4423 | 0.2825 |
| reading_to_interview | threshold_adapted | 0.8475 | 0.8877 | 0.6771 | 0.6623 | 0.9310 | 0.4231 | 0.2825 |

## Interpretation

- Category: `pooled_training_advantage`
- Reading-to-interview ROC-AUC gap versus matched interview: -0.0590
- Interview-to-reading ROC-AUC gap versus matched reading: -0.0547
- Pooled-to-interview ROC-AUC gain versus reading-to-interview: 0.0660
- Pooled-to-reading ROC-AUC gain versus interview-to-reading: 0.0557

## Safeguards

- Leakage status: `passed`
- No WavLM extraction, audio preprocessing, WavLM fine-tuning, task fusion, or new model family was performed.
- Participant IDs and participant-level predictions remain only in ignored local outputs.
