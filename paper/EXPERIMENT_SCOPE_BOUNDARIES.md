# Experiment Scope Boundaries

This file states what each major experiment may and may not support. It is intended to prevent cross-corpus, zero-shot, target-adaptation, within-corpus and clinical-readiness claims from being mixed.

## Scope Categories

- Cross-corpus: training or frozen score source comes from DAIC-WOZ and evaluation is on Androids Corpus.
- Zero-shot: no Android target labels are used for fitting, threshold adaptation or calibration.
- Target adaptation: Android labels are used for thresholds or calibration while the frozen DAIC score generator is retained.
- Within-corpus: training and evaluation occur within Androids Corpus or within DAIC-WOZ folds.
- Cross-task transfer: source and target tasks differ within Androids Corpus under participant-coherent folds.
- Clinical validation: no experiment in this study is clinical validation.

## Boundary Table

| experiment | cross_corpus | zero_shot | target_labels_used | classifier_refit | threshold_adapted | calibration_used | within_corpus | clinical_validation | permitted_claim | prohibited_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1A metadata baseline | no | no | yes, DAIC labels | yes, DAIC | yes, DAIC development threshold | no | yes, DAIC | no | Metadata-only DAIC sanity baseline. | External validation or clinical utility. |
| Phase 1A final-layer WavLM baseline | no | no | yes, DAIC labels | yes, DAIC | yes, DAIC development threshold | no | yes, DAIC | no | Default WavLM baseline before layer analysis. | Best WavLM representation or external validation. |
| Phase 1B WavLM layer-wise analysis | no | no | yes, DAIC training labels | yes, DAIC folds | no external threshold | no | yes, DAIC | no | DAIC method selection for layer 9. | Cross-corpus generalization. |
| Phase 1C aggregation analysis | no | no | yes, DAIC training labels | yes, DAIC folds | no external threshold | no | yes, DAIC | no | DAIC method selection for participant mean aggregation. | External validation. |
| Phase 1D timing analysis | no | no | yes, DAIC training labels | yes, DAIC folds | no external threshold | no | yes, DAIC | no | DAIC timing features carry guided-interview signal. | Android timing transfer. |
| Phase 1E late fusion | no | no | yes, DAIC training labels | yes, DAIC folds | no external threshold | no | yes, DAIC | no | Fusion did not improve over timing-only in DAIC. | Portable multimodal superiority. |
| Phase 1F frozen WavLM candidate | no | no | yes, DAIC labels | frozen after development | frozen DAIC threshold | no | yes, DAIC | no | Provisional local DAIC test result for free-speech-compatible candidate. | Official hidden AVEC result. |
| Phase 1F frozen timing candidate | no | no | yes, DAIC labels | frozen after development | frozen DAIC threshold | no | yes, DAIC | no | Provisional local DAIC test result for guided-interview timing candidate. | Android-compatible timing model. |
| Phase 2A Android audit | no | not_applicable | no modeling | no | no | no | audit only | no | Android inventory and compatibility audit. | Model performance. |
| Phase 2B DAIC-to-Android interview zero-shot | yes | yes | no Android labels | no | no | no | no | no | Limited cross-corpus ranking transfer with all-negative thresholded predictions. | Successful operational classifier. |
| Phase 2C Protocol A frozen DAIC reference | yes | yes | no Android labels | no | no | no | no | no | Frozen DAIC zero-shot reference. | Target-domain adaptation. |
| Phase 2C Protocol B threshold-only adaptation | yes | no | yes, Android train-fold thresholds | no | yes | no | no | no | Target-domain threshold adaptation using frozen scores. | Pure zero-shot evaluation. |
| Phase 2C Protocol C frozen-score calibration | yes | no | yes, Android train-fold calibration | no source refit | after calibration protocol | yes | no | no | Target-domain score calibration and probability-scale diagnosis. | Improved zero-shot ranking. |
| Phase 2C Protocol D Android-trained interview | no | no | yes, Android labels | yes, Android interview | yes within folds | no separate external calibration | yes, Android interview | no | Strong within-Android interview discrimination. | DAIC-to-Android model transfer or external validation. |
| Phase 2C domain-shift analysis | yes | no | aggregate labels/records only | no model claim | no | no | descriptive comparison | no | Feature-distribution and boundary-shift interpretation. | Causal mechanism or clinical explanation. |
| Phase 2D frozen DAIC-to-Android reading | yes | yes | no Android reading labels | no | no | no | no | no | Modest zero-shot reading ranking with failed thresholds. | Successful cross-corpus classifier. |
| Phase 2D Android-trained reading | no | no | yes, Android reading labels | yes, Android reading | yes within folds | no separate external calibration | yes, Android reading | no | Strong within-Android reading discrimination. | External validation. |
| Phase 2D reading/interview comparison | no | no | yes in separate task protocols | yes by task | yes by task | no | yes, Android tasks | no | Descriptive task comparison. | Controlled cross-task transfer. |
| Phase 2E matched reading | no | no | yes, paired Android folds | yes, reading | yes within folds | no | yes, Android reading | no | Participant-coherent matched reading reference. | External validation. |
| Phase 2E matched interview | no | no | yes, paired Android folds | yes, interview | yes within folds | no | yes, Android interview | no | Participant-coherent matched interview reference. | External validation. |
| Phase 2E reading-to-interview | no | no | yes, source-task Android folds | yes, reading source task | source/fold thresholding only | no | yes, cross-task within Android | no | Reading-to-interview transfer within Android. | Cross-corpus generalization. |
| Phase 2E interview-to-reading | no | no | yes, source-task Android folds | yes, interview source task | source/fold thresholding only | no | yes, cross-task within Android | no | Interview-to-reading transfer within Android. | Cross-corpus generalization. |
| Phase 2E pooled-to-reading | no | no | yes, pooled Android folds | yes, pooled reading plus interview | fold thresholding only | no | yes, Android pooled-task | no | Pooled-task reading estimate with cautious gain interpretation. | Definitive pooled superiority. |
| Phase 2E pooled-to-interview | no | no | yes, pooled Android folds | yes, pooled reading plus interview | fold thresholding only | no | yes, Android pooled-task | no | Pooled-task interview estimate with cautious gain interpretation. | Definitive pooled superiority. |
| Phase 2E boundary-analysis summary | no | no | yes, paired Android fold outputs | descriptive analysis | not_applicable | no | yes, Android task comparison | no | Task-sensitive classifier-boundary interpretation. | Causal feature importance. |

## Clinical Boundary

None of these experiments is clinical validation. The study did not perform prospective deployment, clinician-in-the-loop evaluation, demographic fairness analysis, clinical utility testing, diagnostic workflow integration, or post-deployment monitoring. Permitted wording should therefore remain research-focused.
