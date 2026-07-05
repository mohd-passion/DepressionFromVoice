# Terminology Style Guide

This guide records preferred journal-neutral terms for the manuscript and package artifacts.

| Concept | Preferred term | Avoid | Rationale |
| --- | --- | --- | --- |
| Corpus name | DAIC-WOZ | DAIC alone at first mention | Full corpus name is clearer and matches primary sources. |
| Corpus name | Androids Corpus | Android dataset; Android corpus | Avoid confusion with Android operating-system data. |
| Labels | PHQ-8 binary labels | depression diagnosis for DAIC | DAIC labels are questionnaire-derived supervisory labels. |
| Android labels | HC/PT group labels or healthy-control versus depression-patient group labels | PHQ-8 labels for Android | Androids Corpus uses clinical group labels, not DAIC PHQ-8 labels. |
| Model type | research classification model | diagnostic system | The project is not clinically validated. |
| Clinical status | research-only classification results | clinically validated; deployment-ready | Required safeguard from Stage 2D. |
| Cross-corpus protocol | DAIC-to-Android zero-shot external evaluation | Android-trained external accuracy | Only frozen DAIC-to-Android evaluation is external and zero-shot. |
| Within-corpus protocol | within-corpus nested cross-validation | external validation | Android-trained results are within Androids Corpus. |
| Target adaptation | target-domain threshold adaptation | pure zero-shot | Android labels are used for threshold selection. |
| Calibration | target-domain score calibration | zero-shot calibration | Calibration uses Android training-fold labels. |
| Cross-task protocol | participant-coherent cross-task transfer within Android | cross-corpus transfer | Phase 2E changes task, not corpus. |
| Pooled protocol | pooled-task within-corpus evaluation | definitive pooled superiority | Gain intervals include zero. |
| Phase 1F test | provisional local labeled DAIC test | official hidden AVEC test | Official hidden-test provenance was not established. |
| Metric | ROC-AUC | AUC without defining first | Use ROC-AUC consistently. |
| Metric | PR-AUC | average precision unless source uses it | Keep metric labels aligned with compact records. |
| Metric | balanced accuracy | UAR in prose | Balanced accuracy is clearer for binary classification. |
| Metric | macro F1 | F1 unless table heading defines it | Avoid ambiguity with positive-class F1. |
| Data availability | raw audio and participant-level artifacts excluded | anonymized data released | Artifact exclusion is not equivalent to anonymization. |
| Governance | privacy and governance assessment | privacy solved | The project has documentation safeguards but no formal governance review. |

## Required Phrases

- `research-only`
- `not diagnostic tools`
- `within-Android`
- `cross-corpus zero-shot external evaluation`
- `target-domain threshold adaptation, not pure zero-shot evaluation`
- `target-domain score calibration rather than zero-shot evaluation`
- `provisional local labeled DAIC test`

## Prohibited Or High-Risk Phrases

- `clinically validated`
- `diagnostic system`
- `diagnoses depression`
- `deployment-ready`
- `generalizes to all populations`
- `bias-free`
- `fair across demographics`
- `external accuracy` for Android-trained models
- `official AVEC hidden test` for Phase 1F
