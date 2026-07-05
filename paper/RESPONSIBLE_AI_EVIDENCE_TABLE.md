# Responsible-AI Evidence Table

This table records responsible-AI issues raised by the Stage 2D literature and the current project evidence status. It is a limitations and governance table, not a results table.

| Issue | Literature support | Current project status | Evidence currently present | Evidence not completed | Manuscript handling |
| --- | --- | --- | --- | --- | --- |
| Mental-health data sensitivity | [@who2021aiHealthEthics; @who2025depression] | partially_addressed | Manuscript treats depression as sensitive health context and avoids clinical-use claims. | Formal ethics review documentation is not included in paper artifacts. | Discussed in Introduction and 5.8. |
| Speech privacy | [@nautsch2019gdprSpeech; @ravi2022speakerDisentanglement] | partially_addressed | Raw audio is excluded from versioned paper artifacts. | No formal speech-data privacy impact assessment. | Discussed in 5.8 and Data and Code Availability. |
| Voice re-identification | [@tomashenko2021voicePrivacy; @nautsch2019gdprSpeech] | partially_addressed | Participant-level artifacts, embeddings and predictions are excluded. | No anonymization or attack-model evaluation. | Stated as a privacy risk and limitation. |
| Informed consent and dataset access | [@nautsch2019gdprSpeech; @who2021aiHealthEthics] | partially_addressed | Dataset redistribution is avoided and original access terms are acknowledged. | Consent terms are not re-audited beyond dataset-use constraints. | Data and Code Availability section. |
| Data minimization | [@nist2023aiRmf; @nautsch2019gdprSpeech] | partially_addressed | Compact registry records omit raw audio and fitted models. | No formal minimization protocol beyond artifact exclusion. | Reproducibility and availability language. |
| Linguistic bias | [@koenecke2020racialDisparitiesASR; @alghowinem2016crossCultural] | not_completed | Corpus/task/language differences are described. | No dialect/accent subgroup analysis. | Listed as fairness gap. |
| Demographic bias | [@nist2023aiRmf; @koenecke2020racialDisparitiesASR] | not_completed | No demographic fairness claims are made. | No sex, age, dialect, accent, site or intersectional subgroup audit. | Explicitly stated in 5.8 and Limitations. |
| Dataset imbalance | [@nist2023aiRmf; @kelly2019clinicalImpactAI] | partially_addressed | Class counts and participant-level splits are documented. | No bias-specific reweighting or subgroup balancing analysis. | Treated as a readiness gap, not a performance claim. |
| Subgroup evaluation | [@liu2020consortAI; @nist2023aiRmf] | not_completed | Participant-level evaluation exists. | No subgroup error-rate reporting. | Identified as future clinical-validation requirement. |
| External validation | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | partially_addressed | DAIC-to-Android zero-shot evaluation was completed. | Android-trained models were not evaluated on an independent third corpus. | Carefully separated from within-Android validation. |
| Prospective clinical validation | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | not_completed | Retrospective corpus experiments only. | No prospective clinical or workflow study. | Explicit limitation in 5.8 and 6. |
| Calibration in target populations | [@vanCalster2019calibration; @kelly2019clinicalImpactAI] | partially_addressed | Retrospective target-domain calibration was tested for Android interview scores. | No clinical threshold or calibration-in-use study. | Discussed as probability-scale limitation. |
| Screening versus diagnosis | [@uspstf2023depressionScreening; @kelly2019clinicalImpactAI] | partially_addressed | Manuscript states models are research-only and not diagnostic tools. | No screening workflow or confirmatory diagnostic pathway was evaluated. | Explicit in 5.8. |
| Human oversight | [@who2021aiHealthEthics; @nist2023aiRmf; @liu2020consortAI] | not_completed | No autonomous-use claim is made. | No human-AI interaction, escalation or override protocol. | Listed as future deployment requirement. |
| Deployment monitoring | [@nist2023aiRmf; @kelly2019clinicalImpactAI] | not_completed | No deployment was performed. | No post-deployment monitoring, drift response or decommissioning plan. | Stated as future governance requirement. |
| Clinical utility and patient outcomes | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | not_completed | Technical metrics are reported with caveats. | No decision-curve, workflow, patient-outcome or cost-benefit analysis. | Explicit limitation. |

## Summary

The project has reproducible retrospective research evidence and several privacy-preserving artifact choices, but it has not completed the fairness, prospective validation, clinical utility, human-oversight or deployment-monitoring work that would be required for responsible clinical use.
