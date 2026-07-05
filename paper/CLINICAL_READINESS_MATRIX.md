# Clinical Readiness Matrix

This matrix summarizes what has and has not been demonstrated. Status values are restricted to `completed`, `partially_completed`, `not_completed` and `not_applicable`.

Project is not clinically deployment-ready.

| Criterion | Status | Current evidence | Literature basis | Remaining requirement |
| --- | --- | --- | --- | --- |
| Retrospective technical validation | completed | Participant-level DAIC, Android interview, Android reading and Android cross-task experiments are documented in the evidence map. | [@kelly2019clinicalImpactAI] | Maintain registry and evidence-map traceability. |
| Participant-level evaluation | completed | Splits, cross-validation and paired-task folds are participant-coherent. | [@valstar2016avec; @tao2023androids] | Keep leakage safeguards in future extensions. |
| Cross-corpus zero-shot evaluation | completed | Frozen DAIC-to-Android interview and reading transfer were evaluated without Android labels for source-model fitting. | [@alghowinem2016crossCultural; @seneviratne2021generalized; @morenoTorres2012datasetShift] | Do not describe thresholded failures as clinical success. |
| Target-domain calibration | partially_completed | Android interview score calibration was evaluated retrospectively. | [@vanCalster2019calibration; @niculescuMizil2005probabilities] | Validate calibration in the intended clinical setting and population. |
| Independent external validation of Android-trained models | not_completed | Android-trained reading and interview models were only evaluated within Android folds. | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | Evaluate Android-trained models on an independent third corpus or prospective target sample. |
| Subgroup fairness analysis | not_completed | No subgroup error rates or demographic fairness audit are reported. | [@nist2023aiRmf; @koenecke2020racialDisparitiesASR; @liu2020consortAI] | Evaluate performance and calibration across relevant demographic, linguistic and recording subgroups. |
| Prospective clinical validation | not_completed | All experiments are retrospective corpus analyses. | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | Run a prospective study with predefined endpoints, safeguards and monitoring. |
| Comparison with clinician assessment | not_completed | Labels come from dataset protocols and group status, not a new clinician comparison in this study. | [@uspstf2023depressionScreening; @kelly2019clinicalImpactAI] | Compare model outputs with clinician assessment and confirmatory diagnostic workflow. |
| Screening workflow evaluation | not_completed | No screening pathway, referral pathway or follow-up process was evaluated. | [@uspstf2023depressionScreening; @liu2020consortAI] | Define intended use, target population, positive follow-up and escalation protocol. |
| Clinical utility evaluation | not_completed | ROC-AUC, PR-AUC, balanced accuracy, macro F1 and Brier score were reported as technical metrics. | [@kelly2019clinicalImpactAI; @vasey2022decideAI] | Assess patient outcomes, decision impact, harms and net benefit. |
| Privacy and governance assessment | partially_completed | Restricted artifacts are excluded and data availability limits are documented. | [@who2021aiHealthEthics; @nist2023aiRmf; @nautsch2019gdprSpeech] | Complete formal privacy, consent, retention, access-control and update-governance review. |
| Voice anonymization or re-identification evaluation | not_completed | No attack-model or anonymization evaluation was performed. | [@tomashenko2021voicePrivacy; @nautsch2019gdprSpeech] | Evaluate re-identification risk and privacy/utility trade-offs if audio-derived artifacts are shared. |
| Human oversight and escalation | not_completed | No deployment or human-AI interaction workflow is proposed. | [@who2021aiHealthEthics; @nist2023aiRmf; @liu2020consortAI] | Define human review, override, escalation and accountability procedures. |
| Deployment monitoring and drift management | not_completed | No deployed system exists. | [@nist2023aiRmf; @kelly2019clinicalImpactAI] | Define post-deployment monitoring, drift response, incident handling and decommissioning criteria. |
| Dataset redistribution readiness | not_applicable | The project does not redistribute source datasets, raw audio, embeddings, fitted models or participant predictions. | [@gratch2014distress; @valstar2016avec; @tao2023androids; @nautsch2019gdprSpeech] | Users need separate access to underlying datasets under original terms. |

## Interpretation

The project is paper-ready as a retrospective research study of cross-corpus and cross-task transfer. It is not ready for clinical deployment, autonomous screening, diagnostic use or patient-facing decision support.
