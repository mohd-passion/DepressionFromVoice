# Journal-Neutral Front Matter

This file is an editable submission-package scaffold. Stage 3H author and declaration details have been applied; portal-only submission checks remain pending.

## Title

Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations

## Running Title

[AUTHOR INPUT REQUIRED: running title]

## Authors

1. Mohammad Ilyas
2. Dr. Vikram Singh

## Affiliations

Mohammad Ilyas, Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India.

Dr. Vikram Singh, Department of Computer Engineering, National Institute of Technology, Kurukshetra, Haryana, India.

## Corresponding Author

Mohammad Ilyas, Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India. Email: mohd.passion.cse@mitmoradabad.edu.in.

Dr. Vikram Singh is not the corresponding author.

## ORCID

Mohammad Ilyas: not available.

Dr. Vikram Singh: not available.

## Abstract

[EDITABLE DRAFT FROM MANUSCRIPT]

Speech is a non-invasive behavioral signal that has been studied for depression assessment across acoustic, prosodic, timing and voice-quality markers. Models trained on one corpus often face shifts in language, recording conditions, elicitation task and label definition when applied elsewhere. This study evaluated whether frozen self-supervised WavLM representations preserved depression-related information across corpora and speech tasks, and whether failures of transfer reflected representation failure or classifier-boundary shift. The frozen DAIC-WOZ WavLM classifier achieved ROC-AUC 0.6412 in zero-shot transfer to Android interview speech, but both default and frozen DAIC thresholds predicted every Android interview participant as negative. Android-trained interview and reading classifiers achieved within-Android ROC-AUCs of 0.8675 and 0.8796, respectively. Participant-coherent Android cross-task transfer achieved ROC-AUCs of 0.8475 for reading-to-interview and 0.8501 for interview-to-reading; pooled reading-and-interview training achieved ROC-AUCs of 0.9058 and 0.9135. These are research classification findings, not clinical validation.

## Keywords

depression detection; speech analysis; self-supervised learning; WavLM; cross-corpus evaluation; domain shift; cross-task transfer; affective computing

## Funding

No funding was received for this study.

## Conflicts of Interest

The authors declare no competing interests.

## Author Contributions

Mohammad Ilyas: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization, Project administration.

Dr. Vikram Singh: Supervision.

## Acknowledgements

The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets. Mohammad Ilyas acknowledges the academic guidance received during his Ph.D. research work.

## Ethics Statement

This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.

Author-review note: check this wording against institutional policy and the dataset licence terms before submission.

## Data Availability

The source datasets are not redistributed. Access to DAIC-WOZ and Androids Corpus data is controlled by the original dataset terms. Raw audio, embeddings, fitted models, participant predictions and participant-level artifacts are excluded from repository paper artifacts.

## Code Availability

Code, compact configuration records, evidence maps and the experiment registry are versioned in the repository. The frozen source tag for the manuscript is `paper-experiment-freeze-v1`.

## Consent Statement

This study used secondary datasets obtained under their original access terms. Consent for the original data collection was managed by the respective dataset providers. No new human participants were recruited in this study.

## AI-Assistance Disclosure

ChatGPT and Codex were used to assist with code organization, manuscript drafting support, editing, and workflow planning. The authors reviewed and verified the manuscript content, results, references, and final scientific claims.

## Preprint Statement

A preprint is planned.

## BSPC Submission Route

The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.
