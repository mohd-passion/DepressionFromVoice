# Journal-Neutral Front Matter

This file is an editable submission-package scaffold. Placeholder values are explicit and require author input where noted.

## Title

Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations

## Running Title

[AUTHOR INPUT REQUIRED: running title]

## Authors

[AUTHOR INPUT REQUIRED: author names and order]

## Affiliations

[AUTHOR INPUT REQUIRED: affiliations]

## Corresponding Author

[AUTHOR INPUT REQUIRED: corresponding author name, email and address]

## ORCID

[AUTHOR INPUT REQUIRED: ORCID identifiers if available]

## Abstract

[EDITABLE DRAFT FROM MANUSCRIPT]

Speech is a non-invasive behavioral signal that has been studied for depression assessment across acoustic, prosodic, timing and voice-quality markers. Models trained on one corpus often face shifts in language, recording conditions, elicitation task and label definition when applied elsewhere. This study evaluated whether frozen self-supervised WavLM representations preserved depression-related information across corpora and speech tasks, and whether failures of transfer reflected representation failure or classifier-boundary shift. The frozen DAIC-WOZ WavLM classifier achieved ROC-AUC 0.6412 in zero-shot transfer to Android interview speech, but both default and frozen DAIC thresholds predicted every Android interview participant as negative. Android-trained interview and reading classifiers achieved within-Android ROC-AUCs of 0.8675 and 0.8796, respectively. Participant-coherent Android cross-task transfer achieved ROC-AUCs of 0.8475 for reading-to-interview and 0.8501 for interview-to-reading; pooled reading-and-interview training achieved ROC-AUCs of 0.9058 and 0.9135. These are research classification findings, not clinical validation.

## Keywords

depression detection; speech analysis; self-supervised learning; WavLM; cross-corpus evaluation; domain shift; cross-task transfer; affective computing

## Funding

[AUTHOR INPUT REQUIRED: funding statement or "No external funding" if accurate]

## Conflicts of Interest

[AUTHOR INPUT REQUIRED: conflict-of-interest statement]

## Author Contributions

[AUTHOR INPUT REQUIRED: contribution taxonomy and author initials]

## Acknowledgements

[AUTHOR INPUT REQUIRED: acknowledgements or "None" if accurate]

## Ethics Statement

[AUTHOR INPUT REQUIRED: ethics/IRB statement for use of DAIC-WOZ and Androids Corpus data]

## Data Availability

The source datasets are not redistributed. Access to DAIC-WOZ and Androids Corpus data is controlled by the original dataset terms. Raw audio, embeddings, fitted models, participant predictions and participant-level artifacts are excluded from repository paper artifacts.

## Code Availability

Code, compact configuration records, evidence maps and the experiment registry are versioned in the repository. The frozen source tag for the manuscript is `paper-experiment-freeze-v1`.

## Consent Statement

[AUTHOR INPUT REQUIRED: participant consent statement based on original dataset documentation and institutional requirements]

## AI-Assistance Disclosure

[AUTHOR INPUT REQUIRED: disclose AI-assisted drafting/auditing if required by target journal or institution]
