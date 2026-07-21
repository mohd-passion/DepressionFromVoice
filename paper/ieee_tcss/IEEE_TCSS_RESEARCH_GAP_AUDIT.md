# IEEE TCSS Research Gap Audit

Stage: 4F research-gap framing  
Branch: `paper/ieee-tcss-research-gaps-v1`  
Source branch: `paper/ieee-tcss-research-questions-v1`  
Target manuscript: `paper/ieee_tcss/main.tex`

## Current Gap Coverage

- The Introduction already described cross-corpus shift between DAIC-WOZ and Androids Corpus, including language, elicitation, label definition and recording context.
- The Introduction already separated zero-shot transfer, target-domain threshold adaptation, target-domain calibration and target-domain classifier retraining.
- The Introduction already described speech-task variation between interview and reading speech.
- The Stage 4D RQs already covered DAIC-WOZ representation/timing evaluation, WavLM layer and aggregation choices, DAIC-to-Android transfer, and Android reading/interview task effects.
- Ethical and data-governance sections already noted that voice can contain information about identity, language, accent, health, emotion and recording environment.

## Missing or Weak Gap Statements

- Speaker-identity and voiceprint confounding were not explicitly framed as modeling risks in the Introduction.
- The Introduction did not explicitly connect speaker-specific voice characteristics to representation validity.
- The manuscript did not state that a systematic framework is needed to separate representation, aggregation, timing, threshold, calibration, task and corpus effects before the RQs.
- The Discussion did not explicitly warn that the study does not disentangle depression-related cues from biometric voice characteristics.

## Potential Overclaims

- Claiming that the study fully separates depression cues from speaker identity would be unsupported.
- Claiming complete speaker-disentanglement, biometric invariance or causal depression markers would be unsupported.
- Treating Android-trained within-corpus results as external validation would be unsupported.
- Treating threshold-adapted or calibrated protocols as pure zero-shot transfer would be unsupported.
- Clinical diagnosis, clinical validation and deployment-readiness claims remain prohibited.

## Recommended Insertion Location

The recommended insertion location was the Introduction immediately after the existing paragraph beginning `This study developed and audited...` and immediately before the four RQs. This preserves the flow:

1. general motivation;
2. prior speech-depression features and SSL representations;
3. corpus and task shift motivation;
4. explicit research gaps;
5. RQs;
6. contributions.

## Relationship to RQ1

The speaker/voiceprint confounding gap supports RQ1 by motivating a cautious DAIC-WOZ evaluation of whether WavLM and timing features encode participant-level depression-related behavioral signal without treating that signal as speaker-independent or diagnostic.

## Relationship to RQ2

The systematic-framework gap supports RQ2 by motivating layer and aggregation analysis rather than reporting one frozen representation as if representation design were fixed or universal.

## Relationship to RQ3

The cross-corpus gap supports RQ3 by motivating DAIC-to-Android transfer, threshold adaptation, score calibration and target-domain classifier-boundary analysis.

## Relationship to RQ4

The cross-task gap supports RQ4 by motivating Android reading/interview within-task, cross-task and pooled-task analysis under participant-coherent folds.

## Inserted Gap Themes

- Speaker-specific voice characteristics and voiceprint confounding risk.
- Cross-task limitations between spontaneous interview speech and read speech.
- Cross-corpus limitations across population, recording, elicitation, labels and class balance.
- Need for a framework separating representation choice, aggregation, timing descriptors, threshold selection, calibration, task and corpus effects.

## Speaker-Disentanglement Check

The manuscript now states that WavLM representations may encode speaker-specific information together with behavioral cues and that the reported experiments should not be interpreted as having disentangled depression markers from biometric voice characteristics. No claim of full speaker-disentanglement was added.

## Scope Controls

- No experiments were run.
- No metrics, dataset counts, references, author metadata, ORCID identifiers or IEEE membership details were changed.
- No clinical diagnosis, clinical validation, screening utility or deployment-readiness claim was added.
- No limitations were removed.
