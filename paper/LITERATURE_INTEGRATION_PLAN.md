# Literature Integration Plan

Stage 1 plan only. Do not integrate references into `paper/MANUSCRIPT_DRAFT.md` until the next stage verifies bibliographic metadata and source quality.

## Batch A - Dataset and model primary sources

Scope:

- DAIC-WOZ
- AVEC
- Androids Corpus
- WavLM
- PHQ-8

Required placeholders:

- REF-007 `[REFERENCE NEEDED: DAIC-WOZ dataset description]`
- REF-008 `[REFERENCE NEEDED: Androids Corpus dataset description]`
- REF-015 `[REFERENCE NEEDED: WavLM paper]`
- Related but currently non-placeholder need: PHQ-8 scoring/validation support for DAIC-WOZ label description.

Available sources:

- `AVEC-2013.pdf`
- `AVEC 2014.pdf`
- `AVEC-2016.pdf`
- `AVEC-2019.pdf`
- `Automated Depression Recognition Using.pdf`
- `Harnessing multimodal approaches for.pdf`
- `The Androids Corpus A New Publicly Available.pdf`
- `The Relationship Between Speech Features changes when u get depressed-feature correlation.pdf`
- `WavLM Large-Scale Self-Supervised Pre-Training.pdf`
- `pHQ-8(Question).pdf`

Missing sources:

- Original or canonical DAIC-WOZ dataset citation, if not fully covered by local AVEC materials.
- PHQ-8 primary validation/scoring citation.
- Complete WavLM venue/year metadata.
- Complete Androids Corpus publication year/venue metadata.

Expected manuscript sections:

- Abstract, for concise framing if needed.
- 1. Introduction, especially DAIC-WOZ/Androids comparison.
- 3.2 Datasets.
- 3.4 WavLM representation.
- Related Work subsections on AVEC/DAIC and WavLM.

Recommended verification order:

1. Verify WavLM bibliographic metadata from the local PDF and external bibliographic record in a later stage.
2. Verify Androids Corpus title, venue, year, and DOI if any.
3. Verify which local AVEC paper best supports DAIC-WOZ/E-DAIC facts used in the manuscript.
4. Replace the PHQ-8 form with a proper PHQ-8 validation/scoring citation.
5. Only after verification, decide whether the manuscript needs one citation per dataset sentence or a compact citation cluster.

## Batch B - Depression speech evidence

Scope:

- Acoustic and prosodic correlates.
- Timing and pause behaviour.
- Reading versus spontaneous speech.

Required placeholders:

- REF-001 `[REFERENCE NEEDED: depression and speech as behavioral signal]`
- REF-003 `[REFERENCE NEEDED: acoustic and prosodic correlates of depression]`
- REF-005 `[REFERENCE NEEDED: handcrafted acoustic features in depression detection]`
- REF-010 `[REFERENCE NEEDED: reading versus spontaneous speech in clinical speech analysis]`
- REF-011 `[REFERENCE NEEDED: survey of speech-based depression detection]`
- REF-013 `[REFERENCE NEEDED: acoustic feature sets for depression speech]`
- REF-017 `[REFERENCE NEEDED: reading speech and spontaneous speech comparison]`

Available sources:

- `Speech-Based_Depression_Assessment_A_Comprehensive_Survey.pdf`
- `A Novel Decision Tree for Depression recognition in speech.pdf`
- `Depression Detection in Speech Using Transformer and Parallel.pdf`
- `Speech-Based_Depression_Detection_Enhancing_Emotional_Support_via_Clinical_Data.pdf`
- `The Androids Corpus A New Publicly Available.pdf`
- `The Relationship Between Speech Features changes when u get depressed-feature correlation.pdf`

Missing sources:

- Primary clinical/acoustic studies for canonical depression-related speech changes.
- Broader clinical speech source comparing read and spontaneous speech outside Androids.
- Complete bibliographic metadata for several local primary candidates.

Expected manuscript sections:

- Abstract.
- 1. Introduction.
- 2.1 Speech-based depression detection.
- 2.2 Handcrafted acoustic and timing features.
- 2.5 Reading versus spontaneous speech.
- 5.4 Reading and interview speech.

Recommended verification order:

1. Use the survey to identify high-confidence primary sources for acoustic/prosodic correlates.
2. Verify one or two primary studies for handcrafted/timing features.
3. Verify Androids task details for read versus spontaneous comparisons.
4. Find a broader clinical speech/task source if the claim remains general rather than Androids-specific.
5. Consolidate overlapping placeholders before manuscript editing to avoid repetitive citation clusters.

## Batch C - Transfer methodology

Scope:

- Cross-corpus evaluation.
- Domain shift.
- Calibration.
- Task transfer.

Required placeholders:

- REF-006 `[REFERENCE NEEDED: self-supervised speech representation learning]`
- REF-009 `[REFERENCE NEEDED: domain shift and calibration in machine learning]`
- REF-012 `[REFERENCE NEEDED: evaluation variability in depression speech studies]`
- REF-014 `[REFERENCE NEEDED: wav2vec 2.0 and related SSL speech models]`
- REF-016 `[REFERENCE NEEDED: cross-corpus depression detection]`
- REF-018 `[REFERENCE NEEDED: domain adaptation and dataset shift taxonomy]`
- REF-019 `[REFERENCE NEEDED: classifier boundary shift under dataset shift]`
- REF-020 `[REFERENCE NEEDED: prior cross-corpus findings in depression speech]`
- REF-021 `[REFERENCE NEEDED: SSL transfer in paralinguistic or affective speech]`
- Related but currently non-placeholder need: metric citations for ROC-AUC, PR-AUC, balanced accuracy, macro F1, Brier score, and bootstrap confidence intervals if required by venue.

Available sources:

- `WavLM Large-Scale Self-Supervised Pre-Training.pdf`
- `SELF-SUPERVISED REPRESENTATIONS IN SPEECH-BASED DEPRESSION DETECTION.pdf`
- `HIERARCHICAL SELF-SUPERVISED REPRESENTATION LEARNING.pdf`
- `Investigation of Layer-Wise Speech Representations in Self-Supervised.pdf`
- `Depression recognition using voice‑based pre‑training model.pdf`
- `A Step Towards Preserving Speakers’ Identity While Detecting .pdf`
- `Speech-Based_Depression_Assessment_A_Comprehensive_Survey.pdf`
- `AVEC-2013.pdf`
- `AVEC 2014.pdf`
- `AVEC-2016.pdf`
- `AVEC-2019.pdf`

Missing sources:

- General wav2vec 2.0 primary paper.
- General SSL speech overview if needed.
- Canonical dataset-shift/domain-adaptation taxonomy source.
- Probability calibration and scoring-rule sources.
- Classifier-boundary/dataset-shift methodology source.
- Metric and bootstrap-uncertainty sources if the venue expects them.

Expected manuscript sections:

- 1. Introduction.
- 2.3 Self-supervised speech representations.
- 2.4 Cross-corpus depression detection.
- 2.6 Domain shift, calibration and transfer.
- 3.6 Evaluation protocols.
- 3.8 Metrics and uncertainty.
- 5.3 Corpus-specific decision boundaries.
- 5.7 Comparison with prior work.

Recommended verification order:

1. Verify WavLM and wav2vec/SSL primary sources.
2. Verify local depression-SSL and cross-corpus/cross-lingual papers.
3. Add independent ML methodology sources for dataset shift and calibration; do not rely on depression papers for these definitions.
4. Verify metric/bootstrap sources only if required by target venue.
5. Separate citations supporting method definitions from citations supporting depression-speech findings.

## Batch D - Clinical and ethical interpretation

Scope:

- Bias.
- Privacy.
- Screening versus diagnosis.
- Prospective validation.

Required placeholders:

- REF-002 `[REFERENCE NEEDED: depression burden and prevalence]`
- REF-004 `[REFERENCE NEEDED: ethical and clinical limits of speech-based mental-health models]`
- Related but currently non-placeholder need: prospective validation, clinical utility, dataset bias, and fairness references before submission.

Available sources:

- `Speech-Based_Depression_Assessment_A_Comprehensive_Survey.pdf`
- `A Step Towards Preserving Speakers’ Identity While Detecting .pdf`
- `LLAMADRSPrompting Large Language Models for Interview-Based.pdf`
- `a multi-modal open dataset for mental disorder analysis.pdf`
- `ODMA dataset a Multi-modal Open Dataset for Mental disorder analysis.pdf`
- `MADRS) Depression Rating Scale.pdf`

Missing sources:

- Authoritative depression prevalence/burden source.
- Clinical AI validation and prospective deployment source.
- Bias/fairness source for mental-health AI or speech models.
- Privacy source specific to speech/audio mental-health data.
- Source distinguishing screening support from clinical diagnosis.

Expected manuscript sections:

- 1. Introduction.
- 5.8 Clinical interpretation.
- 6. Limitations.
- 7. Conclusion if claims are broadened.

Recommended verification order:

1. Verify burden/prevalence source first because it anchors the introduction.
2. Verify clinical validation and screening-versus-diagnosis sources before touching clinical interpretation language.
3. Verify privacy and speaker-identity sources, including whether the local speaker-disentanglement paper is appropriate.
4. Verify bias/fairness sources separately from privacy sources.
5. Keep any future manuscript edits conservative: research classification findings should remain distinct from clinical validation claims.

## Stage 1 Stop Point

Do not replace placeholders or revise manuscript claims in this stage. The next stage should begin by completing bibliographic metadata for candidate sources and adding missing primary sources before any citation integration.
