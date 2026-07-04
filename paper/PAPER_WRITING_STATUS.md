# Paper Writing Status

## Source Freeze

- Frozen source tag: `paper-experiment-freeze-v1`
- Tag commit: `1bf9434f047c0b69b90c1d5264e665d6ccd941a0`
- Manuscript branch: `paper/manuscript-v1`
- Remote: `https://github.com/mohd-passion/DepressionFromVoice.git`

## Working Title

Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations

## Draft Metrics

- Manuscript draft word count: 5,060
- Reference placeholders: 21
- Result tables prepared: 6
- Planned figures: 7
- Evidence-map entries: 99

## Alternative Titles

1. Domain Shift in Frozen WavLM Speech Representations for Cross-Corpus Depression Detection
2. Reading and Interview Speech for Depression Classification with Frozen Self-Supervised Speech Features
3. Cross-Task Transfer and Pooled Training for Depression Detection from Frozen WavLM Representations

## Completed Sections

- Title
- Abstract
- Keywords
- Introduction
- Related Work with citation placeholders
- Materials and Methods
- Results
- Discussion
- Limitations
- Conclusion
- Evidence map
- Results tables
- Figure plan
- Claim audit
- Reference needs
- Abbreviation list

## Sections Requiring Literature References

- Depression background and burden
- Speech-based depression detection
- Acoustic and timing features
- DAIC-WOZ dataset description
- Androids Corpus dataset description
- Self-supervised speech representation learning
- WavLM
- Cross-corpus depression detection
- Domain shift and calibration
- Reading versus spontaneous speech
- Clinical validation and ethical limitations

## Sections Requiring Author Review

- Abstract framing and target journal fit
- Related Work scope and final citations
- Whether to include DAIC timing Candidate B prominently or mainly as a portability contrast
- How strongly to emphasize pooled-task training given confidence intervals including zero
- Whether Android HC/PT labels should be described with additional corpus-specific clinical detail after citation review

## Tables Prepared

1. Dataset and task summary
2. DAIC development results
3. Cross-corpus Android interview results
4. Android reading results
5. Participant-coherent cross-task results
6. Result-scope and claim matrix

## Figures Planned

1. End-to-end experimental pipeline
2. Phase 1-2E experimental progression
3. Zero-shot versus adapted Android performance
4. Reading/interview and cross-task ROC-AUC comparison
5. Domain-shift feature summary
6. Classifier-boundary similarity analysis
7. Participant-coherent fold design

## Unresolved Methodological Questions

- Which journal or conference format should determine word count and section order?
- Should DAIC local labeled test results be moved to supplementary material because official hidden AVEC provenance was not established?
- Should timing Candidate B be retained as a secondary DAIC result even though it cannot transfer directly to Android?
- Should confidence intervals be added to every table row where compact records provide them, or kept only for primary Android results?

## Claims Requiring Cautious Wording

- Zero-shot transfer: ranking signal was present, but thresholded predictions failed.
- Android-trained performance: within-corpus only, not external validation.
- Cross-task transfer: strong within Android, not cross-corpus transfer.
- Pooled-task training: strongest point estimates, but gains were suggestive.
- Clinical interpretation: no diagnostic or deployment claim.

## Next Writing Task

Replace citation placeholders with verified references, then revise the Related Work and Discussion to match the selected venue and reference style.
