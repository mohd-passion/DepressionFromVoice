# Voice Physiology Insert Audit

## Source Material Used

- User-supplied source draft: `Papers/Paper-Voice(1).docx`.
- Used only as narrow background context for respiratory, phonatory, resonatory and articulatory speech-production mechanisms; source-filter framing; vocal-fold vibration, subglottal pressure and vocal-tract shaping; and cautious depression-related acoustic manifestations.
- No long sections, dataset tables, diagnostic background or methodological claims were copied from the uploaded draft.

## Manuscript Location

- `paper/BSPC_MANUSCRIPT_DRAFT.md`, Related Work, Section 2.2: "Physiological motivation for speech-based depression analysis".
- `paper/BSPC_DOCX_READY_MANUSCRIPT.md`, Related Work, Section 2.2: "Physiological motivation for speech-based depression analysis".

## Word Count

- Inserted subsection word count: 261 words.

## Citations Used

- `leal2025survey`
- `valstar2016avec`
- `tao2023featureCorrelation`
- `chen2022wavlm`
- `yang2021superb`

All citation keys already exist in `paper/references.bib` and are covered by the verified manuscript bibliography/audit materials. No new bibliography entries were added.

## Claims Added

- Speech production involves coordinated respiratory, phonatory, resonatory and articulatory processes.
- Source-filter framing can connect vocal-fold vibration, subglottal pressure, vocal-tract shaping and articulation to measurable acoustic properties.
- Depression-related speech analysis is biologically and behaviorally motivated by possible effects on psychomotor activity, emotional expressivity, cognitive effort and physiological arousal.
- Reported depression-related acoustic patterns may include reduced pitch variability, lower energy, slower articulation, longer pauses, flatter intonation, breathier voice quality and less regular timing, with explicit task and population caveats.
- WavLM is framed as a frozen speech representation for transfer analysis, not as a clinical diagnostic model.

## Claims Deliberately Excluded

| Excluded material from uploaded draft | Reason for exclusion |
| --- | --- |
| Full abstract or broad paper framing | The task requested a concise subsection only. |
| Long DSM-5 diagnostic description | Would distract from the BSPC contribution and risk unsupported medical expansion. |
| Long biopsychosocial determinants discussion | Too broad for the focused manuscript refinement. |
| Old or conflicting DAIC-WOZ dataset counts | Frozen dataset counts are already validated elsewhere in the manuscript. |
| PHQ-8 severity scoring details | Avoids conflict with the validated manuscript label description. |
| Broad real-time monitoring or decision-support language | Not demonstrated by the experiments. |
| Privacy-preserving assessment claims | Not established by this study. |
| Fairness or trustworthiness claims from normalization | No fairness validation was performed. |
| Clinical diagnostic or deployment claims | The study is research-only and not clinically validated. |

## Overclaiming Check

- The insert says acoustic patterns "should not be treated as diagnostic markers by themselves".
- The insert states that corpus labels are evaluated under controlled research protocols rather than clinical diagnosis.
- The insert states that WavLM is not a standalone clinical model.
- No new experiment result, participant count, confidence interval, registry record or frozen metric was added.

## Clinical-Claim Check

- No claim that speech can diagnose depression was introduced.
- No claim that the system is clinically validated was introduced.
- No claim of screening utility, deployment readiness, clinical benefit or privacy-preserving assessment was introduced.
- No unverified medical reference or new citation key was introduced.
