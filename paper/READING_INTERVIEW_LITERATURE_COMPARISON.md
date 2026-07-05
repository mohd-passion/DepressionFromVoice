# Reading and Interview Literature Comparison

This table supports the Stage 2B manuscript framing for reading versus spontaneous/interview speech. It does not support cross-corpus transfer methodology, domain adaptation, calibration or clinical deployment claims.

| Row | Source | Dataset or population | Reading task evidence | Interview or spontaneous evidence | Feature or representation focus | Main comparison used in manuscript | Limitation carried forward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `tao2023androids` | Androids Corpus, 118 native Italian speakers with healthy-control and depression-patient groups | Same short Aesop fairy-tale reading task; 112 read-speech recordings reported in the local paper and project audit | Spontaneous/interview recordings with interviewer questions; 116 full interview recordings and paired participants reported in the corpus/project audit | Corpus design and benchmark task structure | Androids permits direct read/interview comparison when participant leakage is controlled. | Supports corpus facts only; not this project's WavLM metrics or clinical validity. |
| 2 | `tao2023featureCorrelation` | Androids read-speech subset with 112 speakers, including 58 depressed participants | Uses read speech partly because it reduces topic variability not necessarily related to pathology | Does not provide a full spontaneous/interview comparison in the integrated claim | RMS energy, MFCCs, F0, zero-crossing rate, voicing probability and feature correlations | Reading speech can be useful when the goal is to reduce lexical/topic variability. | Read-speech emphasis only; does not prove reading/interview interchangeability. |
| 3 | `leal2025survey` | Survey of speech-based depression assessment literature | Reviews read-speech tasks among other elicitation paradigms | Reviews interview, spontaneous and prompted speech among other paradigms | Survey-level task, feature and dataset taxonomy | Speech type is a meaningful study-design variable for speech depression recognition. | Secondary source; task-level generalizations should remain cautious. |
| 4 | `valstar2016avec` | DAIC-WOZ/AVEC guided-interview challenge context | No reading task | Wizard-of-Oz/guided interview speech in DAIC-WOZ/AVEC | COVAREP and acoustic challenge baselines | Guided interview speech preserves conversational timing and interviewer/participant structure unavailable in read-only corpora. | Does not provide a reading comparison; used only as interview-feature context. |

## Manuscript Interpretation

- Reading speech is described as more lexically controlled, not as clinically superior.
- Interview speech is described as preserving spontaneous fluency, turn-taking and latency, not as automatically more informative.
- Androids is the primary source for the paired task design; the project's cross-task WavLM results remain internally supported by the experiment registry and `paper/PAPER_EVIDENCE_MAP.md`.
