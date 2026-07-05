# Reference Cleanup Report

Stage 3A audited `paper/references.bib` for internal consistency. No reference was replaced or newly added during this stage.

## Summary

| Check | Result | Action |
| --- | --- | --- |
| BibTeX entry count | 31 entries | No action. |
| Duplicate citation keys | None found | No action. |
| Duplicate DOI values | None found | No action. |
| Duplicate source titles | No exact duplicate article/report titles found by manual review | No action. |
| DOI formatting | DOI values are stored without `https://doi.org/` prefixes | Consistent; no action. |
| Stable institutional URLs | WHO and NIST report/fact-sheet entries include stable institutional URLs | No action. |
| Acronym capitalization | DAIC-WOZ, AVEC, WavLM, PHQ-8, SUPERB, CONSORT-AI and DECIDE-AI capitalization is preserved where needed | No action. |
| arXiv entries with possible final publications | `chen2022wavlm`, `tao2023featureCorrelation`, `seneviratne2021generalized`, `tomashenko2021voicePrivacy` retain arXiv entries where final metadata was not fully verified in the local audit trail | Do not replace without evidence-based verification. |
| Conference versus journal versions | `nautsch2019gdprSpeech` uses INTERSPEECH DOI with arXiv URL; `tomashenko2021voicePrivacy` uses arXiv DOI and notes the related journal DOI | Acceptable for current verified metadata. |

## Notes

- The citation audit confirmed that all 28 citation keys used in `paper/MANUSCRIPT_DRAFT.md` exist in `paper/references.bib`.
- Three BibTeX entries are currently unused in the manuscript but retained because they support companion evidence tables or earlier audit artifacts: `zhang2025clinicalSpeech`, `yin2023transformerSpeech`, and `ravi2022speakerDisentanglement`.
- No fabricated venue, page, DOI, author, funding, ethics, or clinical-validation details were introduced.
