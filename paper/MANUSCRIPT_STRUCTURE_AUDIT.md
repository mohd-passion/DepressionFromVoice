# Manuscript Structure Audit

| Section | Issue | Severity | Recommended change | Change implemented | Remaining author decision |
| --- | --- | --- | --- | --- | --- |
| Abstract | Length is 253 words and within the requested 220-300 word range. | low | Keep as journal-neutral abstract. | No text change needed. | Confirm target venue abstract limit later. |
| Abstract | It distinguishes zero-shot transfer from within-Android training and states major clinical limitation. | low | Preserve current scope wording. | No text change needed. | None. |
| Introduction | Motivation, cross-corpus gap and contributions are explicit. | low | Keep contribution list conservative. | No text change needed. | Decide whether to shorten for target venue. |
| Related Work | Organized thematically across speech depression, handcrafted features, SSL, cross-corpus work, task variation and calibration. | low | Avoid adding direct metric comparisons across incompatible studies. | No text change needed. | None. |
| Methods | Reproducibility details include preprocessing, WavLM commit, layer, aggregation, classifier and leakage safeguards. | low | Keep protocol labels explicit. | No text change needed. | Decide whether some implementation details move to supplement. |
| Methods | Registry and compact-record governance is documented. | low | Preserve data-exclusion language. | No text change needed. | None. |
| Results | Chronological flow follows Phase 1 through Phase 2E. | low | Keep result subsections research-question driven. | No text change needed. | None. |
| Results | All major numbers are traceable through `FINAL_NUMERICAL_AUDIT.csv`. | low | Keep evidence-map linkage. | Audit file added. | None. |
| Discussion | Separates representation transfer from classifier transfer and treats pooled-task gains cautiously. | low | Preserve caution on confidence intervals. | No text change needed. | None. |
| Ethical and Clinical Considerations | Strongly states research-only status and missing validation/fairness/utility work. | low | Keep section separate from technical limitations. | No text change needed. | Author may tailor ethics wording to venue. |
| Limitations | Includes technical, dataset, statistical, fairness, privacy and clinical limitations. | low | Avoid repeating all Stage 2D governance detail. | No text change needed. | None. |
| Conclusion | Concise and avoids deployment claims. | low | Keep as journal-neutral close. | No text change needed. | None. |
| Data and Code Availability | States dataset restrictions, excluded artifacts and frozen tag. | low | Keep until journal-specific data policy is known. | Audit file added. | Author must verify final repository visibility and dataset-license wording. |

## Overall Assessment

The manuscript is internally consistent enough for author review. It is not submission-ready because author, affiliation, ethics, funding, conflict, journal, final figure and format details remain incomplete.
