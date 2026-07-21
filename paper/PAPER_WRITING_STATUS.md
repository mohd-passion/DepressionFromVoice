# Paper Writing Status

## Source Freeze

- Current branch: `paper/bspc-final-declarations-v1`
- Frozen source tag: `paper-experiment-freeze-v1`
- Tag commit: `1bf9434f047c0b69b90c1d5264e665d6ccd941a0`
- Remote: `https://github.com/mohd-passion/DepressionFromVoice.git`
- Stage 3A status: full manuscript consistency audit and journal-neutral submission package prepared.
- Stage 3B status: author-input preparation and verified journal-selection package prepared.
- Stage 3C status: preliminary author preferences applied; Biomedical Signal Processing and Control selected as provisional first target; awaiting author confirmation for target journal.
- Stage 3D status: Biomedical Signal Processing and Control journal-specific preparation files drafted; awaiting author confirmations before figure generation, DOCX/PDF or submission.
- Stage 3E status: confirmed author details and declarations applied; BSPC main figure package generated; awaiting author review; no DOCX/PDF or submission.
- Stage 3F status: BSPC figure visual review completed; DOCX-ready markdown package prepared; supplementary draft and conversion notes prepared; DOCX not generated because Pandoc was unavailable; no PDF or submission.
- Stage 3G status: BSPC author-review DOCX package generated without Pandoc; review copies need author review; no final PDF or submission.
- Stage 3H status: BSPC author declarations finalized, author approvals applied, final-review DOCX files regenerated without Pandoc; portal metadata/upload/system checks, cover letter final review, journal-generated PDF proof review and submission confirmation remain pending; no submission.
- Stage 4A status: IEEE Transactions on Computational Social Systems LaTeX preparation package created on `paper/ieee-tcss-latex-v1`; no IEEE upload, portal account creation, final PDF or submission.
- Stage 4B status: IEEE TCSS template alignment, LaTeX source sanity, citation, figure, table and local-build readiness audits completed on `paper/ieee-tcss-template-compile-v1`; no PDF generated because `pdflatex` and `bibtex` are not on PATH.
- Stage 4D status: four IEEE TCSS research questions inserted and manuscript framing aligned on `paper/ieee-tcss-research-questions-v1`; no experiments run, metrics changed, references added, author metadata invented or clinical claims introduced.
- Stage 4E status: RQ-aligned Overleaf ZIP regenerated and PDF visual-review checklist prepared on `paper/ieee-tcss-research-questions-v1`; local compile-check succeeded, Overleaf upload remains manual, IEEE submission not performed.
- Experiment-rationale documentation status: prepared from official frozen project sources only; no experiments run and no metrics changed.
- Voice physiology motivation status: concise subsection added to BSPC manuscript draft and DOCX-ready source; no experiments run and no metrics changed.
- Final BSPC review package status: `bspc_final_review_package_ready_for_submission_portal_checks`; final author/declaration inputs and DOCX approvals applied, with portal-only checks pending.

## Current Counts

- Manuscript draft word count: 6111
- DOCX-ready manuscript markdown word count: 6840
- Reference count: 31
- Citation markers in manuscript: 99
- Unique manuscript citation keys: 28
- Unresolved citation placeholders in manuscript: 0
- Experiment registry records: 42
- Evidence-map entries: 99
- Result tables prepared: 6
- Planned/specification figure entries: 7
- Generated BSPC main figures: 5
- Generated BSPC supplement figures: 0
- Tables retained in DOCX-ready manuscript: 6
- BSPC review DOCX files generated: 4
- Experiment-rationale table rows: 25
- Voice physiology insert word count: 261
- Supplementary items planned: 11

## Working Title

Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations

## Completed Sections

- Title
- Abstract
- Keywords
- Introduction
- Related Work
- Materials and Methods
- Results
- Discussion
- Ethical and Clinical Considerations
- Limitations
- Conclusion
- Data and Code Availability
- References

## Completed Stage 3A Package Files

- `paper/FINAL_NUMERICAL_AUDIT.csv`
- `paper/FINAL_RESULT_SCOPE_AUDIT.md`
- `paper/FINAL_CITATION_AUDIT.csv`
- `paper/REFERENCE_CLEANUP_REPORT.md`
- `paper/TERMINOLOGY_STYLE_GUIDE.md`
- `paper/MANUSCRIPT_STRUCTURE_AUDIT.md`
- `paper/REDUNDANCY_EDIT_LOG.md`
- `paper/CONTRIBUTION_CLAIM_AUDIT.md`
- `paper/FINAL_TABLE_AUDIT.md`
- `paper/FIGURE_SPECIFICATIONS.md`
- `paper/SUPPLEMENTARY_MATERIAL_PLAN.md`
- `paper/JOURNAL_NEUTRAL_FRONT_MATTER.md`
- `paper/DATA_CODE_AVAILABILITY_AUDIT.md`
- `paper/FINAL_CLINICAL_CLAIM_AUDIT.md`
- `paper/SUBMISSION_READINESS_CHECKLIST.md`

## Completed Stage 3B Package Files

- `paper/AUTHOR_INPUT_FORM.md`
- `paper/AUTHOR_DECISION_LOG.md`
- `paper/MANUSCRIPT_AUTHOR_REVIEW_CHECKLIST.md`
- `paper/JOURNAL_CANDIDATE_MATRIX.md`
- `paper/JOURNAL_SELECTION_CRITERIA.md`
- `paper/JOURNAL_SHORTLIST.md`
- `paper/SUBMISSION_STRATEGY.md`
- `paper/JOURNAL_FIT_GAP_ANALYSIS.md`
- `paper/MANUSCRIPT_AUTHOR_REVIEW_COPY.md`
- `paper/JOURNAL_INFORMATION_VERIFICATION_LOG.md`

## Completed Stage 3C Package Files

- `paper/AUTHOR_PREFERENCES_PRELIMINARY.md`
- `paper/PROVISIONAL_TARGET_JOURNAL_DECISION.md`
- `paper/JOURNAL_ADAPTATION_PLAN.md`
- `paper/AUTHOR_FINAL_QUESTIONS.md`

## Completed Stage 3D Package Files

- `paper/BSPC_GUIDELINES_VERIFICATION.md`
- `paper/BSPC_ADAPTATION_PLAN.md`
- `paper/BSPC_MANUSCRIPT_DRAFT.md`
- `paper/BSPC_HIGHLIGHTS.md`
- `paper/BSPC_GRAPHICAL_ABSTRACT_PLAN.md`
- `paper/BSPC_DECLARATIONS_DRAFT.md`
- `paper/BSPC_FIGURE_PRODUCTION_PLAN.md`
- `paper/BSPC_TABLE_FORMATTING_PLAN.md`
- `paper/BSPC_REFERENCE_FORMATTING_PLAN.md`
- `paper/BSPC_COVER_LETTER_DRAFT.md`

## Completed Stage 3E Package Files

- `paper/BSPC_FIGURE_CAPTIONS.md`
- `paper/BSPC_FIGURE_DATA_AUDIT.csv`
- `paper/figures/README.md`
- `paper/figures/scripts/generate_bspc_figures.py`
- `paper/figures/source/bspc_figure_source_values.csv`
- `paper/figures/output/figure_generation_log.txt`
- `paper/figures/output/figure1_study_design_protocol_overview.svg`
- `paper/figures/output/figure1_study_design_protocol_overview.png`
- `paper/figures/output/figure2_daic_phase1_key_results.svg`
- `paper/figures/output/figure2_daic_phase1_key_results.png`
- `paper/figures/output/figure3_daic_android_transfer_adaptation.svg`
- `paper/figures/output/figure3_daic_android_transfer_adaptation.png`
- `paper/figures/output/figure4_android_cross_task_results.svg`
- `paper/figures/output/figure4_android_cross_task_results.png`
- `paper/figures/output/figure5_domain_shift_boundary_summary.svg`
- `paper/figures/output/figure5_domain_shift_boundary_summary.png`

## Completed Stage 3F Package Files

- `paper/BSPC_FIGURE_VISUAL_REVIEW.md`
- `paper/BSPC_DOCX_READY_MANUSCRIPT.md`
- `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md`
- `paper/BSPC_DOCX_CONVERSION_NOTES.md`
- Updated `paper/BSPC_COVER_LETTER_DRAFT.md`
- Updated `paper/AUTHOR_FINAL_QUESTIONS.md`
- Updated `paper/BSPC_TABLE_FORMATTING_PLAN.md`
- Updated `paper/SUBMISSION_READINESS_CHECKLIST.md`

## Completed Stage 3G Package Files

- `paper/scripts/create_bspc_review_docx.py`
- `paper/submission_review_package/README.md`
- `paper/submission_review_package/AUTHOR_REVIEW_PACKET_CHECKLIST.md`
- `paper/submission_review_package/logs/BSPC_DOCX_GENERATION_LOG.md`
- `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`
- `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`
- Updated `paper/AUTHOR_FINAL_QUESTIONS.md`
- Updated `paper/SUBMISSION_READINESS_CHECKLIST.md`

## Completed Stage 3H Package Files

- Updated `paper/AUTHOR_INPUT_FORM.md`
- Updated `paper/AUTHOR_DECISION_LOG.md`
- Updated `paper/AUTHOR_FINAL_QUESTIONS.md`
- Updated `paper/JOURNAL_NEUTRAL_FRONT_MATTER.md`
- Updated `paper/BSPC_DECLARATIONS_DRAFT.md`
- Updated `paper/BSPC_MANUSCRIPT_DRAFT.md`
- Updated `paper/BSPC_DOCX_READY_MANUSCRIPT.md`
- Updated `paper/BSPC_COVER_LETTER_DRAFT.md`
- Updated `paper/SUBMISSION_READINESS_CHECKLIST.md`
- Updated `paper/submission_review_package/README.md`
- Updated `paper/submission_review_package/AUTHOR_REVIEW_PACKET_CHECKLIST.md`
- Updated `paper/submission_review_package/logs/BSPC_DOCX_GENERATION_LOG.md`
- Regenerated `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`
- Regenerated `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`
- Generated `paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx`
- Generated `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx`

## Completed Experiment-Rationale Package Files

- `paper/STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md`
- `paper/STUDY_EXPERIMENTS_RATIONALE_TABLE.csv`
- `paper/EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md`
- `paper/EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md`
- `paper/EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md`
- `paper/EXPERIMENT_DEPENDENCY_FLOW.md`
- `paper/EXPERIMENT_SCOPE_BOUNDARIES.md`
- Updated `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md`
- Updated `paper/SUBMISSION_READINESS_CHECKLIST.md`

## Tables Prepared

1. Dataset and task summary
2. DAIC development results
3. Cross-corpus Android interview results
4. Android reading results
5. Participant-coherent cross-task results
6. Result-scope and claim matrix

## Planned Figures

1. Study design/protocol overview - generated, awaiting author review
2. DAIC Phase 1 key results - generated, awaiting author review
3. DAIC-to-Android transfer and adaptation - generated, awaiting author review
4. Android reading, interview and cross-task results - generated, awaiting author review
5. Domain-shift and classifier-boundary summary - generated, awaiting author review
6. Optional supplement: classifier-boundary fold metrics - not generated
7. Optional supplement: participant-coherent fold design - not generated

## Supplement Plan

Supplementary candidates include Phase 1 layer-wise results, aggregation ablation, timing/fusion ablations, selected C values, thresholds, calibration diagnostics, bootstrap intervals, domain-shift summaries, classifier-boundary fold metrics, claim/evidence governance and reproducibility workflow. Participant-level data, raw audio, embeddings, predictions and fitted models remain excluded.

## Remaining Portal-Only Work

- Confirm the cover-letter statement: `[AUTHOR FINAL CONFIRMATION REQUIRED: the manuscript is not under consideration elsewhere.]`
- Complete final BSPC/Elsevier portal metadata entry.
- Upload the manuscript, supplementary DOCX and required figure/source files in the BSPC portal.
- Complete final BSPC submission-system technical checks.
- Complete final author review of the cover letter.
- Review the journal-generated PDF proof.
- Confirm the final submission action.

## Remaining Formatting And Portal Work

- Final journal portal metadata entry.
- Final reference style and Word checks required by the portal.
- Final artwork-format checks required by the portal.
- Any required reporting-guideline checklist requested by the portal.
- Journal-generated PDF proof review after upload.

## Journal Selection Status

`bspc_final_review_package_ready_for_submission_portal_checks`

## Current Readiness Category

`bspc_final_review_package_ready_for_submission_portal_checks`

## Stage 3B Completion Notes

- Author details missing: author names, order, final affiliations, institutional emails, ORCID identifiers and corresponding-author designation.
- Declaration details missing: funding, competing interests, ethics approval or exemption, consent to participate, consent for publication, acknowledgements and AI-assisted writing disclosure.
- Journal preferences missing: maximum APC, no-APC requirement, open-access preference, indexing requirements, preferred audience and preprint preference.
- Candidate journals reviewed: Biomedical Signal Processing and Control; PLOS Digital Health; Frontiers in Digital Health; Computer Speech & Language; Artificial Intelligence in Medicine; Speech Communication.
- Primary candidates: Biomedical Signal Processing and Control; PLOS Digital Health; Frontiers in Digital Health.
- Provisional first target: Biomedical Signal Processing and Control.
- Final journal not selected.
- Figures not generated.
- Journal formatting not applied.
- Manuscript not submitted.

## Stage 3C Completion Notes

- Preliminary preferences recorded: no SCI/SCIE mandate, no Q1 mandate, Scopus acceptable if SCI/SCIE is unavailable, no-APC subscription route required, open access not mandatory and maximum APC not specified.
- Preferred audiences recorded: biomedical signal processing, speech/language processing, medical AI/digital health and affective computing.
- Biomedical Signal Processing and Control is provisional only because Stage 3B verified a subscription route with no publication fee and strong biomedical signal-processing fit.
- PLOS Digital Health and Frontiers in Digital Health are deferred under current preferences because their fully open-access APC models conflict with the preliminary no-APC/subscription-route requirement unless support is confirmed.
- Current journal guide, fee route, article limits, data/code, ethics, preprint, AI-disclosure and any indexing/ranking details must be reverified before final journal selection.
- Author confirmation remains required for final target, title, article type, author list/order, affiliation wording, ORCID, corresponding author, CRediT roles, funding, competing interests, ethics/exemption, consent, acknowledgements, preprint and AI-assisted writing disclosure.

## Stage 3D Completion Notes

- Official BSPC/Elsevier/ScienceDirect guidelines were checked on 2026-07-06 and recorded in `paper/BSPC_GUIDELINES_VERIFICATION.md`.
- BSPC manuscript preparation copy created at `paper/BSPC_MANUSCRIPT_DRAFT.md`.
- Abstract tightened below the 250-word BSPC cap and keywords reduced to seven.
- Separate BSPC highlights file prepared with five evidence-based highlights.
- Graphical abstract, figure, table and reference plans prepared; no final figures generated.
- Declaration and cover-letter drafts contain placeholders only and require author confirmation.
- Current official BSPC page verifies open-access APC and subscription/no-publication-fee routes, but final route preference remains author-confirmed.
- No author declarations, ORCID, funding, ethics, consent, conflicts or acknowledgements were invented.
- Manuscript not submitted, DOCX/PDF not generated and experiment registry records not modified.

## Stage 3E Author/Declaration Notes

- Final title decision applied: keep current title.
- Author 1 applied: Mohammad Ilyas, Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India; email `mohd.passion.cse@mitmoradabad.edu.in`; ORCID not available; corresponding author Yes.
- Author 2 applied: Dr. Vikram Singh, Department of Computer Engineering, National Institute of Technology, Kurukshetra, Haryana, India; Stage 3H later finalized email `viks@nitkkr.ac.in`, ORCID not available and corresponding-author status No.
- Funding statement applied: No funding was received for this study.
- Competing-interest statement applied: The authors declare no competing interests.
- Ethics wording applied: secondary analysis of previously collected datasets; no new participants recruited.
- Stage 3H later finalized consent, acknowledgements, AI-assisted writing/tool-use disclosure, preprint status and BSPC subscription-route wording.

## Stage 3E Figure Package Notes

- Five main BSPC figure candidates were generated as SVG and PNG files under `paper/figures/output/`.
- Figure captions were created in `paper/BSPC_FIGURE_CAPTIONS.md`.
- Every plotted value was audited in `paper/BSPC_FIGURE_DATA_AUDIT.csv`; the audit uses compact aggregate records only.
- Manuscript figure callouts were inserted for Figures 1-5 and table callouts for Tables 1-6.
- Optional supplementary figures were not generated in Stage 3E.
- No raw audio, participant identifiers, embeddings, fitted models, prediction files or participant-level outputs were read or committed.

## Stage 3F DOCX Package Notes

- Figure visual review completed in `paper/BSPC_FIGURE_VISUAL_REVIEW.md`; all five generated figures are `ready_for_author_review`.
- Main figure count: 5.
- Supplement figure count: 0.
- Table count retained in DOCX-ready manuscript: 6.
- DOCX-ready manuscript markdown prepared at `paper/BSPC_DOCX_READY_MANUSCRIPT.md`.
- Supplementary material draft prepared at `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md`.
- DOCX conversion notes prepared at `paper/BSPC_DOCX_CONVERSION_NOTES.md`.
- DOCX generated: no; Pandoc was unavailable on PATH during Stage 3F.
- PDF generated: no.
- Manuscript validation status before Stage 3F branch creation: passed.
- Unit-test status before Stage 3F branch creation: passed, 76 tests.
- Registry validation status before Stage 3F branch creation: passed, 42 records.
- No submission was performed.
- No metrics, author details, consent wording, acknowledgements, AI disclosure or clinical claims were invented.

## Stage 3G Review DOCX Package Notes

- Stage 3G review package status: `bspc_review_docx_generated_needs_author_review`.
- Review DOCX manuscript generated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`.
- Review DOCX supplement generated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`.
- DOCX generation tool: `python-docx`; Pandoc was not used.
- Five main BSPC figure PNG review copies were inserted into the manuscript DOCX.
- Supplementary figures generated: 0.
- Fallback text files generated: no.
- Final PDF generated: no.
- LibreOffice/soffice render QA: not completed because the renderer dependency was unavailable on PATH.
- DOCX structural QA confirmed the manuscript DOCX contains five embedded images, visible unresolved author placeholders and no submitted/submission-ready marker.
- Manuscript validation status after Stage 3G: passed.
- Unit-test status after Stage 3G: passed, 78 tests.
- Registry validation status after Stage 3G: passed, 42 records.
- No submission was performed.
- No author metadata, consent wording, acknowledgements, AI disclosure, BSPC route decision, frozen metrics or clinical claims were invented.

## Experiment-Rationale Documentation Notes

- Experiment-rationale summary prepared at `paper/STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md`.
- Experiment-rationale table prepared at `paper/STUDY_EXPERIMENTS_RATIONALE_TABLE.csv` with 25 rows.
- Reviewer explanation prepared at `paper/EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md`.
- Manuscript-ready insert prepared at `paper/EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md`; it was not inserted into the main manuscript.
- Presentation notes prepared at `paper/EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md`.
- Experiment dependency flow prepared at `paper/EXPERIMENT_DEPENDENCY_FLOW.md`.
- Experiment-scope boundaries prepared at `paper/EXPERIMENT_SCOPE_BOUNDARIES.md`.
- `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md` was updated only with a short pointer to these support files.
- No experiments were run.
- No experiment registry records, raw outputs, participant counts, metrics, confidence intervals or interpretation categories were changed.
- No new scientific claims were introduced beyond the existing audited evidence.

## Voice Physiology Motivation Notes

- Concise physiology motivation subsection inserted at `paper/BSPC_MANUSCRIPT_DRAFT.md`, Related Work Section 2.2.
- Matching DOCX-ready subsection inserted at `paper/BSPC_DOCX_READY_MANUSCRIPT.md`, Related Work Section 2.2.
- Voice physiology insert audit prepared at `paper/VOICE_PHYSIOLOGY_INSERT_AUDIT.md`.
- No clinical diagnostic, deployment, privacy-preserving, old DAIC count or PHQ-8 severity-scoring claims were introduced.

## Final BSPC Review Package Consolidation Notes

- Consolidation branch: `paper/bspc-final-review-package-v1`.
- Source branches consolidated: `paper/bspc-review-docx-v1`, `paper/experiment-rationale-summary-v1` and `paper/voice-physiology-motivation-v1`.
- Merge conflicts: none; `paper/bspc-review-docx-v1` was already an ancestor of the physiology branch.
- Final review package readiness category after Stage 3G consolidation: `bspc_final_review_package_generated_needs_author_review`.
- Review DOCX manuscript regenerated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`.
- Review DOCX supplement regenerated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`.
- The regenerated manuscript DOCX includes the physiology motivation subsection and five main figure PNG review copies.
- Experiment-rationale documentation remains available in the paper support files.
- Stage 3H later resolved Dr. Vikram Singh email, ORCID availability, CRediT role, consent wording, acknowledgements, AI/tool-use disclosure, preprint decision and final BSPC route decision.
- LibreOffice/soffice render QA: attempted with `render_docx.py` but not completed because the renderer dependency was unavailable on PATH.
- Final consolidation validation status: passed.
- Final consolidation unit-test status: passed, 80 tests.
- Final consolidation registry validation status: passed, 42 records.
- No submission was performed.
- No metrics, references, author details, declarations or clinical claims were invented.

## Stage 3H Final Declaration Package Notes

- Stage 3H branch: `paper/bspc-final-declarations-v1`.
- Readiness category: `bspc_final_review_package_ready_for_submission_portal_checks`.
- Biomedical Signal Processing and Control selected for the Stage 3H final review package.
- Author details finalized: Mohammad Ilyas, Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India; email `mohd.passion.cse@mitmoradabad.edu.in`; ORCID not available; corresponding author Yes.
- Author details finalized: Dr. Vikram Singh, Department of Computer Engineering, National Institute of Technology, Kurukshetra, Haryana, India; email `viks@nitkkr.ac.in`; ORCID not available; corresponding author No.
- CRediT statement finalized: Mohammad Ilyas: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization, Project administration.
- CRediT statement finalized: Dr. Vikram Singh: Supervision.
- Funding finalized: No funding was received for this study.
- Competing interests finalized: The authors declare no competing interests.
- Ethics finalized: This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.
- Consent finalized: This study used secondary datasets obtained under their original access terms. Consent for the original data collection was managed by the respective dataset providers. No new human participants were recruited in this study.
- Acknowledgements finalized: The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets. Mohammad Ilyas acknowledges the academic guidance received during his Ph.D. research work.
- AI disclosure finalized: ChatGPT and Codex were used to assist with code organization, manuscript drafting support, editing, and workflow planning. The authors reviewed and verified the manuscript content, results, references, and final scientific claims.
- Preprint status finalized: A preprint is planned.
- BSPC route finalized: The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.
- Figures approved: all five main BSPC figures.
- Manuscript DOCX approved for final review package: yes.
- Supplementary DOCX approved for final review package: yes.
- Stage 3H DOCX generation tool: `python-docx`; Pandoc was not used.
- Stage 3H manuscript final-review DOCX generated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx`.
- Stage 3H supplementary final-review DOCX generated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx`.
- No Elsevier/BSPC upload, journal-generated PDF proof approval or submission confirmation was performed.
- Remaining portal-only tasks: final portal metadata entry, final portal upload, final BSPC submission-system checks, final cover letter author review, journal-generated PDF proof review and submission confirmation.
- Manuscript validation status after Stage 3H: passed.
- Unit-test status after Stage 3H: passed, 80 tests.
- Registry validation status after Stage 3H: passed, 42 records.
- No clinical claims, dataset counts, PHQ scoring details, frozen metrics or unverified references were introduced.

## Stage 4A IEEE TCSS LaTeX Migration Notes

- Stage 4A branch: `paper/ieee-tcss-latex-v1`.
- Source branch: `paper/bspc-final-declarations-v1`.
- Target journal changed for preparation only: IEEE Transactions on Computational Social Systems.
- IEEE TCSS LaTeX package created at `paper/ieee_tcss/`.
- Working manuscript created at `paper/ieee_tcss/main.tex` using `\documentclass[lettersize,journal]{IEEEtran}`.
- Bibliography copied to `paper/ieee_tcss/bib/references.bib`.
- Five approved main figures copied to `paper/ieee_tcss/figures/` as SVG and PNG files.
- Six audited result tables converted under `paper/ieee_tcss/tables/`.
- Supplementary draft converted to `paper/ieee_tcss/supplementary/supplementary.tex`.
- IEEE TCSS guideline verification recorded in `paper/ieee_tcss/IEEE_TCSS_GUIDELINES_VERIFICATION.md` using official IEEE sources only.
- IEEE TCSS journal-fit audit recorded in `paper/ieee_tcss/IEEE_TCSS_JOURNAL_FIT_AUDIT.md`.
- IEEE reference conversion audit recorded in `paper/ieee_tcss/IEEE_REFERENCE_CONVERSION_AUDIT.md`; missing citation keys: none.
- Title alternatives recorded in `paper/ieee_tcss/TITLE_REFRAMING_OPTIONS.md`; the current title was retained pending author decision.
- Uploaded template exception recorded: `/mnt/data/bare_jrnl_new_sample4.tex` was not found in the local workspace or Codex attachment cache, so `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex` records the missing-template condition and is not asserted to be the original uploaded template.
- Remaining IEEE author confirmations: ORCID identifiers if available, final title choice, IEEE biography wording if required, and the cover-letter not-under-consideration statement.
- No IEEE upload, portal account creation, submission, final PDF generation, new experiments, registry modification, frozen-metric change, ORCID invention, IEEE membership invention, clinical validation claim or deployment-readiness claim was performed.

## Stage 4B IEEE TCSS Template and Compile-Readiness Notes

- Stage 4B branch: `paper/ieee-tcss-template-compile-v1`.
- Source branch: `paper/ieee-tcss-latex-v1`.
- Template search result: the real uploaded `/mnt/data/bare_jrnl_new_sample4.tex` was not found; only the Stage 4A placeholder existed at `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex`.
- Template alignment audit added at `paper/ieee_tcss/IEEE_TEMPLATE_ALIGNMENT_AUDIT.md`.
- LaTeX source sanity audit added at `paper/ieee_tcss/LATEX_SOURCE_SANITY_AUDIT.md`.
- Figure audit added at `paper/ieee_tcss/LATEX_FIGURE_AUDIT.md`; all `main.tex` figure includes use PNG files, not SVG files.
- Table audit added at `paper/ieee_tcss/LATEX_TABLE_AUDIT.md`; all six table input files exist and preserve zero-shot, adapted and within-corpus distinctions.
- Reference conversion audit updated; all 28 `main.tex` citation keys resolve against `paper/ieee_tcss/bib/references.bib`, with no duplicate BibTeX keys.
- Local LaTeX setup notes added at `paper/ieee_tcss/LOCAL_LATEX_SETUP_NOTES.md`.
- Build scripts improved at `paper/ieee_tcss/build_latex.ps1` and `paper/ieee_tcss/build_latex.py`.
- Build result: no PDF generated because `pdflatex` and `bibtex` are not on PATH; `paper/ieee_tcss/logs/LATEX_BUILD_LOG.md` records the missing-toolchain result.
- No IEEE upload, portal account creation, submission, final PDF generation, new experiments, registry modification, frozen-metric change, reference addition, author-metadata invention, ORCID invention, IEEE membership invention, clinical validation claim or deployment-readiness claim was performed.

## Stage 4D IEEE TCSS Research-Question Alignment Notes

- Stage 4D branch: `paper/ieee-tcss-research-questions-v1`.
- Source branch: `paper/ieee-tcss-template-compile-v1`.
- Four research questions were inserted near the end of the Introduction in `paper/ieee_tcss/main.tex`, before the contribution list.
- The contribution list was aligned with RQ1 through RQ4.
- Results signposts were added sparingly to connect DAIC development, DAIC-to-Android transfer, domain-shift diagnosis and Android task analyses to the RQs.
- Discussion now includes `Findings in Relation to the Research Questions`.
- The Conclusion was revised to answer the four RQs concisely and cautiously.
- RQ audit created at `paper/ieee_tcss/IEEE_TCSS_RESEARCH_QUESTIONS_AUDIT.md`.
- RQ-to-evidence map created at `paper/ieee_tcss/RQ_TO_EVIDENCE_MAP.md`.
- No abstract change was needed.
- Build result: local LaTeX build succeeded and generated `paper/ieee_tcss/build/main.pdf` for review only; no IEEE upload or submission was performed.
- No experiments were run; no experiment registry records, dataset counts, table values, figure values, frozen metrics, references, author details, ORCID identifiers, IEEE membership details, clinical validation claims or deployment-readiness claims were changed or invented.

## Stage 4E IEEE TCSS Overleaf Refresh and PDF Visual Review Notes

- Stage 4E branch: `paper/ieee-tcss-research-questions-v1`.
- RQ-aligned Overleaf ZIP regenerated at `paper/ieee_tcss/overleaf_upload/ieee_tcss_overleaf_package.zip`.
- Overleaf README created at `paper/ieee_tcss/overleaf_upload/OVERLEAF_UPLOAD_README.md`.
- Overleaf ZIP manifest created at `paper/ieee_tcss/overleaf_upload/OVERLEAF_ZIP_MANIFEST.md`.
- PDF visual-review checklist created at `paper/ieee_tcss/IEEE_PDF_VISUAL_REVIEW_CHECKLIST.md`.
- LaTeX compile QA report created at `paper/ieee_tcss/IEEE_LATEX_COMPILE_QA_REPORT.md`.
- Local compile check produced `paper/ieee_tcss/build/stage4e_main.pdf`; the existing `paper/ieee_tcss/build/main.pdf` remained locked by another process and was not overwritten.
- PDF page count: 14.
- Final compile-check pass had no unresolved citations, unresolved references, missing figures or missing tables.
- Overleaf upload remains manual.
- IEEE portal upload and submission were not performed.
- Final PDF visual review remains pending.
- Manuscript-not-under-consideration confirmation remains pending.
- No experiments were run; no experiment registry records, dataset counts, table values, figure values, frozen metrics, references, author details, ORCID identifiers, IEEE membership details, clinical validation claims or deployment-readiness claims were changed or invented.

Do not mark `submitted` until portal upload, system checks, journal-generated PDF proof review and final submission confirmation are complete.
