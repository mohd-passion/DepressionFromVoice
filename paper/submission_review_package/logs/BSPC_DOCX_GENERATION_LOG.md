# BSPC DOCX Generation Log

Date generated: 2026-07-09T15:42:09+05:30
Branch: `paper/bspc-final-declarations-v1`
Commit at generation: `9c8ec12`
Design preset: `standard_business_brief`

## Stage 3H Final Declaration Metadata

- Stage 3H branch: `paper/bspc-final-declarations-v1`
- Source branch: `paper/bspc-final-review-package-v1`.
- Final author/declaration details applied: yes.
- Author approvals applied: figures approved, manuscript DOCX approved and supplementary DOCX approved.
- Dr. Vikram Singh email applied: `viks@nitkkr.ac.in`.
- ORCID status applied: not available for both authors; no ORCID identifier was invented.
- CRediT statement applied: Mohammad Ilyas full supplied role list; Dr. Vikram Singh: Supervision.
- Consent, acknowledgements and AI disclosure applied from Stage 3H author input.
- Preprint status applied: a preprint is planned.
- BSPC route applied: subscription route.
- No Elsevier/BSPC upload, journal-generated PDF proof approval or submission confirmation was performed.

## Consolidation Metadata

- Consolidation branch retained in history: `paper/bspc-final-review-package-v1`
- Source branch confirmed: `paper/bspc-review-docx-v1` for Stage 3G review package scripts, figures and DOCX package structure.
- Source branch confirmed: `paper/experiment-rationale-summary-v1` for experiment-rationale documentation.
- Source branch confirmed: `paper/voice-physiology-motivation-v1` for the physiology motivation subsection.
- Merge status: `paper/bspc-review-docx-v1` was already an ancestor of the consolidation branch; no merge conflicts occurred.
- Physiology subsection included: yes, `2.2 Physiological motivation for speech-based depression analysis`.
- Experiment-rationale documentation included: yes.
- Review DOCX files regenerated after Stage 3H declarations: yes.
- Final-review DOCX copies generated: yes.
- No Elsevier/BSPC submission or final PDF generation was performed.

## Source Markdown Files

- `paper/BSPC_DOCX_READY_MANUSCRIPT.md`
- `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md`
- `paper/BSPC_FIGURE_CAPTIONS.md`

## Output Files

- `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`
- `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`
- `paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx`
- `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx`

## Tooling

- Tool used: python-docx 1.2.0-compatible generator
- python-docx available: yes
- python-docx import error: not_applicable
- Figures inserted: yes
- Main figures inserted: 5 of 5
- LibreOffice/soffice available for render QA: no
- Visual render QA status: not_completed_soffice_unavailable
- Final PDF generated: no
- Submission/upload performed: no

## Portal-Only Markers Retained

- yes: `[AUTHOR FINAL CONFIRMATION REQUIRED: the manuscript is not under consideration elsewhere.]`
- yes: `[AUTHOR FINAL ACTION REQUIRED: final BSPC/Elsevier portal metadata entry and upload.]`
- yes: `[AUTHOR FINAL ACTION REQUIRED: final BSPC submission-system technical checks.]`
- yes: `[AUTHOR FINAL REVIEW REQUIRED: final cover letter review and journal-generated PDF proof check.]`

## Known Formatting Limitations

- Review DOCX files are generated from Markdown and are not submitted BSPC files.
- References remain a bibliography placeholder and must be generated from `paper/references.bib` later.
- Stage 3H author/declaration fields are applied; portal-only review markers remain visible.
- LibreOffice/soffice availability is recorded below; page-image render QA depends on that local renderer.
- page-image render QA via `render_docx.py` could not be completed in this environment because LibreOffice/soffice was unavailable on PATH.
- Final Word pagination and BSPC artwork compliance still require portal/manual review.

## Next Manual Checks Needed

- Final cover letter author review, including the not-under-consideration confirmation.
- Final BSPC/Elsevier portal metadata entry.
- Final portal upload of manuscript, supplement and required figure/source files.
- Final BSPC submission-system technical checks.
- Journal-generated PDF proof review before any submission confirmation.
- Final references formatting from `paper/references.bib`.
