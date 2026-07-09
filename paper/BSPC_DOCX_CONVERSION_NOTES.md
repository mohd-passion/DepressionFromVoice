# BSPC DOCX Conversion Notes

Status: `docx_pending_tooling_or_manual_conversion`

Stage 3F prepared the DOCX-ready markdown source at `paper/BSPC_DOCX_READY_MANUSCRIPT.md`. No PDF was generated. DOCX generation was not performed because `pandoc` was not available on the local PATH during Stage 3F verification.

Stage 3H regenerated the BSPC review DOCX package with `python-docx` and applied final author/declaration details. These notes remain as a record of the Stage 3F Pandoc check and as manual Word/portal guidance.

## Recommended Conversion Tool

Preferred command if Pandoc is installed later:

```powershell
pandoc paper/BSPC_DOCX_READY_MANUSCRIPT.md `
  -o paper/BSPC_DOCX_READY_MANUSCRIPT.docx `
  --bibliography=paper/references.bib
```

Optional supplementary conversion:

```powershell
pandoc paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md `
  -o paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.docx
```

## Stage 3F Tool Check

- Tool checked: `pandoc`
- Result: unavailable on PATH
- DOCX generated: no
- PDF generated: no
- Validation impact: not a failure; validation must not require Pandoc or Microsoft Word.

## Manual Word Checks

- Confirm title, author list, affiliations and corresponding-author details during portal entry.
- Confirm that Dr. Vikram Singh email, ORCID availability, corresponding-author status and Supervision-only CRediT role match the Stage 3H final-review package.
- Confirm consent, acknowledgements, AI-assisted writing/tool-use disclosure, preprint decision and BSPC subscription route during portal entry.
- Insert figures from SVG where accepted by Word/BSPC, otherwise use high-resolution PNG review copies or journal-required converted artwork.
- Ensure all figures and tables are cited in order.
- Check that captions remain directly below figures/tables or in the journal-required caption list.
- Convert tables to editable Word tables; avoid images of tables.
- Define abbreviations in table notes and first manuscript use.
- Confirm reference style and citation numbering against BSPC/Elsevier requirements.
- Add line numbering if requested by the submission system.
- Add page numbering if requested by the submission system.
- Check that no portal-only reminder is accidentally removed before the corresponding portal action is complete.

## Figure Insertion Instructions

Use files from `paper/figures/output/`:

- Figure 1: `figure1_study_design_protocol_overview.svg` or `.png`
- Figure 2: `figure2_daic_phase1_key_results.svg` or `.png`
- Figure 3: `figure3_daic_android_transfer_adaptation.svg` or `.png`
- Figure 4: `figure4_android_cross_task_results.svg` or `.png`
- Figure 5: `figure5_domain_shift_boundary_summary.svg` or `.png`

Final artwork should be checked at journal size for readability, line weight, color contrast and any BSPC-specific file-format requirements.

## Table Formatting Instructions

- Use editable text tables from `paper/PAPER_RESULTS_TABLES.md`.
- Preserve protocol-scope caveats distinguishing zero-shot, target-adapted and within-corpus results.
- Keep Table 5 readable; landscape orientation may be needed.
- Keep Table 6 in the main manuscript unless final page/length constraints require moving it to supplementary material.

## File Naming Convention

Suggested files after author review:

- `BSPC_Manuscript_MohammadIlyas_DrVikramSingh.docx`
- `BSPC_Supplementary_Material.docx`
- `BSPC_Figure1_StudyDesign.svg`
- `BSPC_Figure2_DAICPhase1.svg`
- `BSPC_Figure3_DAICAndroidTransfer.svg`
- `BSPC_Figure4_AndroidCrossTask.svg`
- `BSPC_Figure5_DomainShiftBoundary.svg`

Do not submit until final portal metadata entry, upload, BSPC submission-system checks, cover letter author review, journal-generated PDF proof review and submission confirmation are complete.
