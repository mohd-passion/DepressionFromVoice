# IEEE TCSS Submission Checklist

Status values: `complete`, `pending_author`, `pending_portal`, `not_verified`, `not_applicable`.

| Item | Status | Notes |
| --- | --- | --- |
| IEEE TCSS LaTeX package created | complete | `paper/ieee_tcss/main.tex` created. |
| Source branch recorded | complete | Branched from `paper/bspc-final-declarations-v1`. |
| Uploaded template preserved | not_verified | `/mnt/data/bare_jrnl_new_sample4.tex` was not found locally; placeholder note created. |
| Official TCSS guideline verification | complete | Official IEEE sources checked on 2026-07-13. |
| Journal-fit audit | complete | TCSS fit assessed as moderate with reframing needed. |
| Title decision | pending_author | Current title retained; alternatives documented. |
| Author block | complete | Confirmed names, affiliations and emails used; no ORCID or membership invented. |
| ORCID | pending_author | IEEE requires ORCID; author ORCID values currently not available. |
| Bibliography | complete | `paper/references.bib` copied to `paper/ieee_tcss/bib/references.bib`. |
| Citation-key validity | complete | LaTeX citations use copied BibTeX keys only. |
| Figures | complete | Five approved BSPC figures copied as SVG and PNG; PNG used in LaTeX for review compilation. |
| Tables | complete | Six audited result tables converted to LaTeX snippets. |
| Supplementary material | complete | Supplementary draft converted to `supplementary/supplementary.tex`. |
| Frozen metrics | complete | Metrics preserved; no experiments run. |
| Experiment registry | complete | Registry validation remains read-only. |
| Clinical-claim safeguards | complete | No clinical validation or deployment-readiness claim introduced. |
| Cover letter | pending_author | Draft created; not-under-consideration statement requires author confirmation. |
| IEEE portal metadata | pending_portal | No IEEE portal entry completed. |
| IEEE upload/submission | not_applicable | No upload or submission performed. |
| LaTeX build | complete | Stage 4D local pdfLaTeX build succeeded and generated `build/main.pdf` for review only; no IEEE upload or submission. See `logs/LATEX_BUILD_LOG.md`. |
| Template alignment audit | complete | `IEEE_TEMPLATE_ALIGNMENT_AUDIT.md` created; real uploaded template was not found, and the existing placeholder is not represented as the original template. |
| LaTeX source sanity audit | complete | `LATEX_SOURCE_SANITY_AUDIT.md` records passing static checks and missing TeX tooling. |
| Citation audit refresh | complete | All 28 citation keys used in `main.tex` resolve in `bib/references.bib`; no duplicate BibTeX keys found. |
| Figure include audit | complete | All `main.tex` includes use PNG files; no SVG file is included in the pdfLaTeX path. |
| Table input audit | complete | All six table inputs exist and retain zero-shot/adaptation/within-corpus distinctions. |
| Local LaTeX setup notes | complete | `LOCAL_LATEX_SETUP_NOTES.md` added with MiKTeX/TeX Live and local-build steps. |
| Build script robustness | complete | `build_latex.ps1` and `build_latex.py` check TeX tools, stage build files and record missing-toolchain status without modifying manuscript content. |
| Research questions | complete | Four manuscript-grounded RQs inserted in the Introduction and recorded in `IEEE_TCSS_RESEARCH_QUESTIONS_AUDIT.md`. |
| RQ-to-evidence map | complete | `RQ_TO_EVIDENCE_MAP.md` maps each RQ to sections, tables, figures, main answer and limitations. |
| RQ-aligned discussion and conclusion | complete | Discussion and Conclusion answer RQ1-RQ4 cautiously; no metrics, references, author metadata or clinical claims changed. |
| LaTeX compile QA report | complete | `IEEE_LATEX_COMPILE_QA_REPORT.md` records Stage 4E compile-check status, page count and warnings. |
| RQ-aligned Overleaf ZIP | complete | Fresh package generated at `overleaf_upload/ieee_tcss_overleaf_package.zip`; generated ZIP remains untracked by policy. |
| Overleaf ZIP manifest | complete | `overleaf_upload/OVERLEAF_ZIP_MANIFEST.md` lists all 15 included files and confirms RQ1-RQ4 are present. |
| Overleaf upload | pending_portal | Upload remains manual; no Overleaf upload was performed. |
| PDF visual review | pending_author | `IEEE_PDF_VISUAL_REVIEW_CHECKLIST.md` prepared; no review item is pre-approved. |
| Manuscript not under consideration elsewhere | pending_author | Confirmation remains required before IEEE portal submission. |
| IEEE portal submission | not_started | No IEEE portal upload or submission was performed. |
