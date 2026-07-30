# IEEE TCSS LaTeX Compile QA Report

Stage: 4E Overleaf package refresh after research-question alignment  
Branch: `paper/ieee-tcss-research-questions-v1`  
Target journal: IEEE Transactions on Computational Social Systems  
Manuscript source: `paper/ieee_tcss/main.tex`

## Android Reading/Interview MMD2 Figure 5 Compile

| Item | Result |
| --- | --- |
| Branch | `paper/android-reading-interview-mmd2-v1` |
| Source branch | `paper/ieee-tcss-figure-table-callouts-v1` |
| Android reading/interview MMD2 computed | yes |
| Primary paired-participant MMD2 | 0.2864 |
| Figure 5 updated | yes |
| Review PDF path | `paper/ieee_tcss/build/main.pdf` |
| PDF generated | yes |
| Page count | 15 |
| Final-pass unresolved citations | none |
| Final-pass unresolved references | none |
| Citation question marks in extracted PDF text | none |
| PDF text contains MMD2 values | yes: 0.6091 and 0.2864 |
| Overleaf ZIP status | stale after Figure 5 MMD2 edit |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The Android reading/interview MMD2 value was computed from existing participant-level WavLM layer-9 reading and interview embeddings using the recovered Phase 2C biased RBF MMD2 estimator and median squared-distance bandwidth rule. It is methodologically comparable in estimator, kernel, bandwidth rule, feature level and raw embedding scale to the DAIC-to-Android MMD2 audit, but it is a within-Android task comparison rather than a corpus-transfer comparison. The local MiKTeX build succeeded after Figure 5 and its caption were updated.

## Stage 4J Figure/Table Callout Compile

| Item | Result |
| --- | --- |
| Stage 4J branch | `paper/ieee-tcss-figure-table-callouts-v1` |
| Review PDF path | `paper/ieee_tcss/build/main.pdf` |
| PDF generated | yes |
| Page count | 15 |
| All figures cited in text | yes |
| All tables cited in text | yes |
| No undefined figure references | yes |
| No undefined table references | yes |
| No duplicate labels | yes |
| No citation question marks | yes |
| Overleaf ZIP status | stale after manuscript edit |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

Stage 4J inserted one first textual callout for each of the five figures and six tables, using `Fig.~\ref{...}` and `Table~\ref{...}`. The local MiKTeX build succeeded, final-pass logs contained no unresolved citation or reference warnings and extracted PDF text contained no `??` markers. No captions, table values, figure files, references, author metadata, clinical claims or deployment claims were changed. Overleaf ZIP must be regenerated after this edit.

## Stage 4I Final Overleaf Package Refresh

| Item | Result |
| --- | --- |
| Stage 4I branch | `paper/ieee-tcss-intro-evaluation-prospects-v1` |
| Review PDF path | `paper/ieee_tcss/build/main.pdf` |
| PDF generated | yes |
| Page count | 14 |
| Compile method | `build_latex.ps1` with local MiKTeX on PATH for the process |
| Unresolved citations | none in final pass |
| Unresolved references | none in final pass |
| Missing figures | none detected |
| Missing tables | none detected |
| Overleaf ZIP path | `paper/ieee_tcss/overleaf_upload/ieee_tcss_overleaf_package.zip` |
| ZIP entry count | 15 |
| Research-gap framing in ZIP | yes |
| All four RQs in ZIP | yes |
| Evaluation-findings paragraph in ZIP | yes |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The Stage 4I package was regenerated after the Stage 4H Introduction evaluation-findings and future-prospects paragraph. The ZIP contains only `main.tex`, `declarations.tex`, `bib/references.bib`, five PNG figures, six table inputs and `supplementary/supplementary.tex`.

## Stage 4H Introduction Evaluation-Prospects Compile

| Item | Result |
| --- | --- |
| Stage 4H branch | `paper/ieee-tcss-intro-evaluation-prospects-v1` |
| Review PDF path | `paper/ieee_tcss/build/main.pdf` |
| PDF generated | yes |
| Page count | 14 |
| Compile method | `build_latex.ps1` with local MiKTeX on PATH for the process |
| Unresolved citations | none in final pass |
| Unresolved references | none in final pass |
| Missing figures | none detected |
| Missing tables | none detected |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The Stage 4H compile verified the Introduction paragraph connecting research gaps, RQs, evaluation findings and future prospects. No manuscript metrics, references, author metadata, clinical validation claim, deployment-readiness claim or speaker-disentanglement claim was changed.

## Stage 4G Final Overleaf Package Refresh

| Item | Result |
| --- | --- |
| Stage 4G branch | `paper/ieee-tcss-research-gaps-v1` |
| Review PDF path | `paper/ieee_tcss/build/main.pdf` |
| PDF generated | yes |
| Page count | 14 |
| Compile method | `build_latex.ps1` with local MiKTeX on PATH for the process |
| Unresolved citations | none in final pass |
| Unresolved references | none in final pass |
| Missing figures | none detected |
| Missing tables | none detected |
| Overleaf ZIP path | `paper/ieee_tcss/overleaf_upload/ieee_tcss_overleaf_package.zip` |
| ZIP entry count | 15 |
| Research-gap framing in ZIP | yes |
| All four RQs in ZIP | yes |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The Stage 4G package was regenerated after the Stage 4F research-gap integration. The ZIP contains only `main.tex`, `declarations.tex`, `bib/references.bib`, five PNG figures, six table inputs and `supplementary/supplementary.tex`.

## Stage 4F Compile Addendum

| Item | Result |
| --- | --- |
| Stage 4F branch | `paper/ieee-tcss-research-gaps-v1` |
| Fresh Stage 4F compile-check PDF path | `paper/ieee_tcss/build/stage4f_main.pdf` |
| PDF generated | yes |
| Page count | 14 |
| Compile method | Direct MiKTeX `pdflatex` and `bibtex` executable paths, job name `stage4f_main` |
| Unresolved citations | none in final pass |
| Unresolved references | none in final pass |
| Missing figures or tables | none detected |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The primary build script was attempted first, but the current shell did not expose `pdflatex` or `bibtex` on PATH. A direct MiKTeX compile verified the Stage 4F research-gap source without changing manuscript metrics, references, author metadata or submission state.

## Compile Result

| Item | Result |
| --- | --- |
| Local TeX toolchain | MiKTeX with `pdflatex` and `bibtex` available |
| Primary review PDF path | `paper/ieee_tcss/build/main.pdf` |
| Fresh compile-check PDF path | `paper/ieee_tcss/build/stage4e_main.pdf` |
| PDF generated | yes |
| Page count | 14 |
| Primary script status | `build/main.pdf` was locked by another process, so `build_latex.ps1` could not overwrite it |
| Stage 4E workaround | Recompiled the same copied `main.tex` under job name `stage4e_main` |
| IEEE upload/submission | not performed |
| Overleaf upload | not performed |

The existing `build/main.pdf` remains available for review. The fresh Stage 4E compile check produced `build/stage4e_main.pdf` from the latest RQ-aligned source.

## Final-Pass QA Checks

| Check | Result |
| --- | --- |
| Unresolved citations | none in final pass |
| Unresolved references | none in final pass |
| Missing figures | none detected |
| Missing tables | none detected |
| Figure count in manuscript | 5 PNG figure includes |
| Table input count in manuscript | 6 table inputs |
| RQ markers in manuscript source | RQ1, RQ2, RQ3 and RQ4 present |
| Bibliography command | `\bibliography{bib/references}` present |

## Warnings for Visual Review

- `lettersize` is reported as an unused global option by the installed IEEEtran class.
- Text page 7 contains only floats.
- Two overfull vbox warnings are reported, both 8.29189pt too high.
- Underfull hbox/vbox warnings remain and should be checked visually.

## Scope Controls

- No experiments were run.
- No metrics, dataset counts, references, author details, ORCID identifiers or IEEE membership details were changed.
- No clinical diagnosis, clinical validation, screening utility or deployment-readiness claim was added.
- No IEEE submission, IEEE portal upload or Overleaf upload was performed.
