# IEEE TCSS LaTeX Compile QA Report

Stage: 4E Overleaf package refresh after research-question alignment  
Branch: `paper/ieee-tcss-research-questions-v1`  
Target journal: IEEE Transactions on Computational Social Systems  
Manuscript source: `paper/ieee_tcss/main.tex`

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
