# BSPC Reference Formatting Plan

Current citation style: Pandoc/BibTeX-style manuscript citation keys such as `[@chen2022wavlm]`.

Target BSPC/Elsevier style: numeric references in square brackets, numbered in order of appearance, with consistent reference-list metadata and DOI use where available.

## Conversion Needed

| Item | Status | Action |
|---|---|---|
| Citation style conversion | needs_formatting | Convert manuscript citations from author-key syntax to numeric style only after final author approval. |
| Reference list generation | needs_formatting | Generate from `paper/references.bib` using CSL/Pandoc or reference manager tooling if available. |
| Strict submission formatting | ready | BSPC does not require strict reference formatting at first submission if references are complete and consistent. |
| DOI checking | needs_verification | Final DOI/URL verification should be run before submission. |
| Dataset/software references | needs_formatting | Ensure dataset/software references are represented consistently if included. |
| Web reference access dates | needs_formatting | Add access dates for web references if any remain in final reference list. |

## Tooling

| Tool | Availability | Notes |
|---|---|---|
| BibTeX file | available | `paper/references.bib` contains the current reference metadata. |
| CSL/Pandoc workflow | needs_verification | Not run in Stage 3D. |
| Manual conversion | not_recommended | Avoid manual rewrite unless automated/reference-manager tooling cannot be used. |

## References Needing Final Metadata Review

- All entries in `paper/references.bib` should be checked for DOI/URL completeness before final submission.
- Any source-dataset references should be checked for dataset access wording consistency.
- Official journal-guide URLs used for preparation should not be added to the article reference list unless cited in the manuscript text.

No reference list was manually rewritten in Stage 3D.
