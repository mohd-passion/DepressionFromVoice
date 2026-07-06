# BSPC Table Formatting Plan

Source tables: `paper/PAPER_RESULTS_TABLES.md`

Guideline basis: BSPC/Elsevier tables should be editable text, cited in text, numbered sequentially, captioned, and use notes below the body. Vertical rules and shading should be avoided.

Stage 3E update: table callouts were inserted into `paper/BSPC_MANUSCRIPT_DRAFT.md` for Tables 1-6. Stage 3F retained all six audited tables in the DOCX-ready manuscript source because each table provides a distinct safeguard for dataset scope, protocol scope or result interpretation. Table content remains in `paper/PAPER_RESULTS_TABLES.md`; final editable table layout is still deferred until DOCX conversion/manual Word formatting.

| Table | Stage 3F decision | Required formatting | Rounding | Footnotes | Abbreviations | Result-scope caveats | BSPC suitability |
|---|---|---|---|---|---|---|---|
| Table 1: Dataset and task summary | main manuscript | Editable text; concise caption; avoid vertical rules. | Counts as integers. | DAIC test labels provisional; Android labels HC/PT group labels. | Define DAIC-WOZ, HC, PT. | Dataset/task differences limit direct clinical interpretation. | suitable_main |
| Table 2: DAIC development results | main manuscript | Editable text; consider compact rows if page length is tight. | Four decimals for ROC-AUC. | Phase 1B/1C train-only nested CV; Phase 1F not official hidden AVEC. | Define ROC-AUC and CV. | Development/provisional local test only. | suitable_main |
| Table 3: Cross-corpus Android interview results | main manuscript | Editable text; preserve protocol columns. | Four decimals. | Protocols B/C use Android labels and are not pure zero-shot. | Define PR-AUC, Brier, macro F1. | Android-trained row is within-Android, not external validation. | suitable_main |
| Table 4: Android reading results | main manuscript | Editable text; keep zero-shot versus Android-trained contrast. | Four decimals; CI as reported. | Android-trained reading is within-corpus nested CV. | Define CI. | Not independent external validation. | suitable_main |
| Table 5: Participant-coherent cross-task results | main manuscript | Editable text; may need landscape/compact formatting. | Four decimals; intervals as reported. | All rows within Androids Corpus; pooled gains include zero. | Define matched, cross-task, pooled. | Pooled advantage is suggestive, not conclusive. | suitable_main_or_supplement_if_space_limited |
| Table 6: Result-scope and claim matrix | main manuscript | Editable text; can move to supplement if BSPC length constraints require it. | Not metric-heavy. | Validation scope is methodological, not clinical. | Define external validation and target-domain training. | Prevents overclaiming; keep somewhere in submission package. | suitable_main_or_supplement_if_space_limited |

Do not remove caveats that distinguish zero-shot external evaluation, target-domain adaptation, within-corpus validation and clinical validation.

## Stage 3E Table Callout Map

| Callout | Manuscript placement | Status |
|---|---|---|
| Table 1 | Materials and Methods, Datasets | callout_inserted |
| Table 2 | Results 4.1, DAIC representation development | callout_inserted |
| Table 3 | Results 4.2-4.3, DAIC-to-Android transfer and adaptation | callout_inserted |
| Table 4 | Results 4.4, Android reading-speech evaluation | callout_inserted |
| Table 5 | Results 4.6, participant-coherent cross-task transfer | callout_inserted |
| Table 6 | Results 4.7, classifier-boundary analysis/result-scope caveats | callout_inserted |
