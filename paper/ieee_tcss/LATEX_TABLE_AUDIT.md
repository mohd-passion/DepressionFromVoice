# LaTeX Table Audit

All `\input{tables/...}` paths in `main.tex` were checked. The six table snippets preserve the audited result distinctions between zero-shot transfer, target-domain adaptation and within-corpus evaluation.

| Order | Input path | File exists | Caption | Label | Tabular syntax present | Scope-note check |
| --- | --- | --- | --- | --- | --- | --- |
| Table 1 | `tables/table1_dataset_task_summary.tex` | yes | Dataset and task summary | `tab:dataset-task-summary` | yes | zero-shot, within |
| Table 2 | `tables/table2_daic_development_results.tex` | yes | DAIC development results | `tab:daic-development` | yes | scope note present/not applicable |
| Table 3 | `tables/table3_cross_corpus_android_interview.tex` | yes | Cross-corpus Android interview results | `tab:android-interview-transfer` | yes | zero-shot, adaptation, within, not external validation, not pure zero-shot |
| Table 4 | `tables/table4_android_reading_results.tex` | yes | Android reading results | `tab:android-reading` | yes | zero-shot, within |
| Table 5 | `tables/table5_cross_task_results.tex` | yes | Participant-coherent cross-task results | `tab:cross-task` | yes | within |
| Table 6 | `tables/table6_result_scope_claim_matrix.tex` | yes | Result-scope and claim matrix | `tab:result-scope` | yes | zero-shot, adaptation, within, not pure zero-shot |

No table content, metrics or interpretation categories were changed in Stage 4B.
