# IEEE Figure and Table Callout Audit

Stage: 4J figure/table callout audit

Branch: `paper/ieee-tcss-figure-table-callouts-v1`

Manuscript source: `paper/ieee_tcss/main.tex`

Audit date: 2026-07-29

## Summary

All five figure labels and all six table labels already existed before Stage 4J. No labels were renamed and no duplicate labels were added. Stage 4J inserted one concise first-callout sentence for each figure and each table using IEEE-style `Fig.~\ref{...}` and `Table~\ref{...}` references.

## Android MMD2 Addendum

Branch `paper/android-reading-interview-mmd2-v1` updated Figure 5 after computing a verified Android reading/interview MMD2 value of 0.2864 from existing participant-level WavLM layer-9 embeddings. The Figure 5 caption now reports both DAIC-to-Android MMD2 0.6091 and Android reading/interview MMD2 0.2864. This addendum does not alter the Stage 4J callout findings: all figure and table labels and first-callout references remain unchanged.

## Callout Audit

| Item | Label | File/source | Caption summary | Existing callout found | Recommended insertion location | Final status |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 1 | `fig:study-design` | `figures/figure1_study_design_protocol_overview.png` | Study design and protocol overview | no | End of `Study design` subsection, before first figure float | complete: cited with `Fig.~\ref{fig:study-design}` |
| Figure 2 | `fig:daic-phase1` | `figures/figure2_daic_phase1_key_results.png` | DAIC Phase 1F key results | no | End of `DAIC representation development`, before Figure 2 float and Table 2 input | complete: cited with `Fig.~\ref{fig:daic-phase1}` |
| Figure 3 | `fig:transfer-adaptation` | `figures/figure3_daic_android_transfer_adaptation.png` | DAIC-to-Android transfer and adaptation behavior | no | End of `Domain-shift diagnosis`, before Figure 3 float | complete: cited with `Fig.~\ref{fig:transfer-adaptation}` |
| Figure 4 | `fig:android-cross-task` | `figures/figure4_android_cross_task_results.png` | Android reading, interview and cross-task results | no | End of `Reading-versus-interview comparison`, before Figure 4 float | complete: cited with `Fig.~\ref{fig:android-cross-task}` |
| Figure 5 | `fig:domain-shift-boundary` | `figures/figure5_domain_shift_boundary_summary.png` | Domain-shift and classifier-boundary summary with DAIC-to-Android MMD2 0.6091 and Android reading/interview MMD2 0.2864 | no | End of `Classifier-boundary analysis`, before Figure 5 float and Table 6 input | complete: cited with `Fig.~\ref{fig:domain-shift-boundary}` |
| Table 1 | `tab:dataset-task-summary` | `tables/table1_dataset_task_summary.tex` | Dataset and task summary | no | End of `Androids Corpus` dataset subsection, before Table 1 input | complete: cited with `Table~\ref{tab:dataset-task-summary}` |
| Table 2 | `tab:daic-development` | `tables/table2_daic_development_results.tex` | DAIC development results | no | End of `DAIC representation development`, before Figure 2 float and Table 2 input | complete: cited with `Table~\ref{tab:daic-development}` |
| Table 3 | `tab:android-interview-transfer` | `tables/table3_cross_corpus_android_interview.tex` | Cross-corpus Android interview results | no | End of `Zero-shot DAIC-to-Android interview transfer`, before Table 3 input | complete: cited with `Table~\ref{tab:android-interview-transfer}` |
| Table 4 | `tab:android-reading` | `tables/table4_android_reading_results.tex` | Android reading results | no | End of `Android reading-speech evaluation`, before Table 4 input | complete: cited with `Table~\ref{tab:android-reading}` |
| Table 5 | `tab:cross-task` | `tables/table5_cross_task_results.tex` | Participant-coherent cross-task results | no | End of `Participant-coherent cross-task transfer`, before Table 5 input | complete: cited with `Table~\ref{tab:cross-task}` |
| Table 6 | `tab:result-scope` | `tables/table6_result_scope_claim_matrix.tex` | Result-scope and claim matrix | no | End of `Classifier-boundary analysis`, before Figure 5 float and Table 6 input | complete: cited with `Table~\ref{tab:result-scope}` |

## Integrity Checks

- All figures cited in text: yes.
- All tables cited in text: yes.
- No undefined figure references: yes.
- No undefined table references: yes.
- No duplicate labels: yes.
- No citation question marks: yes.
- No `Figure \ref{...}` callouts introduced: yes.
- No `Tab. \ref{...}` callouts introduced: yes.

## Scope Controls

- No experiment was run.
- No metric, dataset count, table value or figure content was changed.
- No reference was added, removed or renamed.
- No author metadata, ORCID identifier or IEEE membership status was changed.
- No clinical validation, deployment-readiness or speaker-disentanglement claim was introduced.
- No IEEE upload, Overleaf upload or submission was performed.
