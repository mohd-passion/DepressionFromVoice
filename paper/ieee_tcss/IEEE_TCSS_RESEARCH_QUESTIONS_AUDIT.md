# IEEE TCSS Research Questions Audit

Stage: 4D research-question alignment  
Branch: `paper/ieee-tcss-research-questions-v1`  
Source branch: `paper/ieee-tcss-template-compile-v1`  
Target manuscript: `paper/ieee_tcss/main.tex`

## Source Material Reviewed

- `paper/ieee_tcss/main.tex`
- `paper/ieee_tcss/tables/`
- `paper/ieee_tcss/figures/`
- `paper/ieee_tcss/supplementary/supplementary.tex`
- `paper/ieee_tcss/IEEE_TCSS_JOURNAL_FIT_AUDIT.md`
- `paper/ieee_tcss/IEEE_TCSS_MIGRATION_AUDIT.md`
- `paper/ieee_tcss/IEEE_REFERENCE_CONVERSION_AUDIT.md`
- `paper/PAPER_WRITING_STATUS.md`
- `paper/FINAL_RESULT_SCOPE_AUDIT.md`
- `paper/FINAL_NUMERICAL_AUDIT.csv`

## Final Research Questions

**RQ1:** How effectively do frozen self-supervised WavLM speech representations and timing-based speech features support participant-level depression-related behavioral signal classification on DAIC-WOZ?

**RQ2:** Which WavLM representation-design choices, particularly transformer-layer selection and participant-level aggregation strategy, influence DAIC-WOZ classification performance?

**RQ3:** To what extent does a DAIC-WOZ-trained WavLM classifier transfer to Androids Corpus speech, and what do threshold adaptation, score calibration and target-domain classifier learning reveal about calibration and classifier-boundary shift?

**RQ4:** How do Android reading and interview task conditions affect within-corpus, cross-task and pooled-task depression-related behavioral signal modeling?

## Manuscript Integration

- The four RQs were inserted near the end of the Introduction, after the motivation/gap sequence and before the contribution list.
- The contribution list was revised so each empirical contribution maps to RQ1, RQ2, RQ3 or RQ4.
- Short Results signposts were added for DAIC representation development, DAIC-to-Android transfer, domain-shift diagnosis, Android reading evaluation and participant-coherent cross-task transfer.
- A Discussion subsection, `Findings in Relation to the Research Questions`, was added to summarize the answer to each RQ using existing results only.
- The Conclusion was revised to answer the four RQs concisely and cautiously.
- The Abstract was not changed because it already summarizes the study question, protocols, major findings and non-clinical interpretation.

## Claim and Evidence Controls

- No experiments were run.
- No frozen metrics, dataset counts, table values or figure values were changed.
- No references or citation keys were added.
- No author details, ORCID identifiers or IEEE membership details were added.
- No clinical diagnosis, screening utility, deployment readiness or clinical validation claim was added.
- No limitations were removed.

## RQ-to-Claim Scope

| RQ | Evidence scope | Claim guardrail |
| --- | --- | --- |
| RQ1 | DAIC-WOZ Phase 1 WavLM, timing and provisional local labeled-test results | Described as participant-level depression-related behavioral signal classification, not diagnosis. |
| RQ2 | DAIC-WOZ Phase 1B layer analysis and Phase 1C aggregation analysis | Described as representation-design sensitivity, not a universal layer or aggregation rule. |
| RQ3 | Phase 2B/2C DAIC-to-Android interview transfer, threshold adaptation, calibration and Android-trained comparison; Phase 2D zero-shot reading transfer | Described as limited transfer, calibration shift and classifier-boundary shift, not successful deployment. |
| RQ4 | Phase 2D reading/interview comparison and Phase 2E participant-coherent cross-task and pooled-task results | Described as task effects and partial transfer within Android, not external validation. |

## Excluded Wording

- `proves`
- `diagnoses`
- `deployment-ready`
- `clinically validated`
- `universal model`
- any wording that treats Android-trained within-corpus results as DAIC-to-Android external validation
- any wording that treats threshold-adapted or calibrated protocols as pure zero-shot transfer

## Overclaiming Check

The revised RQ framing preserves the manuscript's existing interpretation: frozen WavLM representations can retain depression-related ranking information across corpus and task conditions, but classifier boundaries and probability scales are corpus- and task-sensitive. The text continues to frame all results as retrospective research classification findings.

## Clinical-Claim Check

The updated Introduction, Discussion and Conclusion do not claim clinical diagnosis, screening utility, clinical validation or deployment readiness. Existing ethical and limitation statements remain intact.
