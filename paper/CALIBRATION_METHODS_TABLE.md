# Calibration Methods Table

This table separates calibration, threshold choice and classifier retraining for Stage 2C. It is intended to prevent overstating target-domain adaptation as zero-shot external validation.

| Method or protocol | What is fitted | Uses target-domain labels? | Changes score ranking? | Changes predicted probabilities? | Stage 2C role | Literature support | Guardrail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No calibration / frozen source probabilities | Nothing beyond the already frozen source model | No | No | No | Phase 2B and Phase 2C Protocol A used the frozen DAIC score scale directly. | `vanCalster2019calibration` | A poorly calibrated frozen score can still have above-chance ROC-AUC. |
| Intercept-only recalibration | A calibration intercept or baseline-risk adjustment | Yes, if fit on target labels | No for monotonic score transforms | Yes | Conceptual comparator for probability-scale shift, not the primary reported protocol. | `vanCalster2019calibration` | Requires target-domain outcomes; not zero-shot evaluation. |
| Logistic calibration / Platt scaling | A logistic mapping from model scores to probabilities | Yes, if fit on target labels | Usually preserves ranking when monotonic | Yes | Representative calibration family for frozen-score calibration. | `niculescuMizil2005probabilities`; `vanCalster2019calibration` | Improves probability reliability only if target calibration data are adequate. |
| Isotonic regression | A nonparametric monotonic mapping from scores to probabilities | Yes, if fit on target labels | Can introduce ties while preserving monotonic order | Yes | Representative nonparametric calibration option. | `niculescuMizil2005probabilities` | Can overfit with limited calibration data. |
| Temperature scaling | A scalar rescaling of logits or scores before probability conversion | Yes, if fit on target labels | No when monotonic | Yes | Cited as a common modern calibration method, not necessarily the exact Stage 2C implementation. | `guo2017calibration` | More common for neural-network logits than for all logistic-regression settings. |
| Brier score evaluation | No fitted mapping; a probability-error metric is computed | No, except outcomes are needed for evaluation | No | No | Stage 2C reports Brier score to summarize probability error. | `brier1950probabilityForecasts`; `vanCalster2019calibration` | Lower Brier score does not imply higher ROC-AUC. |
| Threshold-only adaptation | A decision threshold selected on target-domain training folds | Yes | No | No | Phase 2C Protocol B improved macro F1 while preserving ROC-AUC 0.6412. | `morenoTorres2012datasetShift`; `vanCalster2019calibration` | Threshold-only adaptation is not calibration because it does not alter predicted probabilities. |
| Target-domain classifier retraining | A new classifier boundary on frozen target-domain training embeddings | Yes | Yes | Yes | Phase 2C Protocol D tested whether frozen WavLM features supported a new Android boundary. | `morenoTorres2012datasetShift`; `vanCalster2019calibration` | Within-Android validation, not DAIC-to-Android model transfer. |

## Stage 2C Interpretation Checks

- Target-domain threshold adaptation, not pure zero-shot evaluation: Protocol B uses Android labels to select thresholds.
- Target-domain score calibration rather than zero-shot evaluation: Protocol C uses Android labels to fit calibration mappings.
- Threshold-only adaptation does not alter predicted probabilities.
- Calibration can improve probability scores without improving ranking.
- Android-trained classifier results show target-domain classifier learning on frozen WavLM features, not DAIC-to-Android model transfer.
