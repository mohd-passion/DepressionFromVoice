# BSPC Figure Production Plan

No figures are generated in Stage 3D. This plan uses `paper/FIGURE_SPECIFICATIONS.md`.

| Figure | Placement | Required source data | Script needed | Output format | Expected caption | Author review needed | Privacy risk | Status |
|---|---|---|---|---|---|---|---|---|
| Figure 1: End-to-End Experimental Pipeline | main | configs and manuscript Methods 3.1-3.9 | yes, diagram script | EPS/PDF plus editable source if possible | Frozen WavLM layer-9 pipeline used across DAIC-WOZ and Androids Corpus experiments. | yes | low; schematic only | needs_formatting |
| Figure 2: Phase 1-2E Experimental Progression | main | experiment registry and evidence map | yes, timeline script | EPS/PDF plus editable source if possible | Sequential experiment governance from DAIC-WOZ representation development through Android cross-task transfer. | yes | low; no participant data | needs_formatting |
| Figure 3: Zero-Shot Versus Adapted Android Interview Performance | main | `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv` | yes, plotting script | EPS/PDF and source CSV | DAIC-to-Android interview domain-shift diagnosis. | yes | low; aggregate metrics only | needs_formatting |
| Figure 4: Reading, Interview and Cross-Task ROC-AUC Comparison | main | `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv`; transfer gaps | yes, plotting script | EPS/PDF and source CSV | Participant-coherent Android reading/interview transfer. | yes | low; aggregate metrics only | needs_formatting |
| Figure 5: Domain-Shift Feature Summary | main | Phase 2C and 2D reports | yes, plotting script | EPS/PDF and source summaries | Feature-shift summaries for frozen WavLM layer-9 participant embeddings. | yes | low; aggregate shift statistics only | needs_formatting |
| Figure S1: Classifier-Boundary Similarity Analysis | supplementary | `records/runs/phase2e_20260704_151323/phase2e_classifier_boundary_analysis.csv` | yes, plotting script | EPS/PDF and source CSV | Classifier-boundary similarity across participant-coherent folds. | yes | low; fold-level aggregate metrics only | needs_formatting |
| Figure S2: Participant-Coherent Fold Design | supplementary | Phase 2E protocol/report | yes, schematic script | EPS/PDF plus editable source | Participant-coherent fold design for Android reading/interview transfer. | yes | low if schematic placeholders only | needs_formatting |

## BSPC Suitability Notes

- Keep five main figures unless current final formatting checks show stricter limits.
- Move optional classifier-boundary and fold-design details to supplement.
- Do not include raw audio, embeddings, participant predictions, fitted models or real participant identifiers.
- Do not use generative AI to create final submitted artwork.
