# BSPC Figure Package

Stage 3E generates a BSPC-oriented figure package from compact, non-sensitive project records only.

## Directories

- `source/`: aggregate source-value manifest generated from compact records.
- `output/`: SVG and PNG figure outputs plus the generation log.
- `scripts/`: reproducible figure-generation script.

## Generation Command

Run from the repository root:

```powershell
python paper\figures\scripts\generate_bspc_figures.py
```

The script writes:

- `paper/figures/output/figure1_study_design_protocol_overview.svg`
- `paper/figures/output/figure1_study_design_protocol_overview.png`
- `paper/figures/output/figure2_daic_phase1_key_results.svg`
- `paper/figures/output/figure2_daic_phase1_key_results.png`
- `paper/figures/output/figure3_daic_android_transfer_adaptation.svg`
- `paper/figures/output/figure3_daic_android_transfer_adaptation.png`
- `paper/figures/output/figure4_android_cross_task_results.svg`
- `paper/figures/output/figure4_android_cross_task_results.png`
- `paper/figures/output/figure5_domain_shift_boundary_summary.svg`
- `paper/figures/output/figure5_domain_shift_boundary_summary.png`
- `paper/figures/source/bspc_figure_source_values.csv`
- `paper/BSPC_FIGURE_CAPTIONS.md`
- `paper/BSPC_FIGURE_DATA_AUDIT.csv`

## Privacy And Scope

The script reads only manuscript planning files, registry summaries, and compact aggregate run records. It does not read raw audio, participant identifiers, embeddings, fitted models, prediction files, or generated participant-level outputs.

The figures are marked as awaiting author review. They are not submission-ready until the authors confirm figure content, captions, consent wording, acknowledgement wording, and the required AI-assisted writing disclosure.
