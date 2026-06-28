# DepressionFromVoice

Research code and compact experiment records for cross-corpus depression-from-voice experiments using DAIC-WOZ and Androids Corpus speech.

## Project Objective

This project studies whether participant speech representations, especially frozen WavLM embeddings and compact timing features, carry useful depression-related signal across datasets.

## Current Research Scope

- Phase 1 DAIC-WOZ baseline development is frozen and closed.
- Phase 2A Androids Corpus preparation and compatibility audit is complete.
- Phase 2B frozen DAIC-to-Android zero-shot interview validation is complete.
- The next planned experiment is Phase 2C: Android adaptation analysis to separate calibration shift, classifier-boundary shift, and representation limitations.

## Datasets

Datasets are not included in this repository.

- DAIC-WOZ: English clinical-interview data with PHQ8_Binary labels.
- Androids Corpus: Italian reading and interview data with clinical patient/control grouping.

Users must obtain all datasets through authorized sources and comply with the original dataset licenses, privacy terms, and access restrictions.

## Privacy And Licensing Warning

Do not commit, upload, or redistribute raw audio, transcripts, participant-level predictions, embeddings, fitted models, downloaded corpora, extracted archives, or local dependency caches. The repository is configured to track only source code, configuration, documentation, compact run records, and the experiment registry.

## Repository Structure

```text
src/                         Research scripts
configs/                     Project-relative configuration
tests/                       Lightweight repository validation tests
docs/                        GitHub workflow and safety documentation
records/runs/                Compact archival records for official runs
outputs/experiment_registry/ Compact experiment registry and generated log
.github/                     Issue templates, PR template, CI workflow
```

## Completed Phases

- Phase 1A: metadata-only and final-layer WavLM baselines.
- Phase 1B: frozen WavLM layer-wise analysis; selected layer 9.
- Phase 1C: participant aggregation analysis; retained mean aggregation.
- Phase 1D: timing-feature and early-fusion analysis; selected timing-only for guided interview.
- Phase 1E: probability-level late fusion; fusion did not improve reliability.
- Phase 1F: final frozen DAIC candidates evaluated on provisional local labeled test.
- Phase 2A: Androids Corpus preparation and compatibility audit.
- Phase 2B: frozen DAIC WavLM zero-shot external validation on Android interview speech.

## Current Results Summary

- Frozen DAIC free-speech candidate: WavLM base-plus layer 9, mean aggregation, logistic regression C=0.001, DAIC dev threshold 0.555840134621.
- Frozen DAIC guided-interview candidate: 28 timing features, logistic regression C=100, Phase 1E threshold about 0.292728.
- Phase 2B Android interview zero-shot WavLM: ROC-AUC 0.6412, PR-AUC 0.6809. The ranking signal transferred weakly, but the frozen DAIC threshold failed: all Android participants were predicted negative.

See [outputs/experiment_registry/EXPERIMENT_LOG.md](outputs/experiment_registry/EXPERIMENT_LOG.md) for the generated chronological record.

## Reproducing Experiments

The repository contains scripts and compact records, not datasets or generated embeddings. To reproduce experiments:

1. Obtain DAIC-WOZ and Androids Corpus through authorized channels.
2. Restore the expected local data layout described in the configuration and phase reports.
3. Install required research dependencies locally.
4. Run only the intended phase script for the planned experiment.
5. Save structured outputs locally, then update the experiment registry.

Do not rerun frozen phases just to change historical results.

## Updating The Experiment Registry

```powershell
python outputs\experiment_registry\update_registry.py --run-dir <new_run_directory>
python outputs\experiment_registry\update_registry.py --validate
```

For a full official-run refresh:

```powershell
python outputs\experiment_registry\update_registry.py --scan
```

## Starting A New Experiment Branch

```powershell
git checkout main
git pull
git checkout -b exp/phase2c-domain-shift
```

Record the hypothesis and frozen protocol in a GitHub issue before running the experiment. Commit protocol/configuration changes before implementation and results.

## Clinical Safety Disclaimer

This project is research-only. It is not a diagnostic tool, not clinically validated, and must not be used to make medical or mental-health decisions.
