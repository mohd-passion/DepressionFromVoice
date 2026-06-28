# GitHub Research Workflow

Use this workflow for every future experiment.

## Before An Experiment

1. Create or confirm a GitHub issue.
2. Record the hypothesis and frozen protocol.
3. Start from updated `main`.
4. Create an experiment branch.

Recommended branch patterns:

```text
exp/phase2c-domain-shift
exp/android-reading-zero-shot
exp/android-interview-baseline
fix/android-fold-parsing
docs/experiment-registry
```

## During The Experiment

1. Commit protocol and configuration first.
2. Commit implementation separately.
3. Run tests.
4. Execute the experiment locally or on the GPU machine.
5. Do not upload datasets, embeddings, model binaries, or participant prediction files.

## After The Experiment

1. Save structured reports.
2. Update the experiment registry.
3. Copy compact reports into `records/runs/`.
4. Run registry validation.
5. Run repository tests.
6. Commit results and documentation.
7. Push the experiment branch.
8. Open a pull request.
9. Merge only after checks pass.
10. Tag major frozen milestones.

## Required Commands

```powershell
python outputs\experiment_registry\update_registry.py --run-dir <new_run_directory>
python outputs\experiment_registry\update_registry.py --validate
python -m unittest discover -s tests
```

## Test-Set Discipline

Do not pursue desired accuracy by repeatedly evaluating the same test set. Use training and validation data for model development, freeze decisions before test evaluation, record every inspected test result, document decisions made after viewing test performance, and judge improvements using cross-validation stability, confidence intervals, and external validation.
