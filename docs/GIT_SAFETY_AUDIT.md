# Git Safety Audit

Generated: `2026-06-28T02:17:13.160222+00:00`

This audit is repository-safety documentation only. No files were deleted, staged, committed, pushed, or moved.

## Summary

| Check | Result |
|---|---:|
| Files scanned | 143555 |
| Restricted/generated local files detected | 143436 |
| Files larger than 25 MB | 925 |
| Files larger than 100 MB | 205 |
| Restricted files already tracked before initial commits | 0 |
| Unignored staging candidates before selective staging | 109 |
| Restricted unignored staging candidates | 0 |
| High-confidence possible secrets in intended trackable text | 0 |
| Absolute Windows paths in intended trackable text | 0 |

## Restricted Local Content Found

|category|count|examples|
|---|---:|---|
|generated_output|248|outputs\phase1_daic\daic_phase1_participant_manifest_clean.csv; outputs\phase1_daic\daic_phase1_split_summary_clean.csv; outputs\phase1_daic\daic_phase1_window_manifest_clean.csv|
|generated_output+participant_prediction_file|14|outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\layerwise_outer_predictions.csv; outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\selected_layer_dev_predictions.csv; outputs\phase1_daic\phase1c_aggregation_runs\20260622_135800\aggregation_outer_predictions.csv|
|generated_output+restricted_suffix:.joblib|6|outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\models\selected_layer_logistic_regression_pipeline.joblib; outputs\phase1_daic\phase1c_aggregation_runs\20260622_135800\models\selected_aggregation_pipeline.joblib; outputs\phase1_daic\phase1d_timing_fusion_runs\20260622_142132\models\selected_phase1d_pipeline.joblib|
|generated_output+restricted_suffix:.npy|52867|outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\participant_layer_embeddings\300.npy; outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\participant_layer_embeddings\301.npy; outputs\phase1_daic\phase1b_layerwise_runs\20260622_123725\participant_layer_embeddings\302.npy|
|restricted_directory|34425|.phase1_deps\isympy.py; .phase1_deps\scipy-1.18.0-cp312-cp312-win_amd64.whl; .phase1_deps\threadpoolctl.py|
|restricted_directory+restricted_suffix:.npy|31|.phase1_deps\joblib\test\data\joblib_0.9.2_pickle_py27_np16.pkl_01.npy; .phase1_deps\joblib\test\data\joblib_0.9.2_pickle_py27_np16.pkl_02.npy; .phase1_deps\joblib\test\data\joblib_0.9.2_pickle_py27_np16.pkl_03.npy|
|restricted_directory+restricted_suffix:.npz|20|.phase1_deps\numpy\lib\tests\data\py2-objarr.npz; .phase1_deps\numpy\lib\tests\data\py3-objarr.npz; .phase1_deps\scipy\stats\_sobol_direction_numbers.npz|
|restricted_directory+restricted_suffix:.pkl|11|.phase1_deps\joblib\test\data\joblib_0.10.0_pickle_py27_np17.pkl; .phase1_deps\joblib\test\data\joblib_0.10.0_pickle_py33_np18.pkl; .phase1_deps\joblib\test\data\joblib_0.10.0_pickle_py34_np19.pkl|
|restricted_directory+restricted_suffix:.pth|1|.phase1_deps\distutils-precedence.pth|
|restricted_directory+restricted_suffix:.wav|55813|.phase1_deps\scipy\io\tests\data\test-1234Hz-le-1ch-10S-20bit-extra.wav; .phase1_deps\scipy\io\tests\data\test-44100Hz-2ch-32bit-float-be.wav; .phase1_deps\scipy\io\tests\data\test-44100Hz-2ch-32bit-float-le.wav|

## Restricted Files Already Tracked

Git is initialized at the project root on branch `main`. Before the first commit, `git ls-files` reported no tracked files. A pre-staging scan of `git ls-files --others --exclude-standard` found 109 unignored candidates and 0 restricted candidates.

If future `git status` shows restricted files tracked, do not rewrite history automatically. Use ordinary removal from the index first, for example:

```powershell
git rm --cached -r Data Android-Corpus outputs .phase1_deps
```

Only consider history rewriting after a separate, explicit approval and backup.

## Files Larger Than 25 MB: Top 20

|path|size_mb|
|---|---:|
|Data\380_P\380_CLNF_hog.bin|1005.4|
|Data\337_P\337_CLNF_hog.bin|956.8|
|Data\364_P\364_CLNF_hog.bin|911.0|
|Data\305_P\305_CLNF_hog.bin|871.3|
|Data\367_P\367_CLNF_hog.bin|837.1|
|Data\483_P\483_CLNF_hog.bin|806.4|
|Data\405_P\405_CLNF_hog.bin|802.9|
|Data\466_P\466_CLNF_hog.bin|801.0|
|Data\314_P\314_CLNF_hog.bin|798.9|
|Data\372_P\372_CLNF_hog.bin|774.1|
|.phase1_deps\torch\lib\torch_cuda.dll|774.0|
|Data\438_P\438_CLNF_hog.bin|731.2|
|Data\440_P\440_CLNF_hog.bin|723.1|
|Data\365_P\365_CLNF_hog.bin|708.8|
|Data\411_P\411_CLNF_hog.bin|707.2|
|Data\383_P\383_CLNF_hog.bin|700.1|
|Data\390_P\390_CLNF_hog.bin|693.7|
|Data\368_P\368_CLNF_hog.bin|692.8|
|Data\377_P\377_CLNF_hog.bin|679.2|
|Data\434_P\434_CLNF_hog.bin|674.9|

## Files Larger Than 100 MB: Top 20

|path|size_mb|
|---|---:|
|Data\380_P\380_CLNF_hog.bin|1005.4|
|Data\337_P\337_CLNF_hog.bin|956.8|
|Data\364_P\364_CLNF_hog.bin|911.0|
|Data\305_P\305_CLNF_hog.bin|871.3|
|Data\367_P\367_CLNF_hog.bin|837.1|
|Data\483_P\483_CLNF_hog.bin|806.4|
|Data\405_P\405_CLNF_hog.bin|802.9|
|Data\466_P\466_CLNF_hog.bin|801.0|
|Data\314_P\314_CLNF_hog.bin|798.9|
|Data\372_P\372_CLNF_hog.bin|774.1|
|.phase1_deps\torch\lib\torch_cuda.dll|774.0|
|Data\438_P\438_CLNF_hog.bin|731.2|
|Data\440_P\440_CLNF_hog.bin|723.1|
|Data\365_P\365_CLNF_hog.bin|708.8|
|Data\411_P\411_CLNF_hog.bin|707.2|
|Data\383_P\383_CLNF_hog.bin|700.1|
|Data\390_P\390_CLNF_hog.bin|693.7|
|Data\368_P\368_CLNF_hog.bin|692.8|
|Data\377_P\377_CLNF_hog.bin|679.2|
|Data\434_P\434_CLNF_hog.bin|674.9|

## Possible Secrets

|file|match|
|---|---|
|none||

The scan looked for assignment-like secret patterns in intended trackable text files. It did not inspect binary datasets or dependency packages as source code for commit.

## Absolute Windows Paths In Intended Trackable Text

|file|context|
|---|---|
|none||

Project documentation, compact records, source, and configuration intended for Git should use project-relative paths. Absolute local paths may still exist inside ignored generated outputs and local reports; those are not intended for Git tracking.

## Recommended Corrective Actions

1. Keep `.gitignore` active before staging anything.
2. Do not stage `Data/`, `Android-Corpus/`, `Papers/`, `.phase1_deps/`, generated outputs, embeddings, models, predictions, or archives.
3. Track only source, configs, tests, docs, compact run records, and `outputs/experiment_registry/`.
4. Install Git and initialize only at the project root if needed.
5. Run `python -m unittest discover -s tests` and `python outputs/experiment_registry/update_registry.py --validate` before committing.
6. Review `git status --short` manually before the first commit.
