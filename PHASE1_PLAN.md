# Phase 1 Plan: DAIC-WOZ Audio-Only Depression Detection

## Scope

Phase 1 focuses only on automatic depression detection from DAIC-WOZ audio using frozen WavLM embeddings.

This phase does not use:

- Text transcripts as model input
- Visual or facial features
- MFCC features
- COVAREP features
- FORMANT features
- Android Corpus
- App or user-interface code

The goal is to build a clean, reproducible research baseline before expanding to other corpora, feature families, or model architectures.

Transcript use is allowed only for timing and speaker segmentation. DAIC-WOZ transcript timestamps and speaker labels may be used to keep participant speech and remove interviewer speech, but transcript text must never be used as model input or as a derived semantic feature.

## Objective

Develop a participant-level binary depression detection pipeline using DAIC-WOZ raw audio.

The central research question for Phase 1 is:

Can frozen WavLM speech embeddings extracted from fixed-length DAIC-WOZ audio windows detect participant-level depression labels with reliable validation performance?

## Dataset

Dataset:

- DAIC-WOZ / AVEC depression dataset

Local path:

- `Data/DIAC_WOZ`

Inputs:

- `{participant_id}_AUDIO.wav`
- `{participant_id}_TRANSCRIPT.csv`, timing and speaker labels only

Labels:

- `PHQ8_Binary`

Split metadata:

- `train_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `test_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

Primary prediction target:

- Binary depression classification at participant level

## Unified Manifest

Create one canonical DAIC-WOZ Phase 1 manifest.

Recommended manifest columns:

- `dataset`
- `participant_id`
- `audio_path`
- `transcript_path`
- `label_binary`
- `phq8_score`
- `official_split`
- `gender`
- `audio_duration_sec`
- `participant_speech_duration_sec`
- `sample_rate`
- `num_channels`
- `window_sec`
- `overlap`
- `num_windows`

Manifest rules:

- One row per participant for the participant-level manifest.
- A second derived window-level manifest may be created later, but participant identity must remain attached to every window.
- The manifest is the source of truth for labels, splits, audio paths, and grouping.
- Missing labels must be represented explicitly.
- Test rows without labels must not be used for threshold tuning or supervised evaluation.
- Transcript text must not be copied into the manifest.
- Transcript-derived fields are limited to timing and speaker-segmentation metadata.

## Participant-Level Labels

Use participant-level labels only.

Label source:

- `PHQ8_Binary`

Interpretation:

- `0`: non-depressed
- `1`: depressed

Important constraint:

- Window-level samples inherit participant labels only for representation learning and classifier training. Final evaluation must be participant-level.

## Label Quality Check

Run a label quality check before any model training.

Requirements:

- Verify that each participant ID appears only once in each label file.
- Verify that `PHQ8_Binary` matches `PHQ8_Score` using the DAIC-WOZ depression threshold.
- Flag participants where `PHQ8_Binary` and `PHQ8_Score` disagree.
- Verify that train/dev/test split labels are not duplicated or conflicting.
- Check whether any participant appears in more than one canonical split.
- Create `label_quality_report.csv`.
- Do not automatically correct labels; only flag suspicious cases for manual review.

Default DAIC-WOZ depression threshold:

- Depressed if `PHQ8_Score >= 10`

## Audio Windowing

Audio should first be restricted to participant speech, then segmented into fixed windows before embedding extraction.

Participant speech extraction:

- Use DAIC-WOZ transcript `start_time`, `stop_time`, and `speaker` columns.
- Keep only rows where `speaker` is `Participant`.
- Use the timestamps to extract participant speech intervals from the raw audio.
- Concatenate participant speech intervals or window over participant-only intervals using a documented deterministic strategy.
- Do not use transcript `value` text.
- Do not derive linguistic, lexical, sentiment, or semantic features from transcripts.

Windowing configuration:

- Window length: 10 seconds
- Overlap: 50%
- Hop length: 5 seconds

Rules:

- Keep every window linked to its original participant.
- Do not allow windows from the same participant to appear in multiple splits.
- Handle final partial windows consistently.
- Recommended default: drop windows shorter than 10 seconds unless a padding strategy is explicitly documented.
- Store window start and end times in the window manifest.
- Store whether window times refer to original audio time or participant-speech-concatenated time.

Recommended window-level manifest columns:

- `dataset`
- `participant_id`
- `window_id`
- `audio_path`
- `source_start_sec`
- `source_end_sec`
- `start_sec`
- `end_sec`
- `duration_sec`
- `label_binary`
- `official_split`
- `speaker_source`

## WavLM Frozen Embeddings

Use WavLM as a frozen speech representation extractor.

Model family:

- WavLM

Initial recommended model:

- `microsoft/wavlm-base-plus`

Embedding strategy:

- Load pretrained WavLM.
- Freeze all WavLM parameters.
- Pass each 10-second audio window through WavLM.
- Extract fixed-dimensional window embeddings.
- Save embeddings for reproducible downstream experiments.

Pooling options for each 10-second window:

- Mean pooling over time from the selected hidden layer
- Optional comparison of selected hidden layers after the first baseline is stable

Initial layer strategy:

- Use the final hidden layer for the first baseline.
- Add layer-wise analysis only after the Phase 1 baseline is complete.

Audio preparation for WavLM:

- Resample to 16 kHz.
- Convert to mono.
- Normalize consistently.

## Participant-Level Aggregation

The model must produce participant-level predictions.

Recommended baseline aggregation:

1. Extract one embedding per 10-second window.
2. Aggregate all window embeddings for each participant into one participant-level embedding using mean pooling.
3. Train logistic regression at participant level using participant-level labels.
4. Generate one probability per participant.

Initial aggregation method:

- Mean pooling over all window embeddings for each participant

Secondary comparison after the first baseline is complete:

- Train a window-level classifier using participant labels inherited by windows.
- Generate probabilities for all windows.
- Aggregate window probabilities per participant using mean probability.

Additional aggregation methods for later Phase 1 comparison:

- Median window probability
- Max window probability
- Attention pooling, only after simple aggregation baselines are complete

Primary output:

- One probability score per participant

## Metadata Sanity Baseline

Before reporting WavLM results, run a metadata-only sanity baseline.

Allowed metadata features:

- Participant speech duration
- Number of 10-second windows
- Gender

Purpose:

- Check whether simple duration or demographic artifacts explain performance.
- Provide a non-acoustic reference point for the WavLM baseline.
- Detect possible confounding before claiming voice-representation performance.

Restrictions:

- Do not add age, PHQ-8 item scores, transcript text, visual features, COVAREP, FORMANT, MFCC, or Android Corpus data.
- Report this baseline separately from the WavLM audio baseline.

## Group-Safe Train/Dev/Test Split

Splits must be participant-safe.

Rules:

- The participant is the grouping unit.
- No participant can appear in more than one split.
- All windows from one participant must remain in the same split.
- Threshold tuning must use only the dev set.
- Test set evaluation must use the threshold selected on dev.

Recommended starting protocol:

- Use DAIC-WOZ official train split for training.
- Use DAIC-WOZ official dev split for threshold tuning and validation.
- Use labeled test data only if labels are available and appropriate for supervised reporting.

If test labels are unavailable:

- Report train/dev performance only.
- Optionally create a group-safe held-out split from the labeled train/dev participants, but document it separately from official AVEC splits.

## Classifier Baseline

Use lightweight classifiers on frozen WavLM embeddings.

Recommended first classifier:

- Logistic regression with `class_weight="balanced"`

Rationale:

- Simple
- Reproducible
- Strong baseline for frozen embeddings
- Easier to interpret and tune
- Handles DAIC-WOZ class imbalance without resampling in the first baseline

Additional classifiers after the first baseline:

- Small MLP classifier

Deferred until later:

- Probability calibration
- Linear SVM with calibrated probabilities

Do not fine-tune WavLM in Phase 1 baseline.

## Threshold Tuning

The classifier should output participant-level probabilities.

Threshold selection:

- Tune threshold on the dev set only.
- Choose the threshold that maximizes macro F1 or balanced accuracy.
- Report which threshold objective was used.
- Freeze the selected threshold before test evaluation.

Recommended default:

- Tune threshold for maximum macro F1 on dev.

Also report:

- Metrics at default threshold `0.5`
- Metrics at tuned threshold

## Metrics

Report participant-level metrics.

Required metrics:

- AUC
- Balanced accuracy
- Macro F1
- Recall
- Precision

Reporting rules:

- Report metrics on dev using the tuned threshold.
- Report AUC independently of threshold.
- If test labels are available, apply the dev-tuned threshold to test without retuning.
- Include confusion matrix counts for each evaluated split.
- For every split, report participant count, class counts, window count, and average windows per participant.

Required split summary:

| Split | Participants | Non-depressed | Depressed | Windows | Avg windows/participant |
|---|---:|---:|---:|---:|---:|
| Train | TBD | TBD | TBD | TBD | TBD |
| Dev | TBD | TBD | TBD | TBD | TBD |
| Test, if labeled | TBD | TBD | TBD | TBD | TBD |

## Reproducibility Requirements

Every run should record:

- Dataset path
- Manifest version
- Participant IDs in each split
- WavLM model name
- WavLM layer used
- Window length and overlap
- Transcript segmentation policy
- Split-level participant, class, and window counts
- Audio preprocessing settings
- Classifier type and hyperparameters
- Random seed
- Threshold tuning objective
- Selected threshold
- Evaluation metrics

Recommended fixed defaults:

- Window length: 10 seconds
- Overlap: 50%
- Hop length: 5 seconds
- Audio sample rate: 16 kHz
- WavLM frozen: yes
- Initial classifier: logistic regression with `class_weight="balanced"`
- Threshold tuning split: dev
- Probability calibration: not used in first baseline

## Expected Phase 1 Outputs

Phase 1 should produce:

- DAIC-WOZ participant-level manifest
- DAIC-WOZ window-level manifest
- Split summary with participant counts, class counts, window counts, and average windows per participant
- Metadata sanity baseline using duration, number of windows, and gender
- Frozen WavLM embeddings for each 10-second window
- Participant-level prediction file
- Dev-set threshold tuning report
- Evaluation report with AUC, balanced accuracy, macro F1, recall, and precision
- Reproducibility metadata for each experiment run

## Success Criteria

Phase 1 is complete when:

- DAIC-WOZ audio files are indexed in a unified manifest.
- Participant-level labels are correctly attached.
- Transcript timestamps and speaker labels are used only to keep participant speech.
- 10-second windows with 50% overlap are generated without participant leakage.
- The metadata sanity baseline is reported separately.
- Frozen WavLM embeddings are extracted for all usable windows.
- Participant-level probabilities are produced through aggregation.
- Threshold tuning is performed only on the dev set.
- Required metrics are reported at participant level.
- Split-level participant counts, class counts, window counts, and average windows per participant are reported.
- The full experiment can be rerun from documented configuration and manifests.

## Implementation Task List

No code should be written until these tasks are approved.

### Task 1: Confirm Phase 1 Data Contract

- Confirm DAIC-WOZ root path.
- Confirm label source files.
- Confirm use of official train and dev splits.
- Confirm whether labeled test rows in `full_test_split.csv` should be used for final reporting.
- Document that transcript text is prohibited as model input.

### Task 2: Define Manifest Schemas

- Define participant-level manifest fields.
- Define window-level manifest fields.
- Define split-summary report fields.
- Define experiment metadata fields.
- Decide where generated manifests and reports will be stored.

### Task 3: Build Participant-Level Index

- Index DAIC-WOZ participant folders.
- Link each participant to audio path.
- Link each participant to transcript path.
- Attach `PHQ8_Binary`, `PHQ8_Score`, gender, and official split.
- Validate missing labels and missing files.

### Task 4: Run Label Quality Check

- Verify duplicate participant IDs within each label file.
- Verify `PHQ8_Binary` against `PHQ8_Score` using the DAIC-WOZ threshold.
- Flag binary/score disagreements.
- Verify duplicated or conflicting labels across split files.
- Verify that no participant appears in more than one canonical split.
- Write `label_quality_report.csv`.
- Do not automatically correct suspicious labels.

### Task 5: Extract Participant Speech Timing

- Read transcript timing and speaker columns.
- Keep only `Participant` intervals.
- Ignore transcript text.
- Compute participant speech duration per participant.
- Validate participants with zero or very short participant speech.

### Task 6: Generate 10-Second Windows

- Apply 10-second windows with 50% overlap.
- Use 5-second hop length.
- Drop or document short final windows.
- Keep participant ID on every window.
- Produce window counts per participant and split.

### Task 7: Produce Split Summary

- Report participant counts per split.
- Report class counts per split.
- Report total window counts per split.
- Report average windows per participant per split.
- Save the summary with the experiment artifacts.

### Task 8: Extract Frozen WavLM Embeddings

- Resample audio to 16 kHz mono.
- Load WavLM in frozen mode.
- Extract one embedding per valid 10-second participant-speech window.
- Record model name, layer, pooling, and preprocessing choices.
- Save embeddings with stable IDs linked to the window manifest.

### Task 9: Run Metadata Sanity Baseline

- Use only participant speech duration, number of windows, and gender.
- Train logistic regression with `class_weight="balanced"`.
- Tune threshold on dev.
- Report the required participant-level metrics.
- Keep this report separate from WavLM results.

### Task 10: Train WavLM Embedding Baseline

- Aggregate all window embeddings for each participant using mean pooling.
- Train logistic regression with `class_weight="balanced"` on participant-level mean embeddings.
- Predict participant-level probabilities directly.
- Do not use probability calibration in the first baseline.
- Implement window-level classification with mean probability aggregation only as a secondary comparison after this baseline is complete.

### Task 11: Tune Threshold on Dev

- Evaluate candidate thresholds on dev only.
- Select the threshold using the documented objective, initially macro F1.
- Report metrics at threshold `0.5` and at the tuned threshold.
- Freeze the selected threshold for any final test evaluation.

### Task 12: Evaluate and Report

- Report AUC, balanced accuracy, macro F1, recall, and precision at participant level.
- Include confusion matrix counts.
- Include split-level participant, class, and window summaries.
- Include reproducibility metadata.
- Compare metadata sanity baseline against WavLM frozen embedding baseline.

### Task 13: Review Before Expansion

- Check for participant leakage.
- Check whether metadata baseline is unexpectedly strong.
- Check threshold sensitivity.
- Check class imbalance effects.
- Decide whether Phase 1 should next add layer-wise WavLM analysis or remain on the first frozen baseline.
