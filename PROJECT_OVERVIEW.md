# Project Overview: Voice-Based Depression Detection

## Research Objective

This project aims to develop and evaluate voice-feature-based methods for depression detection using clinical and interview-style speech corpora. The primary research goal is to determine whether acoustic, prosodic, spectral, and self-supervised speech representations can distinguish depressed participants from non-depressed or healthy control participants.

The current scope is voice features only. Although some datasets contain transcripts, facial features, video-derived signals, and metadata, these modalities should be treated as supporting context or optional future extensions rather than inputs for the initial modeling pipeline.

## Datasets

### DAIC-WOZ / AVEC Depression Dataset

Location: `Data/DIAC_WOZ`

DAIC-WOZ contains participant-level interview recordings with depression labels derived from PHQ-8 scores. The local copy includes participant folders named with identifiers such as `301_P`, `302_P`, and so on. Each participant folder generally contains:

- Raw interview audio: `{id}_AUDIO.wav`
- Transcript: `{id}_TRANSCRIPT.csv`
- Acoustic feature files: `{id}_COVAREP.csv`, `{id}_FORMANT.csv`
- Visual feature files from CLNF/OpenFace-style extraction

The label and split files are:

- `train_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `test_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

For the voice-only experiments, the most important DAIC-WOZ files are the participant audio recordings, COVAREP features, FORMANT features, and PHQ-8 labels.

### Androids Corpus

Location: `Data/Android_corpus/Androids-Corpus/Androids-Corpus`

The Androids Corpus contains audio recordings from healthy controls and depression patients. It is organized into two speech tasks:

- `Reading-Task`: participants read a fixed passage.
- `Interview-Task`: participants answer interview prompts.

The corpus includes:

- Full interview audio in `Interview-Task/audio/HC` and `Interview-Task/audio/PT`
- Segmented interview clips in `Interview-Task/audio_clip`
- Reading-task audio in `Reading-Task/audio/HC` and `Reading-Task/audio/PT`
- `fold-lists.csv` for predefined 5-fold evaluation
- `interview_timedata.csv` for interview turn segmentation
- `Androids.conf`, an openSMILE configuration file

Android filename format follows:

`nn_XGmm_t.wav`

where `X` indicates condition, `C` means healthy control, `P` means depression patient, `G` indicates gender, `mm` indicates age, and `t` indicates education level.

## Data Modalities

### In Scope: Voice Features

The initial experiments should use only speech/audio-derived information:

- Raw waveform audio (`.wav`)
- Low-level acoustic descriptors
- Prosodic features such as pitch, intensity, rhythm, speaking rate, and pause behavior
- Spectral features such as MFCCs, formants, energy, and voice quality
- COVAREP features from DAIC-WOZ
- FORMANT features from DAIC-WOZ
- openSMILE features, especially using the Androids `Androids.conf` configuration
- Self-supervised speech embeddings such as WavLM, wav2vec 2.0, HuBERT, or related models

### Out of Scope for Initial Voice-Only Experiments

These modalities are available or partially available but should not be used as model inputs for the first voice-only pipeline:

- DAIC-WOZ transcripts
- DAIC-WOZ facial pose, gaze, action units, landmarks, and HOG features
- Any video-derived features
- Demographic metadata such as age, gender, or education, except for analysis, balancing, or reporting

## Expected Outputs

The project should produce:

- A unified audio manifest covering DAIC-WOZ and Androids Corpus
- Clean subject-level labels for depression/control classification
- Preprocessed audio files or references to normalized audio
- Extracted voice feature tables for each participant, session, task, and segment
- Baseline classification results for each dataset
- Cross-dataset or pooled evaluation results where feasible
- Model comparison tables for handcrafted features versus self-supervised speech embeddings
- Error analysis showing where models fail across task type, corpus, gender, age, or recording condition

Possible prediction targets:

- Binary depression detection: depressed vs non-depressed/control
- DAIC-WOZ PHQ-8 score regression, if using DAIC-WOZ alone
- Severity grouping from PHQ-8 scores, if clinically justified and class sizes allow

## Proposed Experiments

### 1. Audio-Only Baseline With Handcrafted Features

Extract conventional acoustic features from all audio files and train classical machine learning models.

Candidate features:

- MFCC statistics
- Pitch and pitch variability
- Energy and loudness
- Speaking rate
- Pause duration and pause frequency
- Jitter, shimmer, and harmonics-to-noise ratio
- Formant statistics
- COVAREP and openSMILE features

Candidate models:

- Logistic regression
- Support vector machine
- Random forest
- Gradient boosting
- XGBoost or LightGBM, if available

### 2. DAIC-WOZ Voice-Only PHQ-8 Classification

Use DAIC-WOZ train and development splits for binary depression classification. Keep evaluation subject-independent. Use only audio-derived features and exclude transcript and visual features.

Recommended comparisons:

- COVAREP only
- FORMANT only
- openSMILE/eGeMAPS or IS09 only
- Combined handcrafted acoustic features

### 3. Androids Corpus 5-Fold Voice Classification

Use `fold-lists.csv` to reproduce the Androids Corpus 5-fold setup. Evaluate separately on:

- Reading task
- Full interview task
- Interview clips aggregated to speaker level

The key comparison is whether spontaneous interview speech is more informative than read speech for depression detection.

### 4. Self-Supervised Speech Embedding Experiments

Extract utterance-level or segment-level embeddings from pretrained speech models.

Candidate models:

- WavLM
- wav2vec 2.0
- HuBERT

Evaluation options:

- Frozen embeddings with classical classifiers
- Mean-pooled segment embeddings at speaker level
- Fine-tuning if hardware and data size permit

### 5. Segment-Level Versus Speaker-Level Modeling

Compare models trained on short speech segments against models trained on participant-level aggregated features.

Important constraint:

All train/test splits must be speaker-independent. Segments from the same participant must never appear in both train and test sets.

### 6. Cross-Corpus Generalization

Train on one corpus and evaluate on the other to test whether voice biomarkers generalize across collection conditions.

Suggested settings:

- Train on DAIC-WOZ, test on Androids interview task
- Train on Androids interview task, test on DAIC-WOZ
- Train pooled models with dataset indicators removed from model inputs

This experiment may be difficult because DAIC-WOZ uses PHQ-8 labels while Androids uses patient/control grouping, so results should be interpreted carefully.

### 7. Feature Ablation and Interpretability

Run ablation studies to identify which voice feature groups contribute most.

Suggested groups:

- Prosody
- Spectral features
- Voice quality
- Pauses and timing
- Formants
- Self-supervised embeddings

Report feature importance, permutation importance, or SHAP values for interpretable models.

## Relevant Papers

The `Papers` folder contains relevant literature for this project, including work on:

- DAIC-WOZ and AVEC depression recognition challenges
- Androids Corpus description and baseline methods
- Speech-based depression assessment surveys
- WavLM and self-supervised speech representation learning
- Transformer-based speech depression detection
- Privacy-preserving speaker identity methods
- Voice feature changes associated with depression

These papers should guide feature selection, baseline design, evaluation protocol, and comparison against prior work.

## Recommended First Implementation Step

Create a voice-only manifest file that indexes every usable audio file with:

- Dataset name
- Subject ID
- Task name
- Segment ID, if applicable
- Audio path
- Label
- Split or fold
- Gender, age, and education when available
- Source label type, such as `PHQ8_Binary` or `HC_PT`

This manifest should become the common input to all later preprocessing, feature extraction, and modeling scripts.
