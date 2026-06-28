# Research Plan: Automatic Depression Detection from Voice

## 1. Research Objectives

The goal of this project is to develop a reproducible, publication-quality framework for automatic depression detection from voice using DAIC-WOZ and the Androids Corpus.

The core research question is:

Can voice-derived acoustic and neural speech representations reliably detect depression across different corpora, speech tasks, recording conditions, and label definitions?

Primary objectives:

- Build a voice-only depression detection pipeline using DAIC-WOZ and Androids Corpus.
- Compare handcrafted acoustic features, deep learning baselines, and transformer-based speech representations.
- Evaluate within-corpus performance and cross-corpus generalization.
- Identify which voice feature families are most robust across datasets.
- Establish leakage-safe, speaker-independent protocols suitable for publication.
- Produce reusable manifests, configuration files, experiment logs, and model evaluation reports.

Secondary objectives:

- Compare interview speech against read speech where Androids Corpus permits this analysis.
- Compare full-recording, turn-level, and segment-level aggregation strategies.
- Analyze model robustness across gender, age, education, task type, and corpus.
- Support future multimodal extensions without mixing non-voice modalities into the initial experiments.

## 2. Dataset Comparison

### DAIC-WOZ / AVEC

Local path: `Data/DIAC_WOZ`

DAIC-WOZ is an interview-based clinical depression dataset. The local copy contains participant folders such as `301_P`, `302_P`, and so on. Each participant folder generally contains raw audio, transcripts, acoustic features, and video-derived facial features.

For this project, only voice-derived data should be used:

- Raw audio: `{id}_AUDIO.wav`
- COVAREP acoustic features: `{id}_COVAREP.csv`
- Formant features: `{id}_FORMANT.csv`
- Optional timing information from transcripts, used only to isolate participant speech if needed

Label files:

- `train_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `test_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

Relevant labels:

- `PHQ8_Binary`
- `PHQ8_Score`
- PHQ-8 item-level scores
- Gender

Expected use:

- Binary depression classification
- PHQ-8 regression or severity prediction in DAIC-only experiments
- Official train/dev/test-style evaluation where appropriate

Important considerations:

- DAIC-WOZ has rich non-audio modalities, but these must be excluded from voice-only modeling.
- Transcripts should not be used as semantic text input in the initial experiments.
- If transcripts are used for speech segmentation, this must be reported clearly as timing-only use.
- One participant, `300_P`, appears to be missing `CLNF_hog.bin`, which is not relevant for voice-only work.

### Androids Corpus

Local path: `Data/Android_corpus/Androids-Corpus/Androids-Corpus`

Androids Corpus is an audio-focused dataset with healthy controls and depression patients. It includes both read speech and interview speech.

Main structure:

- `Reading-Task/audio/HC`: healthy control read speech
- `Reading-Task/audio/PT`: patient read speech
- `Interview-Task/audio/HC`: healthy control interview speech
- `Interview-Task/audio/PT`: patient interview speech
- `Interview-Task/audio_clip`: segmented interview clips
- `fold-lists.csv`: predefined 5-fold evaluation splits
- `interview_timedata.csv`: interview turn timing data
- `Androids.conf`: openSMILE feature extraction configuration

Filename convention:

`nn_XGmm_t.wav`

where:

- `nn`: speaker identifier within group
- `X`: condition, `C` for control and `P` for patient
- `G`: gender, `M` or `F`
- `mm`: age
- `t`: education level, 1 to 4, with `x` when unavailable

Expected use:

- Binary patient/control classification
- Separate read-speech and interview-speech experiments
- Speaker-level aggregation from interview clips
- 5-fold evaluation using `fold-lists.csv`

Important considerations:

- Androids labels are patient/control labels, not PHQ-8 scores.
- Clips must be grouped by speaker to avoid leakage.
- MacOS metadata files such as `__MACOSX`, `.DS_Store`, and `._*` should be ignored.

### Comparison Summary

| Aspect | DAIC-WOZ | Androids Corpus |
|---|---|---|
| Primary task | Clinical interview | Reading and interview |
| Labels | PHQ-8 binary and score | Patient/control |
| Main audio unit | Participant interview | Full recordings and clips |
| Text available | Yes, transcripts | No transcripts found |
| Video-derived features | Yes | No |
| Voice features available | Raw audio, COVAREP, formants | Raw audio, openSMILE config |
| Recommended split | Official AVEC train/dev/test | Provided 5-fold lists |
| Cross-corpus role | Clinical interview benchmark | External speech corpus |

## 3. Data Preprocessing Steps

### 3.1 Manifest Creation

Create a unified voice-only manifest before any modeling. This should be the foundation for all experiments.

Recommended manifest columns:

- `dataset`
- `corpus_version`
- `subject_id`
- `recording_id`
- `task`
- `segment_id`
- `audio_path`
- `feature_path`
- `label_binary`
- `label_source`
- `phq8_score`
- `split`
- `fold`
- `gender`
- `age`
- `education`
- `duration_seconds`
- `sample_rate`
- `channel_count`
- `is_segment`
- `parent_recording_id`

The manifest should support:

- DAIC-WOZ participant-level audio
- Androids full-recording audio
- Androids interview clips
- Future extracted features
- Speaker-independent splits

### 3.2 Audio Standardization

All audio should be standardized before feature extraction.

Recommended format:

- Mono
- 16 kHz sampling rate
- WAV format
- Consistent bit depth
- Loudness or peak normalization

Maintain original audio paths and store normalized audio separately if generated later.

### 3.3 Segmentation

DAIC-WOZ:

- Use full participant audio as the first baseline.
- Optionally use transcript timestamps to isolate participant speech.
- If transcript timestamps are used, exclude transcript text content from model features.

Androids:

- Use full recordings for baseline experiments.
- Use `audio_clip` for segment-level experiments.
- Use `interview_timedata.csv` to verify or reproduce interview segmentation if needed.

### 3.4 Quality Control

Quality checks should include:

- File existence validation
- Duration validation
- Sample rate validation
- Corrupted audio detection
- Silence ratio
- Speech activity ratio
- Clipping detection
- Extremely short segment filtering
- Duplicate speaker leakage checks

### 3.5 Label Harmonization

DAIC-WOZ:

- Use `PHQ8_Binary` for binary classification.
- Use `PHQ8_Score` for regression or severity analysis.

Androids:

- Map `PT` to depressed/patient.
- Map `HC` to control.

Cross-corpus:

- Use binary labels only.
- Clearly report that DAIC labels are PHQ-8-derived while Androids labels are patient/control status.

## 4. Audio Feature Extraction Strategy

The project should compare handcrafted, task-specific, and neural speech representations.

### 4.1 Handcrafted Acoustic Features

Core feature groups:

- MFCCs and delta MFCCs
- Pitch statistics
- Energy and intensity
- Zero-crossing rate
- Spectral centroid, flux, bandwidth, rolloff
- Jitter and shimmer
- Harmonics-to-noise ratio
- Formant frequencies and bandwidths
- Speaking rate estimates
- Pause duration and pause frequency
- Voiced/unvoiced ratio

Aggregation statistics:

- Mean
- Standard deviation
- Median
- Minimum and maximum
- Interquartile range
- Percentiles
- Skewness and kurtosis
- Temporal slope

### 4.2 DAIC-WOZ Existing Features

Use DAIC-provided voice features as baselines:

- COVAREP features
- FORMANT features

Recommended experiments:

- COVAREP only
- FORMANT only
- COVAREP plus FORMANT
- Extracted openSMILE features compared against provided DAIC features

### 4.3 Androids openSMILE Features

Use `Androids.conf` to reproduce Androids baseline-style acoustic extraction.

Recommended experiments:

- openSMILE features on read speech
- openSMILE features on full interview recordings
- openSMILE features on interview clips with speaker-level aggregation

### 4.4 Unified Feature Set

To support fair cross-corpus comparison, extract one shared feature set from raw audio across both corpora.

Candidate shared sets:

- eGeMAPS
- ComParE
- IS09 emotion feature set
- Custom compact acoustic feature set

The shared set is essential for cross-corpus evaluation because DAIC-provided COVAREP files and Androids openSMILE features are not naturally identical.

## 5. Deep Learning Baselines

Deep learning baselines should be added after classical acoustic-feature baselines are reproducible.

### 5.1 Spectrogram CNN

Input:

- Log-Mel spectrograms
- Fixed-length windows

Possible models:

- Simple 2D CNN
- ResNet-style CNN
- CRNN with convolutional front end and recurrent temporal pooling

Use:

- Within-corpus baseline
- Segment-level modeling
- Speaker-level aggregation by mean probability or attention pooling

### 5.2 Recurrent Acoustic Sequence Model

Input:

- Frame-level acoustic features
- COVAREP sequences
- openSMILE low-level descriptors

Possible models:

- BiLSTM
- GRU
- CNN-BiLSTM
- Temporal convolutional network

Use:

- DAIC-WOZ sequence modeling
- Androids clip-level modeling
- Comparison against aggregated handcrafted features

### 5.3 Multiple Instance Learning

Depression labels are often available at the participant level while audio can be split into many segments. This suggests multiple instance learning.

Setup:

- Each participant is a bag.
- Each utterance or segment is an instance.
- The model predicts participant-level depression.

Aggregation options:

- Mean pooling
- Max pooling
- Attention pooling
- Transformer pooling over segment embeddings

## 6. Transformer-Based Approaches

Transformer-based speech models should be treated as the main state-of-the-art direction.

### 6.1 Frozen Speech Embeddings

Extract embeddings from pretrained self-supervised speech models and train lightweight classifiers.

Candidate models:

- WavLM
- wav2vec 2.0
- HuBERT
- data2vec Audio

Recommended first strategy:

- Freeze the speech encoder.
- Extract embeddings at segment level.
- Pool embeddings to participant level.
- Train logistic regression, SVM, MLP, or gradient boosting classifier.

Benefits:

- Lower compute requirement
- Reduced overfitting risk
- Strong reproducibility
- Easier cross-corpus comparison

### 6.2 Layer-Wise Representation Analysis

Prior work suggests that different transformer layers encode different information. Depression-related cues may not be strongest in the final layer.

Experiment:

- Extract embeddings from multiple layers.
- Evaluate each layer separately.
- Compare early, middle, and late layer performance.
- Report the most robust layers for within-corpus and cross-corpus evaluation.

### 6.3 Fine-Tuning

Fine-tuning should be attempted only after frozen-embedding baselines are stable.

Strategies:

- Fine-tune WavLM or wav2vec 2.0 on segment-level labels inherited from speaker labels.
- Use participant-level aggregation during evaluation.
- Apply regularization, early stopping, and small learning rates.
- Freeze lower layers if dataset size is too small.

Risks:

- Overfitting due to small clinical datasets
- Label noise when assigning participant labels to short segments
- Poor cross-corpus generalization

### 6.4 Parameter-Efficient Fine-Tuning

If compute allows, use parameter-efficient methods:

- Adapters
- LoRA
- Layer freezing
- Classification heads over frozen encoders

This can reduce overfitting and make experiments easier to reproduce.

### 6.5 Transformer-Based Segment Aggregation

For long interviews, segment-level embeddings can be fed into a second-stage transformer.

Architecture:

- Speech transformer extracts segment embeddings.
- Segment embeddings are ordered by time.
- A lightweight temporal transformer predicts participant-level depression.

This approach is especially relevant for DAIC-WOZ and Androids interview speech.

## 7. Experimental Design and Metrics

### 7.1 Within-Corpus Evaluation

DAIC-WOZ:

- Train on official train split.
- Tune on dev split.
- Evaluate on available labeled test or held-out split where appropriate.

Androids:

- Use provided 5-fold lists.
- Report mean and standard deviation across folds.
- Evaluate Reading and Interview tasks separately.

### 7.2 Metrics

Classification metrics:

- Accuracy
- Balanced accuracy
- F1-score
- Precision
- Recall
- ROC-AUC
- PR-AUC

Regression metrics for DAIC-WOZ PHQ-8 score:

- MAE
- RMSE
- Pearson correlation
- Spearman correlation

Recommended primary metric:

- Balanced accuracy or macro F1, due to class imbalance.

### 7.3 Statistical Testing

Use statistical comparisons where possible:

- Confidence intervals via bootstrap resampling
- Paired fold-level comparisons
- McNemar test for matched classification outputs
- Effect sizes for feature-group comparisons

## 8. Cross-Corpus Evaluation Plan

Cross-corpus evaluation is central to this project.

### 8.1 Motivation

Within-corpus performance can be inflated by corpus-specific recording conditions, interview protocols, microphones, demographics, or label collection methods. Cross-corpus evaluation tests whether voice-based depression cues generalize beyond a single dataset.

### 8.2 Core Protocols

Protocol A: Train DAIC-WOZ, Test Androids Interview

- Train model on DAIC-WOZ voice features.
- Select threshold using DAIC-WOZ dev split.
- Test directly on Androids interview recordings.
- Report performance without Androids fine-tuning.

Protocol B: Train Androids Interview, Test DAIC-WOZ

- Train using Androids interview folds or train partitions.
- Evaluate on DAIC-WOZ dev/test participants.
- Use only binary labels.

Protocol C: Train Androids Reading, Test Androids Interview

- Tests task generalization within the same corpus.
- Helps estimate how much read speech differs from spontaneous speech.

Protocol D: Train Interview, Test Reading

- Reverse of Protocol C.
- Useful for task robustness analysis.

Protocol E: Pooled Training With Leave-Corpus-Out Evaluation

- Train on one corpus.
- Evaluate on the held-out corpus.
- Optionally add corpus-balanced sampling.

### 8.3 Feature Requirements

Cross-corpus evaluation should use features extracted in the same way for both datasets.

Recommended cross-corpus feature sets:

- eGeMAPS or ComParE from raw audio
- WavLM embeddings
- wav2vec 2.0 embeddings
- HuBERT embeddings

Avoid using DAIC-only COVAREP files or Android-only openSMILE configurations as the sole cross-corpus feature source unless equivalent features are extracted for both corpora.

### 8.4 Domain Shift Analysis

Analyze domain shift across:

- Corpus
- Speech task
- Recording duration
- Demographic distribution
- Label definition
- Segment length
- Speech activity ratio

Recommended analyses:

- Feature distribution plots by corpus
- Embedding visualization using UMAP or t-SNE
- Corpus classifier to quantify dataset separability
- Performance stratified by gender and age group

### 8.5 Adaptation Experiments

After direct transfer baselines:

- Feature normalization per corpus
- CORAL or similar feature alignment
- Domain-adversarial training
- Fine-tuning on a small labeled target-corpus subset
- Leave-one-corpus-out validation if additional corpora are added later

## 9. Publication Roadmap

### Stage 1: Reproducible Baseline Paper

Focus:

- Unified voice-only preprocessing
- DAIC-WOZ and Androids within-corpus baselines
- Handcrafted acoustic features
- Classical machine learning models

Possible contribution:

- A transparent comparison of voice-only depression detection across two public corpora.

Expected outputs:

- Dataset manifests
- Feature extraction scripts
- Reproducible baseline tables
- Leakage-safe evaluation protocol

### Stage 2: Transformer Representation Paper

Focus:

- WavLM, wav2vec 2.0, HuBERT, and related speech transformers
- Frozen embeddings versus fine-tuning
- Layer-wise representation analysis
- Segment aggregation strategies

Possible contribution:

- Evidence about which speech transformer representations are most useful for depression detection and whether they generalize across corpora.

Expected outputs:

- Layer-wise performance tables
- Embedding visualizations
- Within-corpus and cross-corpus transformer results

### Stage 3: Cross-Corpus Generalization Paper

Focus:

- Train/test transfer between DAIC-WOZ and Androids Corpus
- Task generalization between read and interview speech
- Domain shift analysis
- Robustness and calibration

Possible contribution:

- A rigorous study of whether voice-based depression models learn depression-relevant cues or corpus-specific artifacts.

Expected outputs:

- Direct-transfer results
- Domain adaptation comparisons
- Error analysis
- Recommendations for future depression speech benchmarks

### Stage 4: Clinical and Ethical Analysis

Focus:

- Bias across demographic groups
- Model calibration
- Interpretability
- Privacy-preserving voice representations
- Responsible use limitations

Possible contribution:

- A clinically aware discussion of automatic voice-based depression screening, including failure modes and deployment constraints.

## Immediate Next Steps

No code should be written until the experimental protocol is finalized.

Recommended next actions:

1. Finalize whether DAIC transcripts may be used for timing-only segmentation.
2. Decide the first primary task: binary classification only, or binary plus PHQ-8 regression.
3. Define the canonical manifest schema.
4. Select the first shared feature set for both corpora, such as eGeMAPS or WavLM embeddings.
5. Define exact train/dev/test and cross-corpus protocols.
6. Create a repository structure for reproducible experiments.
7. Implement dataset indexing only after the above decisions are documented.
