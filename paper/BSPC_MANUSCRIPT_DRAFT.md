# Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations

Target journal: Biomedical Signal Processing and Control

Article type: Full paper [AUTHOR CONFIRMATION REQUIRED]

Submission status: journal-specific preparation only; not submitted.

Author metadata: [AUTHOR CONFIRMATION REQUIRED]

Corresponding author: [AUTHOR CONFIRMATION REQUIRED]

Affiliations: [AUTHOR CONFIRMATION REQUIRED]

## Highlights

- Frozen DAIC-to-Android transfer retained limited ranking but failed thresholded classification.
- Android-trained reading and interview classifiers achieved strong within-corpus discrimination.
- Participant-safe cross-task transfer showed shared but task-sensitive WavLM representations.
- Pooled reading-and-interview training gave the strongest point estimates, with uncertainty caveats.
- The study separates representation transfer from classifier-boundary transfer.

## Graphical Abstract

[OPTIONAL / AUTHOR DECISION REQUIRED: graphical abstract not generated in Stage 3D.]

## Abstract

Speech is a non-invasive behavioral signal that may reflect affective, psychomotor and cognitive changes, but depression classifiers trained in one corpus can be sensitive to language, recording conditions, elicitation task and label definition [@leal2025survey; @morenoTorres2012datasetShift; @alghowinem2016crossCultural]. We evaluated whether frozen self-supervised WavLM representations preserved depression-related information across corpora and speech tasks, and whether transfer failures reflected representation failure or classifier-boundary shift. In DAIC-WOZ, we developed a leakage-controlled participant-level pipeline using 10-second participant-speech windows, frozen `microsoft/wavlm-base-plus`, transformer layer 9, temporal mean pooling, participant mean aggregation and balanced logistic regression. The frozen DAIC WavLM classifier achieved ROC-AUC 0.6412 in zero-shot transfer to Android interview speech, but both the 0.5 and frozen DAIC thresholds predicted every Android interview participant as negative. An Android-trained interview classifier using the same frozen representation achieved within-Android ROC-AUC 0.8675. Android reading speech showed a similar pattern: zero-shot DAIC-to-reading ROC-AUC was 0.5795 with all-negative thresholded predictions, whereas Android-trained reading evaluation achieved ROC-AUC 0.8796. In participant-coherent Android analysis, reading-to-interview and interview-to-reading transfer achieved ROC-AUCs of 0.8475 and 0.8501; pooled reading-and-interview training achieved ROC-AUCs of 0.9058 for reading and 0.9135 for interview speech. Frozen WavLM features therefore retained useful depression-related ranking information, while decision boundaries and probability scales remained corpus- and task-sensitive. These are research classification findings, not clinical validation; DAIC local test labels were provisional, Android labels reflected HC/PT group status rather than PHQ-8 labels, and no prospective deployment study was performed.

## Keywords

Depression detection; speech analysis; self-supervised learning; WavLM; cross-corpus evaluation; domain shift; cross-task transfer

## 1. Introduction

Depression is a common mental-health condition with substantial personal, functional and public-health impact. The World Health Organization estimates that 5.7% of adults globally experience depression, and the US Preventive Services Task Force frames adult depression screening as a primary-care question only when systems are available for diagnosis, treatment and follow-up [@who2025depression; @uspstf2023depressionScreening]. Speech has long been studied as a behavioral modality that may reflect affective, psychomotor and cognitive changes associated with depressive states, including changes in rhythm, pauses, energy, voice quality and prosody [@leal2025survey; @tao2023featureCorrelation]. Compared with questionnaire-only or clinician-administered assessments, speech-based analysis is attractive because it can be collected during ordinary interactions. However, speech models also raise important questions about label validity, clinical interpretation, privacy, bias, screening versus diagnosis and deployment governance [@who2021aiHealthEthics; @nist2023aiRmf; @nautsch2019gdprSpeech; @kelly2019clinicalImpactAI].

Earlier speech-based depression studies have used handcrafted acoustic, prosodic, timing and interaction features, including pitch or F0, energy, cepstral features such as MFCCs, voice quality, speaking rate, pause behavior and response latency [@leal2025survey; @valstar2016avec; @tao2023featureCorrelation]. These features can be interpretable, but they are often sensitive to recording setup, language, task design and annotation quality. More recently, self-supervised speech models have offered reusable representations trained from large-scale unlabeled audio [@baevski2020wav2vec2; @chen2022wavlm; @yang2021superb]. Frozen representations from such models can reduce the risk of overfitting small clinical datasets, and can support controlled comparisons between corpora and elicitation tasks.

The central challenge addressed here is not only whether a model performs well inside one dataset, but whether the representation and decision rule survive shifts between datasets. DAIC-WOZ and the Androids Corpus differ in language, elicitation, label definition and recording context [@gratch2014distress; @valstar2016avec; @tao2023androids]. DAIC-WOZ uses English clinical-interview speech and PHQ-8 binary labels [@gratch2014distress; @valstar2016avec; @kroenke2009phq8], whereas Androids Corpus uses Italian speech with healthy-control versus depression-patient group labels [@tao2023androids]. These targets are related but not identical. A classifier trained on DAIC-WOZ may therefore fail externally even if its representation still ranks Android participants above chance.

This distinction motivates a careful separation between zero-shot external validation, target-domain threshold adaptation, target-domain calibration and target-domain classifier retraining. In many applied machine-learning settings, these protocols answer different questions: zero-shot evaluation tests direct transfer of the complete frozen model; threshold adaptation tests whether a score scale can be shifted; calibration tests probability mapping; and target-domain retraining tests whether the representation supports a new decision boundary [@morenoTorres2012datasetShift; @niculescuMizil2005probabilities; @vanCalster2019calibration]. For mental-health speech modeling, conflating these protocols can lead to overstated claims.

Speech task is another important source of variation. Interview speech may include spontaneous response patterns, conversational timing and participant-interviewer dynamics. Reading speech is more controlled and may reduce lexical and conversational variability, but it can also remove clinically informative interactive behavior [@tao2023androids; @tao2023featureCorrelation; @leal2025survey]. A representation that works across both tasks would be useful for robust measurement, while task-specific classifier boundaries would imply that representation transfer and classifier transfer must be evaluated separately.

This study developed and audited a reproducible sequence of experiments through DAIC-WOZ methodology development, DAIC-to-Android zero-shot evaluation, Android within-corpus evaluation, and participant-coherent Android reading/interview cross-task transfer. The central research argument is that frozen WavLM speech representations retain useful depression-related information across corpora and tasks, but classifier boundaries are sensitive to corpus and task conditions. Target-corpus training and pooled multi-task exposure substantially improved discrimination compared with direct zero-shot model transfer.

The contributions are:

1. A leakage-controlled frozen WavLM depression-detection pipeline.
2. Layer and aggregation analysis on DAIC-WOZ.
3. Zero-shot DAIC-to-Android external evaluation.
4. Controlled diagnosis of threshold, calibration and classifier-boundary shift.
5. Within-Android reading and interview benchmarks using identical frozen representations.
6. Participant-coherent cross-task transfer between reading and interview speech.
7. Pooled-task evaluation using participant-safe folds.
8. A reproducible experiment registry with paper-claim safeguards.

## 2. Related Work

### 2.1 Speech-based depression detection

Speech-based depression detection has been studied using clinical interviews, prompted responses, reading tasks and other elicitation paradigms [@leal2025survey]. The motivation is that depression may affect psychomotor behavior, speech timing, prosody, energy and voice quality. However, the direction and strength of reported markers are not uniform: acoustic findings can depend on sex, age, language, recording context, speech task and label definition, making cross-study comparison difficult [@leal2025survey]. This heterogeneity is one reason the present study separates corpus, task and classifier-boundary questions rather than treating speech-depression performance as a single pooled benchmark.

### 2.2 Handcrafted acoustic and timing features

Handcrafted features remain important because they provide interpretable links to speech production and interaction behavior. Common categories include pitch or F0, intensity and energy, spectral or cepstral measures such as MFCCs, formant or voice-quality features, pause duration, speech rate and turn-taking or response-latency measures [@leal2025survey; @valstar2016avec; @tao2023featureCorrelation]. In DAIC-WOZ-style guided interviews, response latency and participant-turn timing may be informative because they preserve aspects of the original interview timeline. Feature toolkits used in affective-computing benchmarks, including COVAREP/openSMILE-style low-level descriptors and functionals, reflect this tradition of interpretable acoustic and prosodic measurement [@valstar2016avec; @leal2025survey]. These features may not transfer directly to corpora lacking compatible interviewer and participant timing annotations.

### 2.3 Self-supervised speech representations

Self-supervised speech models learn representations from large unlabeled speech corpora and can be used as frozen feature extractors or fine-tuned for downstream tasks [@baevski2020wav2vec2; @chen2022wavlm; @yang2021superb]. WavLM is one such model family designed to produce general-purpose speech representations [@chen2022wavlm]. Benchmarking work such as SUPERB uses shared pretrained encoders with lightweight downstream heads, while depression-speech studies have explored wav2vec 2.0 or related SSL representations as feature extractors or fine-tuned encoders [@yang2021superb; @wu2023sslDepression; @huang2024voicePretraining]. In small mental-health datasets, frozen encoders can provide a conservative first baseline because they avoid adapting millions of parameters to limited labels. However, SSL pretraining does not by itself establish domain invariance or clinical validity.

### 2.4 Cross-corpus depression detection

Cross-corpus depression detection evaluates whether a model developed on one dataset transfers to another. This is especially important because clinical speech corpora differ in language, microphone conditions, participant recruitment, label source, task design and class prevalence [@alghowinem2016crossCultural; @seneviratne2021generalized; @leal2025survey]. Prior cross-cultural and cross-database depression-speech work motivates the question but does not remove the need to label each transfer protocol precisely. A zero-shot result measures the complete source-trained model, while target-domain training measures whether the feature representation remains useful when a new classifier boundary is learned.

### 2.5 Reading versus spontaneous speech

Reading and spontaneous interview speech may capture overlapping but non-identical information. Reading constrains lexical content and can reduce conversational variability, whereas interview speech may include spontaneous fluency, turn-taking and response latency [@tao2023androids; @tao2023featureCorrelation; @leal2025survey]. The Androids Corpus was designed with both read and spontaneous/interview recordings, including paired participants, which permits a direct comparison of these tasks [@tao2023androids]. Participant-level leakage safeguards are essential when the same person contributes both tasks.

### 2.6 Domain shift, calibration and transfer

Dataset shift can affect feature distributions, class priors and conditional label relationships, and can therefore alter both score calibration and the decision boundary implied by a source-trained classifier [@morenoTorres2012datasetShift]. A classifier can show above-chance ranking under ROC-AUC while failing at a fixed threshold if the probability scale shifts. Calibration assesses whether predicted probabilities correspond to observed outcome frequencies and is distinct from discrimination; Brier score, calibration slope/intercept, Platt scaling, isotonic regression and temperature scaling are common tools for this question [@brier1950probabilityForecasts; @niculescuMizil2005probabilities; @guo2017calibration; @vanCalster2019calibration]. Calibration can improve probability scores without improving ranking, and threshold adaptation can improve thresholded metrics without demonstrating pure zero-shot transfer. For this reason, this paper reports protocol scope explicitly for each result.

## 3. Materials and Methods

### 3.1 Study design

The study followed a sequential Phase 1-2E design. Phase 1 developed and froze DAIC-WOZ research candidates. Phase 1A established participant-level metadata and final-layer WavLM baselines. Phase 1B compared WavLM transformer layers using DAIC training data only and selected layer 9. Phase 1C compared participant-level aggregation methods for layer 9 and retained arithmetic mean aggregation. Phase 1D evaluated speech-timing features and early fusion, while Phase 1E evaluated late fusion. Phase 1F evaluated the frozen DAIC candidates once on a provisional local labeled DAIC test split.

Phase 2 moved to Androids Corpus preparation and external-compatibility testing. Phase 2A audited the Android data and established that WavLM preprocessing for Candidate A could be reproduced, while the DAIC timing model could not be directly transferred because only 17 of 28 timing features were exactly compatible. Phase 2B applied the frozen DAIC WavLM classifier zero-shot to Android interview speech. Phase 2C diagnosed whether failure came from threshold shift, calibration shift or classifier-boundary shift. Phase 2D evaluated Android reading speech. Phase 2E used participant-coherent folds across paired reading and interview tasks to evaluate cross-task transfer and pooled-task training.

### 3.2 Datasets

#### DAIC-WOZ

The DAIC-WOZ Phase 1 clean dataset contained 187 usable participants after excluding participants 409 and 422. The official development split used here contained 106 training, 34 development and 47 local labeled test participants. The corresponding 10-second clean speech-window counts were 9,001 train, 3,314 development and 4,743 local labeled test windows, for 17,058 total windows. Labels were `PHQ8_Binary` labels derived from PHQ-8 questionnaire scoring [@valstar2016avec; @kroenke2009phq8]. The test split used in Phase 1F was a provisional local labeled test derived from `Data/full_test_split.csv`; official AVEC hidden-test provenance was not established. Therefore, DAIC test results are described as provisional local labeled test results rather than official hidden AVEC test performance.

#### Androids Corpus

The Androids Corpus audit identified 118 unique participants, with 54 healthy controls and 64 depression patients [@tao2023androids]. The corpus contained 116 full interview recordings, 112 read-speech recordings and 874 interview clips. Reading recordings included 54 HC and 58 PT participants; interview recordings included 52 HC and 64 PT participants. A total of 110 participants had both reading and interview recordings, with 52 HC and 58 PT participants in the paired Phase 2E subset. Android labels were treated as clinical group status labels, not PHQ-8 labels: HC was encoded as 0 and PT as 1. The Android speech was Italian [@tao2023androids], and Android evaluation used task-specific folds for Phase 2C and Phase 2D, then participant-coherent reading/interview folds for Phase 2E.

### 3.3 Speech preprocessing

DAIC preprocessing used participant-only speech intervals from the clinical interview. Participant speech was reconstructed from clean mapped intervals, converted to mono, resampled to 16 kHz, corrected for DC offset and peak-normalized without dataset-level normalization. Ten-second windows with 50% overlap were used, equivalent to a 5-second hop and 160,000 samples per full window. Incomplete windows shorter than 10 seconds were not used for WavLM extraction. The completed extraction used no data augmentation and no denoising.

Android interview and reading processing preserved task separation. Android interview processing reconstructed participant speech where compatible with Phase 2A timing and clip audits, while Android reading used full reading recordings. Phase 2D reading extraction used the same 16 kHz target sample rate, 10-second windows, 5-second hop, DC-offset removal and peak normalization. The reading extraction report explicitly recorded no dataset normalization, no augmentation and no denoising. Reading and interview speech were not combined before Phase 2E pooled-task model training; pooled-task training combined participant-level examples under participant-coherent folds rather than concatenating a participant's tasks into a single feature vector.

### 3.4 WavLM representation

WavLM is a pretrained self-supervised speech model using masked speech prediction and denoising-oriented pretraining for broad speech representation learning [@chen2022wavlm]. All WavLM experiments used `microsoft/wavlm-base-plus` pinned to model commit `4c66d4806a428f2e922ccfa1a962776e232d487b`. The encoder was frozen and used in evaluation/inference mode. Phase 1B inspected hidden-state indexing and compared the 12 transformer encoder layers. The primary selected representation was transformer layer 9, not the pre-transformer feature-projection state. Each 10-second window was represented by temporal mean pooling over the WavLM time dimension, producing a 768-dimensional vector. Participant-level representations were produced by arithmetic mean pooling across valid window embeddings.

Layer 9 and mean participant aggregation were selected using DAIC development experiments only. Android results were not used to select the WavLM layer or participant aggregation method.

### 3.5 Classifiers

The main classifier family was logistic regression with L2 regularization, `class_weight="balanced"` and `max_iter=10000`. WavLM-only pipelines used a `StandardScaler` followed by logistic regression. Timing models used training-fold imputation and scaling. Hyperparameter selection considered C values of 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0 and 100.0. Scalers, imputers, calibrators and classifiers were fit inside the appropriate training folds only. Thresholds were selected from training or development data as specified by each protocol, and threshold-adapted results were not described as pure zero-shot evaluation.

### 3.6 Evaluation protocols

#### DAIC development and provisional local testing

DAIC development used train-only or train/development protocols depending on phase. Phase 1B through Phase 1E model-selection comparisons used training-data cross-validation and development evaluation where specified. Phase 1F evaluated frozen DAIC candidates once on 47 provisional local labeled test participants. This Phase 1F test was not treated as official hidden AVEC evaluation.

#### DAIC-to-Android zero-shot evaluation

Phase 2B applied the frozen DAIC WavLM Candidate A directly to Android interview participant embeddings. The source corpus was DAIC-WOZ, the target corpus was Androids Corpus, the source task was free-speech-compatible interview speech and the target task was Android interview speech. No Android labels were used for model fitting, scaler fitting, threshold selection, calibration or model revision. This was the main zero-shot external evaluation.

#### Threshold-only adaptation

Phase 2C Protocol B preserved the frozen DAIC score ranking but selected thresholds within Android training folds. This used Android target labels for threshold selection and therefore was target-domain threshold adaptation, not pure zero-shot evaluation. The classifier was not refitted, and the protocol did not change individual predicted probabilities. Under dataset-shift terminology, it tested whether a source score could be made operational under a target-domain threshold rather than whether the source classifier transferred unchanged [@morenoTorres2012datasetShift].

#### Frozen-score calibration

Phase 2C Protocol C calibrated frozen DAIC scores using Android training folds. This used target-domain labels to fit calibration mappings and therefore was target-domain score calibration rather than zero-shot evaluation. The WavLM encoder and source score generator were not refit. This protocol addressed probability calibration, not representation learning or source-model external validation [@niculescuMizil2005probabilities; @vanCalster2019calibration].

#### Android within-corpus interview evaluation

Phase 2C Protocol D trained a logistic-regression classifier within Android interview folds using the same frozen layer-9 participant mean representation. Target labels were used for training, inner hyperparameter selection and fold-specific threshold adaptation. This protocol tested within-Android discrimination, not DAIC-to-Android model transfer.

#### Android within-corpus reading evaluation

Phase 2D evaluated reading speech separately from interview speech. Protocol A applied the frozen DAIC WavLM classifier zero-shot to Android reading embeddings. Protocol B trained an Android reading logistic-regression classifier under participant-level nested five-fold cross-validation using the same frozen layer-9 mean representation. The Android-trained reading result was within-corpus validation, not external validation.

#### Participant-coherent cross-task transfer

Phase 2E created participant-coherent folds over the 110 Android participants with both reading and interview recordings. Matched-task protocols trained and evaluated on the same task. Cross-task protocols trained on one task and evaluated on the other task for held-out participants. Participants did not cross training and evaluation folds, and reading/interview examples from the same participant remained in the same fold.

#### Pooled-task training

Phase 2E pooled reading and interview training examples within participant-coherent folds and evaluated separately on reading and interview held-out participants. Pooled training used the same frozen layer-9 representation and logistic-regression model family. Pooled results were within-Android pooled-task estimates, not cross-corpus external validation.

### 3.7 Leakage prevention

All model evaluation was participant-level. Cross-validation used participant grouping, and the outer evaluation fold was isolated from preprocessing, hyperparameter selection and threshold selection. `StandardScaler` fitting occurred inside training folds only. C selection used inner folds only. Threshold selection used training out-of-fold predictions or development data according to the protocol, never the held-out fold being evaluated. Phase 2E introduced participant-coherent folds so that reading and interview recordings from the same participant remained together.

### 3.8 Metrics and uncertainty

Primary discrimination was reported with ROC-AUC. Additional metrics included PR-AUC, balanced accuracy, macro F1, positive-class precision, sensitivity, specificity, negative predictive value and Brier score. Brier score was used as a proper scoring-rule summary of probability error, while discrimination and calibration were interpreted separately [@brier1950probabilityForecasts; @vanCalster2019calibration]. Balanced accuracy is also interpretable as unweighted average recall. Bootstrap uncertainty estimates used 5,000 stratified participant-level bootstrap replicates where reported. Confidence intervals for Phase 2E paired comparisons were treated as exploratory because repeated cross-validation and paired task protocols induce dependencies.

### 3.9 Reproducibility and experiment governance

The repository preserved frozen protocols, source-code configurations, compact run records and an experiment registry. The frozen source tag for this manuscript was `paper-experiment-freeze-v1`. The registry contained 42 completed experiment or audit records and explicit paper-interpretation metadata distinguishing methodological ablations, provisional local test results, zero-shot external evaluations, target-domain adaptation and within-corpus validation. Registry validation was read-only, and compact records excluded restricted artifacts such as raw audio, embeddings, participant predictions and fitted models.

## 4. Results

### 4.1 DAIC representation development

Phase 1A established initial participant-level baselines. The metadata-only sanity baseline achieved train-CV ROC-AUC 0.5510 and provisional local test ROC-AUC 0.4437. The final-layer WavLM participant baseline achieved train-CV ROC-AUC 0.5115, development ROC-AUC 0.5692 and provisional local test ROC-AUC 0.5195. These early results motivated a frozen layer-wise representation analysis.

Phase 1B evaluated all 12 WavLM transformer layers using DAIC train-only nested cross-validation. Layer 9 had the strongest mean outer ROC-AUC, 0.5956, compared with 0.4988 for layer 12. Layer 9 was therefore selected for subsequent WavLM experiments. This selection used DAIC training data only and did not use Android results.

Phase 1C tested whether participant aggregation limited performance. Mean aggregation for layer 9 achieved mean outer ROC-AUC 0.5956, median aggregation achieved 0.5905 and mean-plus-standard-deviation aggregation with PCA achieved 0.5683. Because alternatives did not improve train-only performance, arithmetic mean aggregation was retained.

Phase 1D examined whether compact speech-timing features captured information lost by concatenating participant speech intervals. Timing-only features achieved train nested-CV ROC-AUC 0.6126, WavLM-only achieved 0.5956 and early fusion achieved 0.5747. Phase 1E then compared timing-only, WavLM-only and late-fusion protocols; timing-only remained strongest among these DAIC guided-interview comparisons, while weighted late fusion and logistic stacking did not improve the final candidate.

Phase 1F froze two DAIC research candidates. Candidate A was the free-speech-compatible WavLM layer-9 mean representation with C=0.001 and frozen threshold 0.5558; it achieved provisional local test ROC-AUC 0.5065. Candidate B was the guided-interview timing model with C=100.0 and frozen threshold 0.2927; it achieved provisional local test ROC-AUC 0.6775. Candidate B was not directly portable to Android because Android timing features were not fully compatible with the Phase 1 timing definitions.

### 4.2 Zero-shot DAIC-to-Android interview transfer

Phase 2B applied the frozen DAIC WavLM Candidate A to 116 Android interview participants, including 52 HC and 64 PT participants. The Android interview preprocessing produced 4,249 valid 10-second windows and the embedding quality checks passed. The frozen DAIC classifier achieved zero-shot ROC-AUC 0.6412 and PR-AUC 0.6809. However, predicted probabilities were compressed below both the default 0.5 threshold and the frozen DAIC threshold of 0.5558: the maximum predicted probability was 0.3938, and every participant was predicted negative at both thresholds. Thresholded balanced accuracy was 0.5000, macro F1 was 0.3095, sensitivity was 0.0000 and specificity was 1.0000. The Brier score was 0.4056.

This result indicates limited cross-corpus ranking transfer, not successful thresholded classification.

### 4.3 Domain-shift diagnosis

Phase 2C separated threshold shift, calibration shift and classifier-boundary shift on the same 116 Android interview participants. Protocol A reproduced the frozen DAIC zero-shot ROC-AUC of 0.6412. Protocol B used Android training folds for threshold-only adaptation and preserved ROC-AUC 0.6412 while improving macro F1 to 0.5688. Protocol C calibrated frozen DAIC scores using Android training folds; Brier score improved to 0.2474, but ROC-AUC decreased to 0.5694. Protocol D trained an Android interview logistic-regression classifier on the same frozen layer-9 participant embeddings and achieved ROC-AUC 0.8675, PR-AUC 0.8892, balanced accuracy 0.8275 and macro F1 0.8263 with fold-specific threshold adaptation. The Phase 2C protocols therefore separate discrimination, thresholded classification and probability calibration rather than treating all improvements as external transfer [@niculescuMizil2005probabilities; @vanCalster2019calibration].

The Android-trained interview classifier improved ROC-AUC over the frozen DAIC reference by 0.2263. Feature-distribution analysis showed a centroid cosine distance of 0.1767, mean absolute standardized shift of 2.3116 and 550 of 768 features shifted by more than one DAIC standard deviation. Together, these findings indicate that classifier-boundary shift was the dominant failure source, while threshold and calibration shifts were secondary.

### 4.4 Android reading-speech evaluation

Phase 2D evaluated 112 Android reading participants, including 54 HC and 58 PT participants. Reading preprocessing produced 955 valid 10-second windows with zero extraction failures, and participant aggregation used all 955 windows.

The frozen DAIC WavLM classifier transferred less strongly to reading speech than to Android interview speech, with zero-shot ROC-AUC 0.5795 and PR-AUC 0.5997. As in interview transfer, all participants were predicted negative at both the 0.5 threshold and the frozen DAIC threshold. In contrast, the Android-trained reading classifier achieved ROC-AUC 0.8796 with 95% CI [0.8113, 0.9377], PR-AUC 0.8929, balanced accuracy 0.7768, macro F1 0.7766 and Brier score 0.1438. This was a within-Android nested cross-validation estimate, not external validation.

### 4.5 Reading-versus-interview comparison

The Phase 2D descriptive task comparison contrasted Android-trained reading and interview classifiers. The reading classifier achieved ROC-AUC 0.8796, while the interview classifier from Phase 2C achieved ROC-AUC 0.8675. The task centroid cosine distance was 0.0442, and within-participant reading/interview cosine similarity averaged 0.9273 across 110 participants with both tasks. Nevertheless, 292 of 768 features shifted by more than one interview standard deviation. Because the original reading and interview task folds differed, this comparison was descriptive rather than a controlled cross-task transfer test.

### 4.6 Participant-coherent cross-task transfer

Phase 2E used participant-coherent folds over 110 paired Android participants, with 52 HC and 58 PT participants. Matched within-task training achieved ROC-AUC 0.9048 for reading and 0.9065 for interview. Cross-task training remained strong but lower than matched-task training: reading-to-interview achieved ROC-AUC 0.8475, and interview-to-reading achieved ROC-AUC 0.8501. The reading-to-interview AUC gap versus matched interview was -0.0590, and the interview-to-reading gap versus matched reading was -0.0547.

Pooled reading-and-interview training achieved ROC-AUC 0.9058 for reading and 0.9135 for interview. Pooled-to-reading improved over interview-to-reading by 0.0557, and pooled-to-interview improved over reading-to-interview by 0.0660. However, the paired bootstrap intervals for these pooled gains included zero, so the pooled advantage should be interpreted as suggestive rather than conclusive.

### 4.7 Classifier-boundary analysis

Phase 2E classifier-boundary analysis found only moderate similarity between reading-trained and interview-trained logistic-regression boundaries. Across folds, coefficient cosine similarity averaged 0.4752, coefficient sign agreement averaged 0.6659 and top-50 absolute-weight feature Jaccard overlap averaged 0.1216. Matched-task score correlation averaged 0.7123, while cross-target score correlation averaged 0.5555. These results support a distinction between shared representation information and partly task-specific classifier boundaries.

## 5. Discussion

### 5.1 Main findings

The experiments support three main findings. First, frozen WavLM layer-9 representations contained depression-related ranking information across corpora and tasks. Second, direct transfer of the DAIC-trained classifier failed at the threshold and probability-scale level, especially on Android interview and reading speech where all zero-shot thresholded predictions were negative. Third, target-corpus classifier learning and pooled task exposure substantially improved discrimination within Android.

### 5.2 Representation transfer versus classifier transfer

The contrast between zero-shot ROC-AUC and thresholded predictions is central. A ROC-AUC of 0.6412 on Android interview speech showed that the frozen DAIC scores ranked PT participants above HC participants more often than chance. Yet the same score distribution lay entirely below the frozen decision thresholds. This means the representation-score combination contained some ranking signal, but the source-trained probability scale and classifier boundary did not transfer sufficiently for direct thresholded use. This distinction follows the calibration literature's separation between discrimination and the reliability of predicted probabilities [@niculescuMizil2005probabilities; @vanCalster2019calibration].

### 5.3 Corpus-specific decision boundaries

The domain-shift analysis suggests that the main limitation was not simply threshold choice. Threshold adaptation improved macro F1, and calibration improved Brier score, but the Android-trained classifier produced the largest discrimination improvement. The 550 shifted WavLM features and the 0.2263 AUC gain from Android classifier training are consistent with corpus-specific decision boundaries. Here, classifier-boundary shift is used descriptively: it means that a new target-domain logistic-regression boundary on the same frozen features separated Android HC/PT participants much better than the DAIC-trained boundary. This interpretation is aligned with broader observations that dataset shift can affect feature distributions, class priors, conditional relationships and classifier calibration [@morenoTorres2012datasetShift; @vanCalster2019calibration].

### 5.4 Reading and interview speech

Reading speech and interview speech produced similar within-Android discrimination when each task had its own classifier. Prior speech-depression literature treats task design as a meaningful source of variation: reading can control lexical content and topic variability, while interviews preserve spontaneous fluency, conversational timing and response structure [@tao2023androids; @tao2023featureCorrelation; @leal2025survey]. The high within-participant representation similarity in this study suggests shared acoustic or vocal information across tasks. At the same time, cross-task transfer was weaker than matched-task training, and classifier coefficients showed only moderate similarity. Reading and interview tasks therefore appear to share useful representation structure while still requiring partly task-specific classifier boundaries.

### 5.5 Benefits and limits of pooled-task learning

Pooled reading-and-interview training produced the strongest point estimates in Phase 2E. Pooled-to-reading ROC-AUC was 0.9058 and pooled-to-interview ROC-AUC was 0.9135. These gains are consistent with the idea that exposure to both tasks can help the classifier learn a more robust boundary. However, the paired bootstrap intervals for pooled gains included zero, so the pooled advantage should be described as suggestive, not definitive.

### 5.6 Implications for cross-corpus depression detection

The results argue for careful protocol labeling in cross-corpus depression detection. A zero-shot external evaluation, a target-domain threshold adaptation, a target-domain calibration and a within-corpus trained classifier answer different questions. The Android-trained interview and reading results demonstrate strong within-corpus discrimination with frozen WavLM features, but they do not prove direct DAIC-to-Android model transfer. Conversely, the DAIC zero-shot results show limited external ranking transfer but not clinically usable thresholded performance. Prior cross-corpus depression-speech work makes this distinction important because transfer settings can mix language, task, label-source and recording changes rather than isolating one cause [@alghowinem2016crossCultural; @seneviratne2021generalized; @leal2025survey].

### 5.7 Comparison with prior work

The findings are broadly consistent with prior concerns that depression speech models can be sensitive to dataset, task and language differences [@alghowinem2016crossCultural; @seneviratne2021generalized; @leal2025survey]. They also align with the expectation that self-supervised representations provide useful general speech features, while downstream classifiers require careful validation under target conditions [@baevski2020wav2vec2; @yang2021superb; @wu2023sslDepression; @huang2024voicePretraining]. These sources support the methodological comparison, but they do not imply that this study has established clinical deployment readiness or domain-invariant depression biomarkers.

### 5.8 Ethical and Clinical Considerations

These are research-only classification results. They do not establish clinical diagnosis, screening utility or deployment readiness. The models are not diagnostic tools. DAIC-WOZ PHQ8_Binary labels and Android HC/PT labels were supervisory labels for model evaluation, not clinical endpoints validated prospectively in this study. Adult depression-screening guidance assumes further diagnostic evaluation and adequate systems for treatment and follow-up, so a speech classifier should not be interpreted as a stand-alone diagnostic pathway [@uspstf2023depressionScreening].

Speech data are also sensitive because voice can contain information about identity, language, accent, health, emotion and recording environment. Speech-data governance should therefore address consent, access control, retention, data minimization, de-identification limits and voice re-identification risk [@nautsch2019gdprSpeech; @tomashenko2021voicePrivacy]. The present repository excludes raw audio, embeddings, participant predictions and fitted models from versioned paper artifacts, and the compact registry records are designed for auditability without redistributing restricted speech data.

Bias and fairness remain open requirements. The corpora differ in language, recruitment, task design and label source, and the study did not perform subgroup evaluation by sex, age, dialect, accent, site or other demographic variables. Prior speech-technology evidence shows that automated speech systems can have differential performance across speaker groups, and the NIST AI RMF treats harmful-bias management, privacy, validity, reliability, safety, transparency and accountability as core trustworthy-AI concerns [@koenecke2020racialDisparitiesASR; @nist2023aiRmf]. No demographic fairness audit was completed.

Clinical deployment would require evidence beyond the retrospective experiments reported here. Prospective or early-stage clinical evaluation should examine actual clinical performance, safety, human factors, error modes, workflow fit, local representativeness, calibration in the target population and downstream patient outcomes [@kelly2019clinicalImpactAI; @vasey2022decideAI; @liu2020consortAI; @vanCalster2019calibration]. No prospective deployment study was performed. No clinical utility study was completed. Any future use should include human oversight, locally validated thresholds, monitoring for distribution shift, clear escalation pathways and governance for updates or decommissioning [@who2021aiHealthEthics; @nist2023aiRmf].

## 6. Limitations

The study had several limitations. Dataset sizes were modest, especially for participant-level cross-validation. DAIC-WOZ and Androids Corpus used related but non-identical target labels: PHQ-8 binary labels in DAIC and clinical group status in Android. The corpora differed in language, recording conditions, task structure and participant recruitment. The Phase 1F DAIC test labels were from a local labeled file without established official hidden AVEC provenance. Android-trained models were not evaluated on an independent third corpus. Phase 2D reading and interview comparisons were initially descriptive because original task folds differed; Phase 2E addressed this with participant-coherent folds but remained within Androids Corpus. No demographic fairness analysis, subgroup calibration study, uncertainty-aware clinical threshold study, prospective validation, workflow evaluation or clinical utility study was performed. The models were not compared with clinician assessment at the point of care. False positives and false negatives were not evaluated in a clinical workflow, and the appropriate actionability of any threshold remains unknown. The privacy analysis was limited to documentation and artifact exclusion rather than a formal governance review. The adaptation comparisons intentionally used logistic regression only, so they do not address whether more complex model families would improve performance. Finally, the confidence intervals for pooled-task gains included zero, so pooled-task superiority was not definitively established.

## 7. Conclusion

Frozen WavLM features captured useful depression-related information across DAIC-WOZ and Androids Corpus, and across Android reading and interview tasks. However, direct classifier transfer across corpora failed at the decision boundary: Android zero-shot scores showed limited ranking transfer but all participants were predicted negative at frozen thresholds. Target-corpus classifier learning restored strong within-Android discrimination, and participant-coherent cross-task experiments showed substantial sharing between reading and interview speech with partly task-specific classifier boundaries. Pooled reading-and-interview training achieved the strongest point estimates, but its gains should be interpreted cautiously. These findings support research on representation transfer and corpus-specific calibration, not clinical readiness. External validation beyond Androids Corpus, demographic fairness evaluation, privacy governance, prospective clinical validation and clinical utility assessment remain necessary before any clinical claim can be made.

## 8. Data and Code Availability

The source datasets are not redistributed. Access to DAIC-WOZ and Androids Corpus data remains governed by the original dataset terms [@gratch2014distress; @valstar2016avec; @tao2023androids]. The repository preserves code, compact configuration records, paper evidence maps and the experiment registry associated with the frozen source tag `paper-experiment-freeze-v1`. Raw audio, participant-level artifacts, embeddings, fitted models and participant predictions are excluded from the paper artifacts because they may contain restricted or sensitive information. This supports auditability of reported claims while limiting third-party reproduction of restricted-data processing to users who separately obtain the underlying datasets under their original access conditions.

## Declaration of Competing Interest

[AUTHOR CONFIRMATION REQUIRED]

## Funding

[AUTHOR CONFIRMATION REQUIRED]

## Ethics Approval

[AUTHOR CONFIRMATION REQUIRED]

## Consent to Participate

[AUTHOR CONFIRMATION REQUIRED]

## Consent for Publication

[AUTHOR CONFIRMATION REQUIRED]

## CRediT Authorship Contribution Statement

[AUTHOR CONFIRMATION REQUIRED]

## Acknowledgements

[AUTHOR CONFIRMATION REQUIRED]

## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

[AUTHOR CONFIRMATION REQUIRED]

## Preprint Statement

[AUTHOR CONFIRMATION REQUIRED]
