# Study Experiments and Rationale Summary

## Purpose of This Document

This document explains why each major experiment in the DepressionFromVoice study was conducted, what question it answered, and how it motivated the next stage. It is a documentation and reviewer-support artifact only. It does not add experiments, change frozen metrics, alter participant counts, or revise the scientific claims already audited in the manuscript package.

Authoritative sources are the frozen experiment registry, the generated experiment log, the paper evidence map, the paper results tables, the final numerical audit, the final result-scope audit, compact run records under `records/runs/`, and official reports under `outputs/phase1_daic/` and `outputs/phase2_android/`.

## Overall Experimental Logic

The study followed a stepwise logic. First, DAIC-WOZ was used for controlled methodological development because it provided the initial English interview dataset, PHQ8_Binary labels, and a structured train/development/provisional local labeled test split. Second, Androids Corpus was used to evaluate whether the frozen DAIC-compatible WavLM representation carried information across corpus, language, label definition, and speech-task changes. Third, Android-specific and task-specific analyses separated representation usefulness from classifier-boundary transfer.

The experiments therefore answer a sequence of related but distinct questions:

1. Which DAIC-compatible speech representation should be frozen?
2. Are DAIC guided-interview timing features useful, and are they portable?
3. Does a frozen DAIC WavLM classifier transfer zero-shot to Android interview or reading speech?
4. If zero-shot thresholded classification fails, is the issue threshold, calibration, representation, or classifier boundary?
5. Does the same frozen representation support Android-trained interview and reading models?
6. Do reading and interview speech share enough information for participant-coherent cross-task transfer?
7. Do pooled reading-and-interview folds provide evidence for multi-task exposure while remaining within Androids Corpus?

The final interpretation is deliberately limited: frozen WavLM features retained useful depression-related ranking information across corpora and tasks, but probability scales and classifier boundaries were corpus- and task-sensitive.

## Phase 1: DAIC-WOZ Methodological Development

### Phase 1A: Baseline Participant-Level Depression Classification

- What was tested: a metadata-only sanity baseline and a final-layer WavLM participant baseline on DAIC-WOZ.
- Why it was necessary: before optimizing WavLM layers or moving to external data, the study needed to check whether trivial metadata signals or default final-layer embeddings were adequate.
- Method: participant-level logistic regression using DAIC train/development data and a provisional local labeled test split. The metadata baseline used duration/window-count style metadata. The WavLM baseline used the final WavLM layer with participant mean aggregation.
- Main result: metadata-only provisional local labeled test ROC-AUC was 0.4437; final-layer WavLM provisional local labeled test ROC-AUC was 0.5195.
- Significance: the metadata baseline was insufficient, and the final WavLM layer was weak enough to motivate layer-wise analysis.
- Limitation: the DAIC local labeled test was provisional and not official hidden AVEC evaluation.
- How it motivated the next phase: Phase 1B evaluated all WavLM layers to identify a better frozen speech representation.

### Phase 1B: WavLM Layer-Wise Analysis

- What was tested: WavLM base-plus transformer layers 1 through 12.
- Why it was necessary: self-supervised speech models often distribute task-relevant information unevenly across layers, so the final layer could not be assumed optimal.
- Method: train-only nested repeated stratified cross-validation on DAIC-WOZ training data, followed by the recorded development protocol.
- Main result: layer 9 was selected. The results table reports train-only mean outer ROC-AUC 0.5956; the registry paper-ready sentence records ROC-AUC 0.6443 under the recorded train/development protocol.
- Significance: layer 9 became the frozen WavLM representation for later analyses.
- Limitation: this was DAIC-only methodological selection, not external validation.
- How it motivated the next phase: Phase 1C tested how to aggregate layer-9 window embeddings to participant-level features.

### Phase 1C: Aggregation Strategy Analysis

- What was tested: participant mean aggregation, median aggregation, and mean plus standard deviation with PCA for layer-9 WavLM embeddings.
- Why it was necessary: depression labels were participant-level, while WavLM features were extracted from speech windows.
- Method: DAIC train-only nested cross-validation using layer-9 embeddings.
- Main result: mean aggregation retained ROC-AUC 0.5956 in the results table; median was 0.5905 and mean plus standard deviation PCA was 0.5683.
- Significance: simple participant mean aggregation was retained because it was competitive, transparent, and stable.
- Limitation: aggregation changes did not solve the larger classifier-transfer question.
- How it motivated the next phase: Phase 1D compared WavLM features with DAIC guided-interview timing and interaction features.

### Phase 1D: Timing and Interaction Feature Analysis

- What was tested: timing-only features, WavLM-only features, and early fusion of timing plus WavLM.
- Why it was necessary: DAIC-WOZ guided interviews include response timing and interaction structure that may carry behavioral information.
- Method: train-only nested DAIC cross-validation under the Phase 1D timing/fusion protocol.
- Main result: timing-only reached train nested-CV ROC-AUC 0.6126; WavLM-only was 0.5956; early fusion was 0.5747.
- Significance: timing features appeared useful inside the DAIC guided-interview setting.
- Limitation: timing features depended on guided-interview structure and were not directly portable to Androids Corpus.
- How it motivated the next phase: Phase 1E tested whether late fusion could combine timing and WavLM more safely than early fusion.

### Phase 1E: Late Fusion Analysis

- What was tested: timing-only, WavLM-only, weighted late fusion, and logistic stacking.
- Why it was necessary: the study needed to know whether separate WavLM and timing signals could be combined without relying on a single feature space.
- Method: leakage-safe DAIC train-only late-fusion protocols.
- Main result: timing-only remained strongest in registry paper-ready interpretation with ROC-AUC 0.6245; weighted late fusion was 0.6044 and logistic stacking was 0.5413 in the results table.
- Significance: fusion did not provide a reliable improvement over timing-only in DAIC.
- Limitation: timing remained guided-interview-dependent and therefore did not solve Android transfer.
- How it motivated the next phase: Phase 1F froze both a free-speech-compatible WavLM candidate and a guided-interview timing candidate for final DAIC local evaluation.

### Phase 1F: Final Frozen DAIC Evaluation

- What was tested: frozen Candidate A, the layer-9 WavLM free-speech-compatible model, and frozen Candidate B, the guided-interview timing model.
- Why it was necessary: final DAIC candidates had to be evaluated once after development choices were frozen.
- Method: one provisional local labeled DAIC test evaluation from `Data/full_test_split.csv`, with no post-test model changes.
- Main result: WavLM Candidate A achieved ROC-AUC 0.5065; timing Candidate B achieved ROC-AUC 0.6775.
- Significance: timing was stronger inside DAIC, but WavLM was the transfer-compatible candidate because it did not require guided-interview timing definitions.
- Limitation: the test was provisional local labeled DAIC evaluation, not official hidden AVEC evaluation.
- How it motivated the next phase: Phase 2A checked whether Androids Corpus could support WavLM transfer and whether DAIC timing features were compatible.

## Phase 2: Androids Corpus Transfer and Task Analysis

### Phase 2A: Androids Corpus Audit and Preparation

- What was tested: corpus inventory, label/task availability, WavLM preprocessing compatibility, and timing-feature compatibility.
- Why it was necessary: Androids Corpus differs from DAIC-WOZ in language, task design, and labels, so data compatibility had to be checked before modeling.
- Method: audit of Android participants, reading and interview recordings, WavLM preprocessing, and DAIC timing-feature portability.
- Main result: Androids Corpus included 118 participants, 116 interviews, 112 readings, and 874 interview clips. WavLM preprocessing was reproducible, while direct timing-model transfer was invalid because response-latency features were unavailable.
- Significance: the audit supported WavLM-based Android analysis and prevented inappropriate timing-feature transfer.
- Limitation: Phase 2A was an audit, not a classifier evaluation.
- How it motivated the next phase: Phase 2B applied the frozen DAIC WavLM classifier to Android interview speech.

### Phase 2B: DAIC-to-Android Zero-Shot Interview Transfer

- What was tested: frozen DAIC WavLM Candidate A applied zero-shot to Android interview speech.
- Why it was necessary: zero-shot evaluation had to precede adaptation so direct model transfer could be separated from target-domain fitting.
- Method: frozen DAIC classifier, frozen threshold settings, and no Android fitting.
- Main result: Android interview ROC-AUC was 0.6412, PR-AUC was 0.6809, and Brier score was 0.4056; both the 0.5 and frozen DAIC thresholds predicted all Android participants negative.
- Significance: the representation retained limited ranking information, but thresholded classification failed.
- Limitation: corpus, language, label definition, and probability-scale shifts limited direct classifier transfer.
- How it motivated the next phase: Phase 2C separated threshold adaptation, calibration, Android-trained classifier learning, and domain-shift analysis.

### Phase 2C: Domain Shift, Calibration, and Android-Trained Interview Models

- What was tested: Protocol A frozen DAIC reference, Protocol B threshold-only adaptation, Protocol C frozen-score calibration, Protocol D Android-trained logistic regression, and aggregate domain-shift diagnostics.
- Why it was necessary: the Phase 2B result could reflect threshold shift, calibration shift, representation limits, or classifier-boundary shift.
- Method: fixed frozen DAIC scores were compared with Android train-fold threshold adaptation, Android train-fold score calibration, and Android-trained logistic regression using the same frozen WavLM representation.
- Main result: Protocol A retained ROC-AUC 0.6412; Protocol B retained ROC-AUC 0.6412 but improved thresholded metrics; Protocol C improved Brier score to 0.2474 but reduced ROC-AUC to 0.5694; Protocol D achieved ROC-AUC 0.8675, PR-AUC 0.8892, balanced accuracy 0.8275, and macro F1 0.8263.
- Significance: results indicated that the frozen representation was useful when an Android-specific classifier boundary was learned, while direct DAIC decision rules did not transfer.
- Limitation: Protocols B and C used Android labels and were not pure zero-shot; Protocol D was within-Android cross-validation, not external validation.
- How it motivated the next phase: Phase 2D tested whether the same pattern held for Android reading speech.

### Phase 2D: Android Reading-Speech Evaluation

- What was tested: frozen DAIC zero-shot reading evaluation, Android-trained reading logistic regression, and descriptive reading-versus-interview comparison.
- Why it was necessary: reading speech is more constrained than interview speech, and the study needed to know whether the WavLM representation remained informative across Android speech tasks.
- Method: frozen DAIC scores were evaluated on Android reading, then Android reading folds were used for participant-level nested logistic regression.
- Main result: frozen DAIC reading ROC-AUC was 0.5795 with all-negative thresholded predictions; Android-trained reading ROC-AUC was 0.8796 with 95 percent CI [0.8113, 0.9377]. The descriptive task comparison recorded task centroid cosine distance 0.0442 and within-participant task cosine similarity 0.9273.
- Significance: Android reading also supported strong within-corpus discrimination after Android-specific training, while zero-shot threshold transfer again failed.
- Limitation: Phase 2D reading/interview comparison was descriptive because task-specific folds and participant sets differed.
- How it motivated the next phase: Phase 2E used paired participants and coherent folds for controlled cross-task transfer.

### Phase 2E: Reading-Interview Cross-Task Transfer

- What was tested: matched reading, matched interview, reading-to-interview transfer, interview-to-reading transfer, pooled-to-reading, pooled-to-interview, and classifier-boundary diagnostics.
- Why it was necessary: the study needed to determine whether reading and interview speech shared useful representation information while preserving participant-level separation.
- Method: participant-coherent nested five-fold cross-validation over 110 paired Android participants.
- Main result: matched reading ROC-AUC was 0.9048; matched interview was 0.9065; reading-to-interview transfer was 0.8475; interview-to-reading transfer was 0.8501; pooled-to-reading was 0.9058; pooled-to-interview was 0.9135. Boundary diagnostics included coefficient cosine similarity 0.4752, sign agreement 0.6659, top-50 Jaccard overlap 0.1216, matched score correlation 0.7123, and cross-target score correlation 0.5555.
- Significance: the results supported shared information across reading and interview representations while showing task-sensitive classifier boundaries.
- Limitation: all Phase 2E protocols were within Androids Corpus, and pooled-task gains were suggestive because gain intervals included zero.
- How it motivated the final interpretation: the study framed WavLM transfer as representation transfer with corpus- and task-specific classifier boundaries, not as deployment evidence.

## Final Experimental Interpretation

The full experimental sequence supports three connected findings. First, DAIC-WOZ development identified a free-speech-compatible WavLM representation but also showed that guided-interview timing features were stronger inside DAIC and less portable. Second, frozen DAIC WavLM scores retained limited ranking information on Android interview and reading speech, but fixed DAIC thresholds and probability scales failed. Third, Android-trained and participant-coherent cross-task models showed that frozen WavLM features were useful within Androids Corpus when classifier boundaries were learned for the relevant corpus or task.

## What the Experiments Demonstrate

- Frozen WavLM layer-9 participant mean features can support research classification analyses in DAIC-WOZ and Androids Corpus.
- Direct DAIC-to-Android zero-shot transfer showed limited ranking signal but failed thresholded classification.
- Target-domain threshold adaptation and score calibration answer different questions from pure zero-shot transfer.
- Android-trained interview and reading classifiers achieved strong within-corpus discrimination under participant-level cross-validation.
- Reading and interview tasks shared useful representation information, but classifier boundaries remained task-sensitive.
- Pooled-task training produced the highest point estimates in Phase 2E, while uncertainty intervals required cautious interpretation.

## What the Experiments Do Not Demonstrate

- They do not establish official AVEC hidden-test performance.
- They do not establish universal generalization across populations, languages, devices, or clinical settings.
- They do not show that Android-trained models are external validation.
- They do not show that threshold-only adaptation or calibration is pure zero-shot transfer.
- They do not establish clinical validation, diagnostic use, screening utility, deployment readiness, or clinical benefit.
- They do not provide causal interpretation of logistic-regression coefficients.

## Clinical and Deployment Boundaries

All experiments are retrospective research analyses. The labels differ across datasets: DAIC-WOZ used PHQ8_Binary labels, while Androids Corpus used healthy-control versus patient group labels. No prospective deployment study, demographic fairness audit, clinical utility study, or clinical workflow evaluation was performed. The results may support future research on speech representations and transfer behavior, but they do not support clinical deployment claims.

## How These Experiments Support the BSPC Manuscript

The BSPC manuscript uses this sequence to justify a central methodological claim: frozen WavLM representations retained useful depression-related information across corpus and task boundaries, while classifier boundaries and probability scales were sensitive to corpus and task conditions. The experiment sequence supports the manuscript's separation between DAIC method development, cross-corpus zero-shot transfer, target-domain adaptation, within-Android benchmarks, and participant-coherent cross-task analysis.

This rationale summary also supports the supplementary package by documenting why each protocol was needed, what claim it can support, and which claims remain prohibited.
