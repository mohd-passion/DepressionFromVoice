# Why Each Experiment Was Conducted

## Starting Question

The central question was whether frozen WavLM speech representations could support depression-related classification across corpora and speech tasks. The study also needed to learn whether failures came from the representation itself or from thresholds, calibration and classifier boundaries.

## Step 1: Build a DAIC-WOZ Baseline

The first step used DAIC-WOZ because it was the source-domain development corpus. A metadata-only baseline checked whether trivial recording-duration or window-count information explained the task. A final-layer WavLM baseline checked whether a default frozen speech embedding was useful. Both were limited, so the project moved to representation selection.

## Step 2: Find the Best WavLM Layer

The study then compared WavLM layers 1 through 12. Layer 9 was selected because it gave the strongest recorded development performance among the WavLM layers. This mattered because the same layer was later frozen for Android transfer and task analysis.

## Step 3: Test Aggregation Strategies

WavLM produced window-level embeddings, but depression labels were participant-level. Mean, median and mean plus standard deviation PCA aggregation were compared. Mean aggregation was retained because it was simple and competitive.

## Step 4: Test Timing Features

DAIC-WOZ is a guided-interview corpus, so timing and interaction features could contain useful behavioral information. Timing-only features were stronger than the WavLM-only candidate inside DAIC, but this did not mean they could be transferred to Androids Corpus because Android did not provide the same response-latency structure.

## Step 5: Test Fusion

Fusion tested whether WavLM and timing signals could be combined. Weighted late fusion and logistic stacking did not improve reliability over timing-only. This supported retaining separate candidate roles: timing for DAIC guided-interview analysis and WavLM for free-speech-compatible transfer.

## Step 6: Freeze the DAIC Model

The final DAIC evaluation froze two candidates before the provisional local labeled test. The timing candidate performed better on DAIC, but the WavLM candidate was the transferable one. This is why Android experiments used WavLM rather than DAIC timing features.

## Step 7: Test Zero-Shot Transfer to Androids Corpus

The frozen DAIC WavLM classifier was applied directly to Android interview speech. It achieved ROC-AUC 0.6412, which suggested some ranking information transferred. However, its thresholds predicted every Android participant as negative. This showed that direct thresholded classification did not transfer well.

## Step 8: Diagnose Domain Shift

The next step separated possible explanations. Threshold adaptation tested whether the cutoff needed to move. Calibration tested whether probabilities needed remapping. Android-trained logistic regression tested whether a new classifier boundary was needed. The Android-trained model performed much better, supporting the interpretation that the representation was useful but the DAIC boundary did not transfer.

## Step 9: Train Android-Specific Models

Android-trained interview and reading models used the same frozen WavLM representation but learned target-domain classifier boundaries. These within-Android models achieved strong ROC-AUC values, indicating that the representation contained useful information in Android speech.

## Step 10: Compare Reading and Interview Speech

Reading speech is controlled, while interview speech is more spontaneous. The study compared both because a robust speech representation should not be understood from one task alone. Phase 2D showed strong Android-trained reading performance and descriptive similarity between reading and interview embeddings.

## Step 11: Test Cross-Task Transfer

Phase 2E used paired participants and coherent folds. Reading-to-interview and interview-to-reading transfer both worked reasonably well but were below matched-task references. Pooled reading-and-interview training produced the strongest point estimates, though the gain intervals included zero.

## Final Takeaway

The experiments suggest that frozen WavLM features can retain depression-related information across corpus and task changes, but classifier boundaries and probability scales are sensitive to dataset and task conditions. The work supports research interpretation, not clinical deployment.
