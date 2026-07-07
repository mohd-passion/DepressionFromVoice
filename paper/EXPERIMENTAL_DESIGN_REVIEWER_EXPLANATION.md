# Reviewer Explanation of Experimental Design

## Why the Study Was Designed in Stages

The study was staged because each experiment answered a different methodological question. DAIC-WOZ was used first to build and freeze candidate representations under controlled development conditions. Androids Corpus was then used to test transfer across corpus, language, label definition, and speech task. Later stages separated zero-shot transfer, target-domain threshold adaptation, calibration, target-domain classifier learning, and cross-task transfer so that the manuscript would not conflate distinct forms of evidence.

This design also reduced interpretation risk. A single high within-corpus result would not show external transfer, and a failed zero-shot threshold result would not necessarily mean that the representation was useless. The staged workflow therefore allowed the paper to separate representation quality from classifier-boundary portability.

## Why DAIC-WOZ Was Used for Initial Method Development

DAIC-WOZ provided the first controlled development setting, with participant-level PHQ8_Binary labels, official train/development splits, and a provisional local labeled test split. It allowed the project to compare metadata, frozen WavLM embeddings, guided-interview timing features, aggregation strategies, and fusion protocols before touching Android transfer results.

Using DAIC-WOZ first was important because the study needed a frozen source-domain candidate. The layer-9 WavLM representation and mean aggregation were selected before Android evaluation. This kept Android experiments from becoming a repeated search for a desired external result.

## Why Androids Corpus Was Used for External Task and Corpus Analysis

Androids Corpus differs from DAIC-WOZ in language, label definition, recording/task structure, and participant population. These differences made it a useful target for evaluating whether a DAIC-trained WavLM classifier could transfer beyond its source corpus.

The corpus also contains both reading and interview speech. That enabled a second question beyond cross-corpus transfer: whether depression-related information in frozen WavLM features was shared across speech tasks inside the same target corpus.

## Why Zero-Shot Transfer Was Tested Before Adaptation

Zero-shot transfer asks whether the complete frozen DAIC model, including its learned classifier boundary and threshold behavior, can be applied directly to Android speech. This had to be tested before adaptation because target-domain labels change the question.

The zero-shot Android interview result reached ROC-AUC 0.6412 but predicted all Android participants as negative at the frozen thresholds. The zero-shot Android reading result similarly retained modest ranking but failed thresholded classification. These results motivated a deeper diagnosis rather than an immediate claim of success or failure.

## Why Calibration and Threshold Adaptation Were Separated

Threshold adaptation and calibration answer related but distinct questions. Threshold adaptation asks whether a decision cutoff can be moved using target-domain training folds while preserving the frozen score ranking. Calibration asks whether probability values can be remapped to target-domain outcomes.

In Phase 2C, threshold-only adaptation retained ROC-AUC 0.6412 while improving thresholded metrics. Frozen-score calibration improved Brier score to 0.2474 but reduced ROC-AUC to 0.5694. Separating the two protocols prevented the manuscript from labeling target-adapted results as pure zero-shot transfer.

## Why Android-Trained Models Were Reported Separately

Android-trained models answer whether the frozen WavLM representation supports a new target-domain classifier boundary. They do not answer whether the DAIC classifier itself transferred.

The Android-trained interview model achieved ROC-AUC 0.8675, and the Android-trained reading model achieved ROC-AUC 0.8796. These results supported the representation-transfer interpretation: useful information was present in the frozen features, but the decision boundary needed target-domain learning.

## Why Reading and Interview Speech Were Compared

Reading and interview speech differ in linguistic constraint, interaction structure, spontaneity, and timing. The study compared them because a representation useful only for one task would have narrower research value.

Phase 2D first provided task-specific benchmarks and descriptive task-shift statistics. Phase 2E then used participant-coherent folds across 110 paired participants to evaluate cross-task transfer without leakage from the same participant appearing in both train and test folds.

## Why Cross-Task Transfer Was Evaluated

Cross-task transfer tested whether a classifier trained on one Android task could evaluate the other task for held-out participants. Reading-to-interview transfer reached ROC-AUC 0.8475, and interview-to-reading transfer reached ROC-AUC 0.8501. These results indicated shared information, but both were below the matched-task point estimates.

The pooled-task protocols then tested whether training with both reading and interview examples helped. Pooled-to-reading reached ROC-AUC 0.9058, and pooled-to-interview reached ROC-AUC 0.9135. The point estimates were strongest in the pooled protocols, but the gain intervals included zero, so the result is presented cautiously.

## Why Timing Features Were Treated Carefully

DAIC timing features were useful inside the guided-interview DAIC setting, where timing and response-latency definitions were available. However, Androids Corpus did not provide directly compatible timing features for the same DAIC timing model.

The study therefore did not transfer the DAIC timing model to Android. Instead, timing results were used to explain DAIC methodology, while WavLM was retained as the free-speech-compatible representation for Android transfer and task analysis.

## Why the Study Remains Research-Only

The experiments are retrospective and dataset-bound. DAIC-WOZ and Androids Corpus use different labels, languages, and elicitation contexts. No prospective clinical evaluation, deployment study, demographic fairness audit, or clinical utility assessment was performed.

For that reason, the manuscript frames the results as evidence about frozen speech representations, domain shift, calibration, classifier-boundary shift, and cross-task transfer. It does not claim clinical validation, diagnostic use, screening utility, or deployment readiness.
