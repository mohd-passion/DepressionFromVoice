# Cross-Corpus Methods Literature Table

This table supports the manuscript's cross-corpus and cross-task transfer framing. It is not an exhaustive review, and it does not treat target-domain training as zero-shot external validation.

| Source | Transfer setting | Speech representation or feature family | Depression label context | Protocol lesson used in manuscript | Limitation for this paper |
| --- | --- | --- | --- | --- | --- |
| Alghowinem et al. [-@alghowinem2016crossCultural] | Cross-cultural depression recognition from vocal biomarkers | Vocal/acoustic biomarkers | Depression-recognition setting across cultural or language conditions | Language and cultural context can matter for depression-speech transfer. | Does not provide DAIC-to-Android metrics or validate WavLM. |
| Seneviratne and Espy-Wilson [-@seneviratne2021generalized] | Cross-corpus depression detection across two depression databases | Inverted vocal tract variables with generalized dilated CNN models | Depression database labels from the source studies | Cross-corpus evaluation is a distinct protocol from within-corpus validation. | arXiv preprint; publication venue not verified. |
| Ravi et al. [-@ravi2022speakerDisentanglement] | DAIC-WOZ and CONVERGE depression-speech modeling with speaker-disentanglement | Wav2vec2-based features and speaker disentanglement | DAIC-WOZ and CONVERGE depression contexts | Cross-dataset depression-speech work often mixes corpus, language and label differences. | Used for methods context; not cited for ethics/privacy conclusions despite the title. |
| Leal et al. [-@leal2025survey] | Survey of speech-based depression assessment across multiple datasets and tasks | Acoustic, prosodic, spectral, voice-quality, timing and learned representations | Heterogeneous depression labels and study designs | Reported depression-speech performance varies with task, population, language, recording and label definition. | Survey-level support only; not a source for this project's metrics. |
| Moreno-Torres et al. [-@morenoTorres2012datasetShift] | General classification under dataset shift | Model-agnostic dataset-shift taxonomy | Not depression-specific | Separates feature-distribution, prior and conditional relationship changes that can affect transfer. | General ML taxonomy; supports framing rather than speech-specific evidence. |
| Yang et al. [-@yang2021superb] | Broad downstream speech benchmark with shared pretrained SSL encoders and task heads | Self-supervised speech representations with lightweight downstream heads | Broad speech tasks, not depression-specific | Supports the methodological idea of frozen or shared encoders with downstream task heads. | Not evidence for clinical depression detection or cross-corpus robustness. |
| Wu et al. [-@wu2023sslDepression] | Speech-based depression detection using SSL representations | Self-supervised speech representations | Depression-speech detection context | SSL representations have been evaluated in depression-speech modeling. | Does not establish DAIC-to-Android transfer or clinical deployment readiness. |
| Huang et al. [-@huang2024voicePretraining] | Depression recognition using voice-based pre-training | wav2vec 2.0-style voice pre-training and fine-tuning | DAIC-WOZ depression-recognition context | Pretrained speech models can be adapted for depression-recognition experiments. | DAIC-WOZ application context only; not cross-corpus validation. |

## Stage 2C Protocol Interpretation

| Protocol in this project | Uses target Android labels? | Score ranking changed? | Probabilities changed? | Classifier boundary refit? | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Phase 2B zero-shot DAIC-to-Android interview | No | No | No | No | Zero-shot external evaluation of the complete frozen DAIC model. |
| Phase 2C Protocol B threshold-only adaptation | Yes, within training folds | No | No | No | Target-domain threshold adaptation, not pure zero-shot evaluation. |
| Phase 2C Protocol C frozen-score calibration | Yes, within training folds | May change under calibration mapping | Yes | No | Target-domain score calibration rather than zero-shot evaluation. |
| Phase 2C Protocol D Android interview classifier | Yes, within training folds | Yes | Yes | Yes | Within-Android classifier retraining on the same frozen representation, not DAIC-to-Android model transfer. |
