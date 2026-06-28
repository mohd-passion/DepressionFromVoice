# Phase 2 Evaluation Protocols

Phase 2A prepared protocols only. It did not extract WavLM embeddings, train models, apply frozen DAIC models, calculate Android prediction metrics, or tune thresholds.

## Protocol A: Zero-Shot DAIC WavLM To Android Interview

Apply frozen DAIC Candidate A without retraining to Android interview participant speech only. Use identical 10-second windows, 50% overlap, WavLM layer 9, temporal mean pooling, participant mean aggregation, and frozen DAIC classifier. Report probability-based metrics. Threshold-based results should be treated cautiously because Android clinical-group labels and prevalence differ from DAIC PHQ-8 labels.

## Protocol B: Zero-Shot DAIC WavLM To Android Reading

Process reading speech separately. Do not claim equivalence to DAIC interview speech. Use as a task-shift robustness experiment.

## Protocol C: Zero-Shot DAIC Timing Model To Android Interview

Status: `not_valid_without_feature_redefinition`.

The frozen DAIC timing model may be applied only if all 28 timing features can be reproduced with matching definitions, units, ordering, and missing-value behavior. Phase 2A found Android response-latency features unavailable because interviewer turns are not represented in `interview_timedata.csv`; therefore direct Candidate B transfer is invalid unless future work formally redefines features and does not pretend to use the frozen DAIC timing model unchanged.

## Protocol D: Within-Android WavLM Baseline

Use Android's official five-fold participant-independent protocol. Train and evaluate exclusively within Android. Keep reading and interview tasks separate. This is a future baseline and is not implemented in Phase 2A.
