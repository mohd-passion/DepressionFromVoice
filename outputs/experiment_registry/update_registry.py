"""Update the DepressionFromVoice experiment registry.

The registry is intentionally compact:

  - experiment_registry.json is the authoritative record.
  - experiment_summary.csv is the comparison table.
  - EXPERIMENT_LOG.md is generated from the JSON.
  - this script is the only update mechanism.

This script records completed experiments only. It does not train models,
generate predictions, rerun experiments, tune thresholds, or modify frozen
pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_VERSION = "1.0"
PROJECT = "DepressionFromVoice"

OFFICIAL_RUNS = {
    "phase1a": Path("outputs/phase1_daic/phase1_participant_baseline_runs/20260622_114625"),
    "phase1b": Path("outputs/phase1_daic/phase1b_layerwise_runs/20260622_123725"),
    "phase1c": Path("outputs/phase1_daic/phase1c_aggregation_runs/20260622_135800"),
    "phase1d": Path("outputs/phase1_daic/phase1d_timing_fusion_runs/20260622_142132"),
    "phase1e": Path("outputs/phase1_daic/phase1e_late_fusion_runs/20260622_145240"),
    "phase1f": Path("outputs/phase1_daic/phase1f_final_evaluation_runs/20260622_151118"),
    "phase2a": Path("outputs/phase2_android/phase2a_preparation_runs/20260622_154228"),
    "phase2b": Path("outputs/phase2_android/phase2b_zero_shot_interview_runs/20260622_162548"),
}

SUPERSEDED_RUNS = [
    {
        "run_directory": "outputs/phase2_android/phase2a_preparation_runs/20260622_154008",
        "phase": "Phase 2A",
        "status": "superseded",
        "authoritative": False,
        "reason": "Fold parsing was corrected to preserve (task, recording_id) identity.",
        "replacement_run_directory": "outputs/phase2_android/phase2a_preparation_runs/20260622_154228",
    }
]

ACCURACY_HUNTING_WARNING = (
    "Do not pursue desired accuracy by repeatedly evaluating the same test set. "
    "Use training and validation data for model development; freeze decisions before test evaluation; "
    "record every inspected test result; document decisions made after viewing test performance; "
    "judge improvements using cross-validation stability, confidence intervals and external validation."
)

CSV_FIELDS = [
    "experiment_id",
    "phase",
    "experiment_name",
    "status",
    "result_scope",
    "dataset",
    "task",
    "representation",
    "wavlm_layer",
    "aggregation",
    "classifier",
    "selected_C",
    "train_cv_auc",
    "train_cv_auc_std",
    "dev_auc",
    "test_auc",
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "threshold",
    "evaluation_type",
    "clinical_status",
    "selected",
    "main_finding",
    "paper_ready_result_sentence",
    "limitation",
    "run_directory",
]

RESULT_SCOPES = {
    "methodological_ablation",
    "within_corpus_cross_validation",
    "within_corpus_held_out_test",
    "provisional_local_labeled_test",
    "cross_corpus_zero_shot_external_evaluation",
    "cross_corpus_target_adaptation",
    "cross_task_transfer",
    "pooled_task_within_corpus_cross_validation",
    "data_quality_audit",
}

PAPER_INTERPRETATION_FIELDS = [
    "result_scope",
    "training_dataset",
    "evaluation_dataset",
    "training_task",
    "evaluation_task",
    "evaluation_protocol",
    "participant_level_split",
    "target_labels_used_for_training",
    "target_labels_used_for_model_selection",
    "target_labels_used_for_threshold_selection",
    "external_validation",
    "cross_corpus",
    "zero_shot",
    "target_domain_refitting",
    "calibration_performed",
    "threshold_adaptation_performed",
    "primary_metric_name",
    "primary_metric_value",
    "confidence_interval",
    "paper_ready_result_sentence",
    "allowed_claims",
    "prohibited_claims",
    "comparison_caveats",
    "clinical_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return registry_dir().parents[1]


def rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(project_root().resolve()))
    except ValueError:
        return str(p)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def get_path(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def base_registry(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = existing or {}
    registry.update(
        {
            "registry_version": REGISTRY_VERSION,
            "project": PROJECT,
            "last_updated": utc_now(),
            "project_status": {
                "phase1_status": "completed_baseline_ready_for_external_validation",
                "phase2_status": "phase2e_cross_task_transfer_completed",
                "current_scope": "research experiment tracking only",
                "accuracy_hunting_warning": ACCURACY_HUNTING_WARNING,
            },
            "datasets": dataset_records(),
            "current_candidates": current_candidates(),
            "research_decisions": research_decisions(),
            "planned_experiments": planned_experiments(),
            "superseded_runs": SUPERSEDED_RUNS,
            "accuracy_hunting_warning": ACCURACY_HUNTING_WARNING,
        }
    )
    registry.setdefault("experiments", [])
    registry.setdefault("validation", {})
    return registry


def dataset_records() -> dict[str, Any]:
    return {
        "daic_woz_phase1_clean": {
            "name": "DAIC-WOZ clean Phase 1 dataset",
            "usable_participants": 187,
            "splits": {"train": 106, "dev": 34, "local_labeled_test": 47},
            "excluded_participants": [409, 422],
            "labels": "PHQ8_Binary",
            "label_provenance": "provisional local labeled test for test split",
            "speech_scope": "participant-only speech",
            "windowing": {"window_sec": 10, "overlap": 0.5, "total_windows": 17058},
            "language": "English",
        },
        "androids_corpus": {
            "name": "Androids Corpus",
            "participants": 118,
            "class_counts": {"HC": 54, "PT": 64},
            "recordings": {"reading": 112, "interview": 116, "interview_clips": 874},
            "audio_quality": "all 1102 audio files readable",
            "sample_rate_hz": 44100,
            "labels": "clinical patient/control grouping",
            "language": "Italian",
            "fold_warning": "cross-task fold inconsistency affecting 85 participants",
        },
    }


def current_candidates() -> dict[str, Any]:
    return {
        "daic_free_speech_candidate": {
            "description": "Free-speech-compatible DAIC candidate",
            "wavlm_model": "microsoft/wavlm-base-plus",
            "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b",
            "layer": "layer_09",
            "participant_aggregation": "mean",
            "classifier": "StandardScaler + LogisticRegression",
            "C": 0.001,
            "threshold": 0.555840134621,
            "status": "frozen_research_candidate",
        },
        "daic_guided_interview_candidate": {
            "description": "Guided-interview timing candidate",
            "features": "28 validated timing features",
            "classifier": "SimpleImputer + StandardScaler + LogisticRegression",
            "C": 100.0,
            "threshold": 0.2927276896337141,
            "status": "frozen_research_candidate",
        },
        "external_validation_result": {
            "description": "Frozen DAIC WavLM on Android interview speech",
            "roc_auc": 0.6412259615384616,
            "pr_auc": 0.6808532952868427,
            "threshold_transfer": "failed; all participants predicted negative",
            "status": "weak_transfer_with_ranking_signal",
            "not_universal_best_model": True,
        },
    }


def research_decisions() -> list[dict[str, Any]]:
    return [
        {"phase": "Phase 1B", "decision": "Selected WavLM layer_09 using train-only nested CV.", "test_used": False},
        {"phase": "Phase 1C", "decision": "Kept participant mean aggregation; alternate aggregation did not improve enough.", "test_used": False},
        {"phase": "Phase 1D", "decision": "Timing-only outperformed WavLM and early fusion on train/dev.", "test_used": False},
        {"phase": "Phase 1E", "decision": "Late fusion did not improve reliability; timing-only retained.", "test_used": False},
        {"phase": "Phase 1F", "decision": "Closed Phase 1 as frozen research candidates for external validation.", "test_used": True},
        {"phase": "Phase 2A", "decision": "Timing model direct transfer to Android invalid; WavLM interview preprocessing feasible.", "test_used": False},
        {"phase": "Phase 2B", "decision": "Zero-shot WavLM ranking signal observed, but frozen threshold failed on Android.", "test_used": True},
    ]


def planned_experiments() -> list[dict[str, Any]]:
    return [
        {
            "id": "phase2c_android_adaptation_analysis",
            "phase": "Phase 2C",
            "status": "completed_by_phase2c_registry_entries",
            "protocols": [
                "frozen DAIC zero-shot reference",
                "threshold-only adaptation",
                "probability calibration",
                "Android-trained logistic regression using fixed layer-9 embeddings",
            ],
            "completed_by": "Phase 2C registry entries when a phase2c_domain_shift run is imported",
            "purpose": [
                "distinguish calibration shift",
                "distinguish classifier-boundary shift",
                "distinguish representation limitation",
            ],
            "metrics": None,
            "run_directory": None,
        }
    ]


def empty_entry(exp_id: str, phase: str, name: str, status: str, run_dir: Path | str) -> dict[str, Any]:
    return {
        "id": exp_id,
        "revision": 1,
        "phase": phase,
        "name": name,
        "status": status,
        "run_directory": rel(run_dir),
        "parent_id": None,
        "data": {
            "dataset_id": "",
            "task": "",
            "language": "",
            "train_n": None,
            "dev_n": None,
            "test_n": None,
            "class_counts": {},
            "split_protocol": "",
            "excluded": [],
        },
        "representation": {
            "feature_family": "",
            "model": "",
            "model_commit": "",
            "layer": None,
            "window_sec": None,
            "overlap": None,
            "window_pooling": "",
            "participant_aggregation": "",
            "dimension": None,
            "timing_feature_count": None,
            "fusion": "",
        },
        "classifier": {
            "name": "",
            "pipeline": "",
            "class_weight": "",
            "C": None,
            "other_parameters": {},
            "selection_protocol": "",
        },
        "threshold": {"source": "", "value": None, "tuned_on": ""},
        "metrics": {"train_cv": {}, "dev": {}, "test": {}, "confidence_intervals": {}},
        "integrity": {
            "test_provenance": "",
            "test_used_for_selection": False,
            "external_validation": False,
            "zero_shot": False,
            "target_model_refitted": False,
            "target_threshold_refitted": False,
            "leakage_status": "",
            "test_result_viewed": False,
            "decisions_made_after_test_view": [],
            "evaluation_reuse_warning": ACCURACY_HUNTING_WARNING,
        },
        "conclusion": {
            "main_finding": "",
            "selected": False,
            "selection_reason": "",
            "rejection_reason": "",
            "limitations": [],
            "warnings": [],
            "next_step": "",
        },
        "artifacts": {"report_json": "", "report_markdown": "", "predictions": "", "model": "", "manifest": ""},
        "environment": {
            "python": None,
            "pytorch": None,
            "transformers": None,
            "scikit_learn": None,
            "cuda": None,
            "gpu": None,
            "git_commit": None,
        },
        "history": [],
    }


def phase1_data() -> dict[str, Any]:
    return {
        "dataset_id": "daic_woz_phase1_clean",
        "task": "DAIC-WOZ participant speech",
        "language": "English",
        "train_n": 106,
        "dev_n": 34,
        "test_n": 47,
        "class_counts": {
            "train": {"non_depressed": 76, "depressed": 30},
            "dev": {"non_depressed": 23, "depressed": 11},
            "local_labeled_test": {"non_depressed": 33, "depressed": 14},
        },
        "split_protocol": "official train/dev plus provisional local labeled test",
        "excluded": [409, 422],
    }


def apply_phase1_common(entry: dict[str, Any]) -> None:
    entry["data"] = phase1_data()
    entry["integrity"].update(
        {
            "test_provenance": "provisional local labeled test when test metrics are present",
            "test_used_for_selection": False,
            "external_validation": False,
            "zero_shot": False,
            "target_model_refitted": False,
            "target_threshold_refitted": False,
            "leakage_status": "train/dev model selection separated from test where applicable",
        }
    )


def dataset_label(dataset_id: str) -> str:
    labels = {
        "daic_woz_phase1_clean": "DAIC-WOZ",
        "androids_corpus": "Androids Corpus",
    }
    return labels.get(dataset_id, dataset_id or "not_applicable")


def first_metric(exp: dict[str, Any], key: str = "roc_auc") -> float | None:
    for section in ["test", "dev", "train_cv"]:
        value = get_path(exp, "metrics", section, key)
        numeric = fnum(value)
        if numeric is not None:
            return numeric
    return None


def fmt_metric(value: Any) -> str:
    numeric = fnum(value)
    if numeric is None:
        return "not_reported"
    return f"{numeric:.4f}"


def paper_defaults(exp: dict[str, Any]) -> dict[str, Any]:
    dataset = dataset_label(str(exp.get("data", {}).get("dataset_id", "")))
    task = str(exp.get("data", {}).get("task", ""))
    metric = first_metric(exp)
    return {
        "result_scope": "methodological_ablation",
        "training_dataset": dataset,
        "evaluation_dataset": dataset,
        "training_task": task or "not_applicable",
        "evaluation_task": task or "not_applicable",
        "evaluation_protocol": str(exp.get("data", {}).get("split_protocol", "")) or "recorded experiment protocol",
        "participant_level_split": str(exp.get("data", {}).get("split_protocol", "")) or "participant-level split recorded where applicable",
        "target_labels_used_for_training": bool(exp.get("classifier", {}).get("name") not in {"none", ""}),
        "target_labels_used_for_model_selection": bool(exp.get("metrics", {}).get("train_cv")),
        "target_labels_used_for_threshold_selection": bool(exp.get("threshold", {}).get("source")),
        "external_validation": False,
        "cross_corpus": False,
        "zero_shot": False,
        "target_domain_refitting": bool(exp.get("classifier", {}).get("name") not in {"none", ""}),
        "calibration_performed": False,
        "threshold_adaptation_performed": bool(exp.get("integrity", {}).get("target_threshold_refitted", False)),
        "primary_metric_name": "ROC-AUC" if metric is not None else "not_applicable",
        "primary_metric_value": metric,
        "confidence_interval": "not_reported",
        "paper_ready_result_sentence": f"{exp.get('name', exp.get('id'))} was evaluated under its recorded protocol; interpret the result only within that methodological scope.",
        "allowed_claims": ["research result within the stated protocol"],
        "prohibited_claims": ["clinical diagnosis", "deployment readiness", "causal claims", "universal model superiority"],
        "comparison_caveats": ["Compare only with experiments using compatible data splits, tasks and target definitions."],
        "clinical_status": "not_clinically_validated",
    }


def _phase1_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    exp_id = exp["id"]
    metric = fmt_metric(p["primary_metric_value"])
    p.update(
        {
            "training_dataset": "DAIC-WOZ",
            "evaluation_dataset": "DAIC-WOZ",
            "external_validation": False,
            "cross_corpus": False,
            "zero_shot": False,
            "allowed_claims": ["DAIC-WOZ research estimate under the recorded split and selection protocol"],
            "prohibited_claims": [
                "official AVEC hidden-test performance",
                "cross-corpus generalization",
                "clinical diagnosis",
                "deployment readiness",
            ],
            "clinical_status": "not_clinically_validated",
        }
    )
    if exp["phase"] in {"Phase 1B", "Phase 1C", "Phase 1D", "Phase 1E"}:
        p["result_scope"] = "methodological_ablation"
        p["evaluation_protocol"] = str(exp["classifier"].get("selection_protocol", "train-only model-development protocol"))
        p["participant_level_split"] = "DAIC train/dev development split; no Phase 1 test evaluation for this record"
        p["target_labels_used_for_training"] = True
        p["target_labels_used_for_model_selection"] = True
        p["target_labels_used_for_threshold_selection"] = bool(exp.get("metrics", {}).get("dev"))
        p["target_domain_refitting"] = True
        p["paper_ready_result_sentence"] = (
            f"{exp['name']} was a DAIC-WOZ methodological ablation with ROC-AUC {metric} under the recorded train/dev protocol; "
            "it should not be presented as external validation."
        )
        p["comparison_caveats"] = ["Ablation results support method selection only within DAIC-WOZ development data."]
    elif exp["phase"] == "Phase 1A":
        p["result_scope"] = "provisional_local_labeled_test"
        p["participant_level_split"] = "DAIC train/dev plus provisional local labeled test split"
        p["target_labels_used_for_training"] = True
        p["target_labels_used_for_model_selection"] = True
        p["target_labels_used_for_threshold_selection"] = True
        p["target_domain_refitting"] = True
        p["paper_ready_result_sentence"] = (
            f"{exp['name']} achieved ROC-AUC {metric} on the provisional local labeled DAIC test split; "
            "this was not an official hidden AVEC test evaluation."
        )
        p["comparison_caveats"] = ["DAIC local labeled test labels are provisional and must not be described as official AVEC hidden-test labels."]
    elif exp["phase"] == "Phase 1F":
        p["result_scope"] = "provisional_local_labeled_test"
        p["evaluation_protocol"] = "Frozen Phase 1 candidate evaluated once on provisional local labeled DAIC test labels from Data/full_test_split.csv"
        p["participant_level_split"] = "DAIC train/dev plus provisional local labeled test split"
        p["target_labels_used_for_training"] = True
        p["target_labels_used_for_model_selection"] = True
        p["target_labels_used_for_threshold_selection"] = True
        p["target_domain_refitting"] = True
        p["paper_ready_result_sentence"] = (
            f"{exp['name']} achieved ROC-AUC {metric} on the provisional local labeled DAIC test split from Data/full_test_split.csv; "
            "this was not the official hidden AVEC test evaluation and no post-test model changes were permitted."
        )
        p["comparison_caveats"] = ["Report as provisional local labeled DAIC test performance only."]
    if exp_id == "phase1f_frozen_wavlm":
        p["training_task"] = "free-speech-compatible participant speech"
        p["evaluation_task"] = "free-speech-compatible participant speech"
    elif exp_id == "phase1f_frozen_timing":
        p["training_task"] = "guided-interview timing features"
        p["evaluation_task"] = "guided-interview timing features"


def _phase2a_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    p.update(
        {
            "result_scope": "data_quality_audit",
            "training_dataset": "not_applicable",
            "evaluation_dataset": "Androids Corpus",
            "training_task": "not_applicable",
            "evaluation_task": "data inventory and compatibility audit",
            "evaluation_protocol": "dataset preparation and DAIC-to-Android compatibility audit only",
            "participant_level_split": "not_applicable_no_modeling",
            "target_labels_used_for_training": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_threshold_selection": False,
            "external_validation": False,
            "cross_corpus": False,
            "zero_shot": False,
            "target_domain_refitting": False,
            "calibration_performed": False,
            "threshold_adaptation_performed": False,
            "primary_metric_name": "not_applicable",
            "primary_metric_value": None,
            "paper_ready_result_sentence": "Phase 2A was a data-quality and compatibility audit; it did not train or evaluate a classifier.",
            "allowed_claims": ["Androids Corpus inventory and compatibility findings"],
            "prohibited_claims": ["model performance", "clinical diagnosis", "external validation"],
            "comparison_caveats": ["No modeling was performed in this audit."],
            "clinical_status": "not_applicable_no_modeling",
        }
    )


def _phase2b_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    p.update(
        {
            "result_scope": "cross_corpus_zero_shot_external_evaluation",
            "training_dataset": "DAIC-WOZ",
            "evaluation_dataset": "Androids Corpus",
            "training_task": "interview/free speech",
            "evaluation_task": "interview",
            "evaluation_protocol": "Frozen DAIC-WOZ classifier applied once to Android interview participant embeddings",
            "participant_level_split": "all eligible Android interview participants evaluated once; no Android fitting",
            "target_labels_used_for_training": False,
            "target_labels_used_for_model_selection": False,
            "target_labels_used_for_threshold_selection": False,
            "external_validation": True,
            "cross_corpus": True,
            "zero_shot": True,
            "target_domain_refitting": False,
            "calibration_performed": False,
            "threshold_adaptation_performed": False,
            "paper_ready_result_sentence": "The frozen DAIC-WOZ classifier achieved a zero-shot ROC-AUC of 0.6412 on Android interview speech, indicating limited cross-corpus ranking transfer; however, its frozen decision thresholds predicted every participant as negative.",
            "allowed_claims": ["limited ranking information transferred across corpora"],
            "prohibited_claims": ["successful cross-corpus classification", "strong external validation", "clinical usefulness", "clinical diagnosis"],
            "comparison_caveats": ["DAIC PHQ8_Binary and Android clinical grouping are related but not identical targets."],
            "clinical_status": "not_clinically_validated",
        }
    )


def _phase2c_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    exp_id = exp["id"]
    p.update(
        {
            "training_dataset": "DAIC-WOZ" if exp_id != "phase2c_protocol_d_android_lr" else "Androids Corpus",
            "evaluation_dataset": "Androids Corpus",
            "training_task": "interview/free speech" if exp_id != "phase2c_protocol_d_android_lr" else "interview",
            "evaluation_task": "interview",
            "participant_level_split": "official Android interview participant-level five-fold protocol",
            "comparison_caveats": ["Android labels and DAIC PHQ8_Binary are related but not identical targets."],
            "clinical_status": "not_clinically_validated",
        }
    )
    if exp_id == "phase2c_protocol_a_frozen_daic_reference":
        p.update(
            {
                "result_scope": "cross_corpus_zero_shot_external_evaluation",
                "evaluation_protocol": "Protocol A: frozen DAIC model, no Android fitting or adaptation",
                "target_labels_used_for_training": False,
                "target_labels_used_for_model_selection": False,
                "target_labels_used_for_threshold_selection": False,
                "external_validation": True,
                "cross_corpus": True,
                "zero_shot": True,
                "target_domain_refitting": False,
                "paper_ready_result_sentence": "Protocol A reproduced the frozen DAIC-to-Android interview zero-shot ROC-AUC of 0.6412 without Android fitting or adaptation.",
                "allowed_claims": ["limited zero-shot ranking transfer"],
                "prohibited_claims": ["successful target-domain classification", "target-domain adaptation", "clinical diagnosis"],
            }
        )
    elif exp_id == "phase2c_protocol_b_threshold_only":
        p.update(
            {
                "result_scope": "cross_corpus_target_adaptation",
                "evaluation_protocol": "Protocol B: frozen DAIC scores with Android training-fold threshold adaptation",
                "target_labels_used_for_training": False,
                "target_labels_used_for_model_selection": False,
                "target_labels_used_for_threshold_selection": True,
                "external_validation": True,
                "cross_corpus": True,
                "zero_shot": False,
                "target_domain_refitting": False,
                "threshold_adaptation_performed": True,
                "paper_ready_result_sentence": "Protocol B adapted decision thresholds using Android training folds while preserving the frozen DAIC score ranking; therefore it was target-domain threshold adaptation, not pure zero-shot evaluation.",
                "allowed_claims": ["threshold shift contributed to DAIC-to-Android transfer failure"],
                "prohibited_claims": ["pure zero-shot evaluation", "new Android classifier training", "clinical diagnosis"],
            }
        )
    elif exp_id == "phase2c_protocol_c_score_calibration":
        p.update(
            {
                "result_scope": "cross_corpus_target_adaptation",
                "evaluation_protocol": "Protocol C: frozen DAIC scores calibrated using Android training folds",
                "target_labels_used_for_training": True,
                "target_labels_used_for_model_selection": True,
                "target_labels_used_for_threshold_selection": False,
                "external_validation": True,
                "cross_corpus": True,
                "zero_shot": False,
                "target_domain_refitting": True,
                "calibration_performed": True,
                "paper_ready_result_sentence": "Protocol C calibrated frozen DAIC scores using Android training folds; Brier score improved but ROC-AUC decreased to 0.5694, so calibration did not improve ranking.",
                "allowed_claims": ["target-domain calibration improved probability scale under the recorded folds"],
                "prohibited_claims": ["pure zero-shot evaluation", "improved ranking transfer", "clinical diagnosis"],
            }
        )
    elif exp_id == "phase2c_protocol_d_android_lr":
        p.update(
            {
                "result_scope": "within_corpus_cross_validation",
                "evaluation_protocol": "Protocol D: Android-trained interview logistic regression under participant-level nested cross-validation",
                "target_labels_used_for_training": True,
                "target_labels_used_for_model_selection": True,
                "target_labels_used_for_threshold_selection": True,
                "external_validation": False,
                "cross_corpus": False,
                "zero_shot": False,
                "target_domain_refitting": True,
                "threshold_adaptation_performed": True,
                "paper_ready_result_sentence": "The Android-trained interview classifier achieved an out-of-fold ROC-AUC of 0.8675 under nested participant-level cross-validation. This demonstrates strong within-Android discrimination after learning an Android-specific classifier boundary, not direct DAIC-to-Android model transfer.",
                "allowed_claims": ["strong within-Android interview discrimination under participant-level cross-validation"],
                "prohibited_claims": ["direct DAIC-to-Android model transfer", "external validation", "clinical diagnosis"],
            }
        )


def _phase2d_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    exp_id = exp["id"]
    p.update(
        {
            "evaluation_dataset": "Androids Corpus",
            "participant_level_split": "official Android task-specific five-fold participant protocol",
            "clinical_status": "not_clinically_validated",
            "comparison_caveats": [
                "Phase 2C interview and Phase 2D reading experiments used task-specific fold assignments.",
                "Participant sets were not identical; direct task differences are descriptive.",
            ],
        }
    )
    if exp_id == "phase2d_android_reading_zero_shot_daic":
        p.update(
            {
                "result_scope": "cross_corpus_zero_shot_external_evaluation",
                "training_dataset": "DAIC-WOZ",
                "training_task": "interview/free speech",
                "evaluation_task": "reading",
                "evaluation_protocol": "Frozen DAIC-WOZ classifier applied to Android reading embeddings",
                "target_labels_used_for_training": False,
                "target_labels_used_for_model_selection": False,
                "target_labels_used_for_threshold_selection": False,
                "external_validation": True,
                "cross_corpus": True,
                "zero_shot": True,
                "target_domain_refitting": False,
                "paper_ready_result_sentence": "The frozen DAIC-WOZ classifier achieved a zero-shot ROC-AUC of 0.5795 on Android reading speech, but its frozen thresholds predicted every participant as negative.",
                "allowed_claims": ["modest zero-shot ranking signal on Android reading speech"],
                "prohibited_claims": ["successful cross-corpus classification", "Android-trained result", "clinical diagnosis"],
            }
        )
    elif exp_id == "phase2d_android_reading_logistic_regression":
        p.update(
            {
                "result_scope": "within_corpus_cross_validation",
                "training_dataset": "Androids Corpus",
                "training_task": "reading",
                "evaluation_task": "reading",
                "evaluation_protocol": "participant-level nested five-fold cross-validation using official reading folds",
                "target_labels_used_for_training": True,
                "target_labels_used_for_model_selection": True,
                "target_labels_used_for_threshold_selection": True,
                "external_validation": False,
                "cross_corpus": False,
                "zero_shot": False,
                "target_domain_refitting": True,
                "threshold_adaptation_performed": True,
                "confidence_interval": "[0.8113, 0.9377]",
                "paper_ready_result_sentence": "The Android-trained reading-speech classifier achieved an out-of-fold ROC-AUC of 0.8796 under participant-level nested cross-validation within the Androids Corpus. This is a within-corpus discrimination estimate and does not constitute external validation or cross-corpus generalization.",
                "allowed_claims": ["strong within-Android reading-speech discrimination under participant-level cross-validation"],
                "prohibited_claims": ["external validation", "cross-corpus generalization", "clinical diagnosis"],
            }
        )
    else:
        p.update(
            {
                "result_scope": "within_corpus_cross_validation",
                "training_dataset": "Androids Corpus",
                "training_task": "reading and interview",
                "evaluation_task": "reading and interview",
                "evaluation_protocol": "descriptive comparison of completed within-Android task-specific protocols",
                "target_labels_used_for_training": True,
                "target_labels_used_for_model_selection": True,
                "target_labels_used_for_threshold_selection": True,
                "external_validation": False,
                "cross_corpus": False,
                "zero_shot": False,
                "target_domain_refitting": True,
                "threshold_adaptation_performed": True,
                "paper_ready_result_sentence": "Phase 2D compared Android-trained reading and interview classifiers descriptively; the task-specific folds and participant sets differed, so this comparison is not a controlled cross-task transfer test.",
                "allowed_claims": ["descriptive comparison of within-Android reading and interview benchmarks"],
                "prohibited_claims": ["controlled cross-task transfer", "external validation", "clinical diagnosis"],
            }
        )


def _phase2e_interpretation(exp: dict[str, Any], p: dict[str, Any]) -> None:
    exp_id = exp["id"]
    ci = {
        "phase2e_matched_reading_within_task": "[0.8435, 0.9562]",
        "phase2e_matched_interview_within_task": "[0.8452, 0.9589]",
        "phase2e_reading_to_interview_transfer": "[0.7699, 0.9158]; gap versus matched interview [-0.1316, 0.0116]",
        "phase2e_interview_to_reading_transfer": "[0.7699, 0.9204]; gap versus matched reading [-0.1436, 0.0325]",
        "phase2e_pooled_to_reading": "[0.8442, 0.9539]; gain versus interview-to-reading [-0.0172, 0.1340]",
        "phase2e_pooled_to_interview": "[0.8515, 0.9635]; gain versus reading-to-interview [-0.0013, 0.1346]",
        "phase2e_cross_task_transfer_conclusion": "paired bootstrap intervals recorded in phase2e_paired_comparisons.csv",
    }
    p.update(
        {
            "training_dataset": "Androids Corpus",
            "evaluation_dataset": "Androids Corpus",
            "evaluation_protocol": "Phase 2E participant-coherent nested five-fold cross-validation over 110 paired Androids Corpus participants",
            "participant_level_split": "new participant-coherent five-fold split shared by reading and interview tasks",
            "target_labels_used_for_training": True,
            "target_labels_used_for_model_selection": True,
            "target_labels_used_for_threshold_selection": True,
            "external_validation": False,
            "cross_corpus": False,
            "zero_shot": False,
            "target_domain_refitting": True,
            "threshold_adaptation_performed": True,
            "confidence_interval": ci.get(exp_id, "not_reported"),
            "allowed_claims": [
                "WavLM representations contain information shared across reading and interview tasks",
                "cross-task transfer remained strong but lower than matched within-task performance",
                "classifier boundaries were partly task-specific",
            ],
            "prohibited_claims": ["external validation", "cross-dataset generalization", "clinical diagnosis", "statistical proof that pooled training is superior"],
            "comparison_caveats": ["All Phase 2E results are within Androids Corpus on 110 paired participants.", "Pooled-task gains were suggestive rather than conclusive because bootstrap confidence intervals included zero."],
            "clinical_status": "not_clinically_validated",
        }
    )
    if exp_id == "phase2e_matched_reading_within_task":
        p.update({"result_scope": "within_corpus_cross_validation", "training_task": "reading", "evaluation_task": "reading", "paper_ready_result_sentence": "Under participant-coherent nested cross-validation on 110 paired Androids Corpus participants, the matched reading classifier achieved ROC-AUC 0.9048."})
    elif exp_id == "phase2e_matched_interview_within_task":
        p.update({"result_scope": "within_corpus_cross_validation", "training_task": "interview", "evaluation_task": "interview", "paper_ready_result_sentence": "Under participant-coherent nested cross-validation on 110 paired Androids Corpus participants, the matched interview classifier achieved ROC-AUC 0.9065."})
    elif exp_id == "phase2e_reading_to_interview_transfer":
        p.update({"result_scope": "cross_task_transfer", "training_task": "reading", "evaluation_task": "interview", "paper_ready_result_sentence": "Reading-to-interview transfer achieved ROC-AUC 0.8475 under participant-coherent nested cross-validation, with a matched-interview AUC gap of -0.0590."})
    elif exp_id == "phase2e_interview_to_reading_transfer":
        p.update({"result_scope": "cross_task_transfer", "training_task": "interview", "evaluation_task": "reading", "paper_ready_result_sentence": "Interview-to-reading transfer achieved ROC-AUC 0.8501 under participant-coherent nested cross-validation, with a matched-reading AUC gap of -0.0547."})
    elif exp_id == "phase2e_pooled_to_reading":
        p.update({"result_scope": "pooled_task_within_corpus_cross_validation", "training_task": "reading plus interview", "evaluation_task": "reading", "paper_ready_result_sentence": "Pooled reading-and-interview training achieved ROC-AUC 0.9058 for reading speech under participant-coherent nested cross-validation; the gain over single-source transfer was suggestive rather than conclusive."})
    elif exp_id == "phase2e_pooled_to_interview":
        p.update({"result_scope": "pooled_task_within_corpus_cross_validation", "training_task": "reading plus interview", "evaluation_task": "interview", "paper_ready_result_sentence": "Pooled reading-and-interview training achieved ROC-AUC 0.9135 for interview speech under participant-coherent nested cross-validation; the gain over single-source transfer was suggestive rather than conclusive."})
    else:
        p.update({"result_scope": "cross_task_transfer", "training_task": "reading and interview", "evaluation_task": "reading and interview", "paper_ready_result_sentence": "Under participant-coherent nested cross-validation on 110 paired Androids Corpus participants, reading-to-interview and interview-to-reading transfer achieved ROC-AUCs of 0.8475 and 0.8501. Pooled reading-and-interview training achieved ROC-AUCs of 0.9058 for reading and 0.9135 for interview speech."})


def paper_interpretation_for(exp: dict[str, Any]) -> dict[str, Any]:
    p = paper_defaults(exp)
    phase = exp.get("phase")
    if phase and str(phase).startswith("Phase 1"):
        _phase1_interpretation(exp, p)
    elif phase == "Phase 2A":
        _phase2a_interpretation(exp, p)
    elif phase == "Phase 2B":
        _phase2b_interpretation(exp, p)
    elif phase == "Phase 2C":
        _phase2c_interpretation(exp, p)
    elif phase == "Phase 2D":
        _phase2d_interpretation(exp, p)
    elif phase == "Phase 2E":
        _phase2e_interpretation(exp, p)
    for field in PAPER_INTERPRETATION_FIELDS:
        p.setdefault(field, None)
    return p


def apply_paper_interpretations(registry: dict[str, Any]) -> None:
    for exp in registry.get("experiments", []):
        p = paper_interpretation_for(exp)
        exp["paper_interpretation"] = p
        integrity = exp.setdefault("integrity", {})
        integrity["external_validation"] = bool(p["external_validation"])
        integrity["zero_shot"] = bool(p["zero_shot"])
        integrity["target_model_refitted"] = bool(p["target_domain_refitting"])
        integrity["target_threshold_refitted"] = bool(p["threshold_adaptation_performed"])


def metric_block(source: dict[str, Any] | None) -> dict[str, Any]:
    if not source:
        return {}
    return {
        "roc_auc": fnum(source.get("roc_auc") or source.get("mean_roc_auc") or source.get("mean_outer_roc_auc")),
        "roc_auc_std": fnum(source.get("roc_auc_std") or source.get("std_outer_roc_auc")),
        "pr_auc": fnum(source.get("pr_auc") or source.get("mean_pr_auc") or source.get("mean_outer_pr_auc")),
        "balanced_accuracy": fnum(source.get("balanced_accuracy") or source.get("mean_balanced_accuracy_at_0_5") or source.get("mean_outer_balanced_accuracy_at_0_5")),
        "macro_f1": fnum(source.get("macro_f1") or source.get("mean_macro_f1_at_0_5") or source.get("mean_outer_macro_f1_at_0_5")),
        "sensitivity": fnum(source.get("positive_recall_sensitivity") or source.get("sensitivity")),
        "specificity": fnum(source.get("specificity")),
        "threshold": fnum(source.get("threshold")),
    }


def parse_phase1a(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "evaluation_report.json")
    entries: list[dict[str, Any]] = []

    meta = report["metadata_sanity_baseline"]
    e = empty_entry("phase1a_metadata_baseline", "Phase 1A", "Metadata-only sanity baseline", report["status"], run_dir)
    apply_phase1_common(e)
    e["representation"].update({"feature_family": "metadata", "dimension": 3, "fusion": "none"})
    e["classifier"].update({"name": "LogisticRegression", "pipeline": "ColumnTransformer + LogisticRegression", "class_weight": "balanced", "C": meta["selected_C"], "selection_protocol": "train-only repeated stratified CV"})
    e["threshold"] = {"source": "dev macro-F1 search", "value": meta["selected_threshold"], "tuned_on": "dev"}
    e["metrics"]["train_cv"] = metric_block(meta["train_cv_selected_C"])
    e["metrics"]["dev"] = metric_block(meta["dev_metrics_at_selected_threshold"])
    e["metrics"]["test"] = metric_block(meta["test_metrics_at_selected_threshold"])
    e["integrity"]["test_result_viewed"] = True
    e["conclusion"].update({"main_finding": "Metadata sanity baseline underperformed and duration was nearly identical to window count.", "selected": False, "rejection_reason": "Weak provisional test ROC-AUC and metadata-only sanity role.", "limitations": ["participant_speech_duration and num_windows were almost perfectly correlated"]})
    e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json"), "predictions": rel(run_dir / "metadata_participant_predictions.csv"), "model": rel(run_dir / "models/metadata_logistic_regression_pipeline.joblib")})
    entries.append(e)

    wav = report["wavlm_participant_baseline"]
    e = empty_entry("phase1a_wavlm_final_layer", "Phase 1A", "Final-layer WavLM participant baseline", report["status"], run_dir)
    apply_phase1_common(e)
    e["representation"].update({"feature_family": "wavlm_embedding", "model": "microsoft/wavlm-base-plus", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b", "layer": "layer_12", "window_sec": 10, "overlap": 0.5, "window_pooling": "temporal mean", "participant_aggregation": "mean", "dimension": 768})
    e["classifier"].update({"name": "LogisticRegression", "pipeline": "StandardScaler + LogisticRegression", "class_weight": "balanced", "C": wav["selected_C"], "selection_protocol": "train-only repeated stratified CV"})
    e["threshold"] = {"source": "dev macro-F1 search", "value": wav["selected_threshold"], "tuned_on": "dev"}
    e["metrics"]["train_cv"] = metric_block(wav["train_cv_selected_C"])
    e["metrics"]["dev"] = metric_block(wav["dev_metrics_at_selected_threshold"])
    e["metrics"]["test"] = metric_block(wav["test_metrics_at_selected_threshold"])
    e["integrity"]["test_result_viewed"] = True
    e["conclusion"].update({"main_finding": "Final WavLM layer was weak on dev and provisional local test.", "selected": False, "rejection_reason": "Layer-wise Phase 1B found layer_09 more promising.", "limitations": ["provisional local test provenance"]})
    e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json"), "predictions": rel(run_dir / "wavlm_participant_predictions.csv"), "model": rel(run_dir / "models/wavlm_logistic_regression_pipeline.joblib"), "manifest": rel(run_dir / "participant_embedding_manifest.csv")})
    entries.append(e)
    return entries


def parse_phase1b(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "evaluation_report.json")
    selected = report["selected_layer_final_evaluation"]
    entries: list[dict[str, Any]] = []
    for row in report["nested_cv_summary_by_layer"]:
        layer = row["layer"]
        e = empty_entry(f"phase1b_wavlm_{layer}", "Phase 1B", f"WavLM {layer} layer-wise analysis", report["status"], run_dir)
        apply_phase1_common(e)
        e["data"]["test_n"] = None
        e["representation"].update({"feature_family": "wavlm_embedding", "model": "microsoft/wavlm-base-plus", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b", "layer": layer, "window_sec": 10, "overlap": 0.5, "window_pooling": "temporal mean", "participant_aggregation": "mean", "dimension": 768})
        e["classifier"].update({"name": "LogisticRegression", "pipeline": "StandardScaler + LogisticRegression", "class_weight": "balanced", "C": selected["selected_C"] if layer == "layer_09" else None, "other_parameters": {"selected_C_frequency": row.get("selected_C_frequency")}, "selection_protocol": "train-only nested repeated stratified CV"})
        e["metrics"]["train_cv"] = metric_block(row)
        if layer == "layer_09":
            e["threshold"] = {"source": "dev macro-F1 search after train-only layer selection", "value": selected["selected_dev_threshold"], "tuned_on": "dev"}
            e["metrics"]["dev"] = metric_block(selected["dev_metrics_at_selected_threshold"])
            e["conclusion"].update({"selected": True, "selection_reason": "Highest mean outer ROC-AUC under train-only nested CV.", "main_finding": "Intermediate WavLM layer_09 outperformed layer_12 in train-only nested CV and improved dev ROC-AUC.", "warnings": ["Layer-selection uncertainty remains; Phase 1B did not use test data."]})
            e["artifacts"].update({"model": rel(run_dir / "models/selected_layer_logistic_regression_pipeline.joblib"), "predictions": rel(run_dir / "selected_layer_dev_predictions.csv")})
        else:
            e["conclusion"].update({"main_finding": f"{layer} was evaluated in train-only nested CV.", "selected": False, "rejection_reason": "Not selected by train-only layer-selection criterion."})
        e["integrity"].update({"test_result_viewed": False, "test_provenance": ""})
        e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json"), "manifest": rel(run_dir / "participant_layer_embedding_manifest.csv")})
        entries.append(e)
    return entries


def parse_phase1c(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "evaluation_report.json")
    selected = report["selected_dev_evaluation"]
    entries: list[dict[str, Any]] = []
    for row in report["nested_cv_summary"]:
        method = row["aggregation_method"]
        e = empty_entry(f"phase1c_aggregation_{method}", "Phase 1C", f"Layer-9 {method} aggregation", report["status"], run_dir)
        apply_phase1_common(e)
        e["data"]["test_n"] = None
        e["representation"].update({"feature_family": "wavlm_embedding", "model": "microsoft/wavlm-base-plus", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b", "layer": "layer_09", "window_sec": 10, "overlap": 0.5, "window_pooling": "temporal mean", "participant_aggregation": method, "dimension": row.get("representation_dimension")})
        e["classifier"].update({"name": "LogisticRegression", "pipeline": "StandardScaler + LogisticRegression" + (" + PCA" if row.get("pca_used") else ""), "class_weight": "balanced", "C": selected["selected_C"] if method == "mean" else None, "other_parameters": {"selected_C_frequency": row.get("selected_C_frequency"), "selected_PCA_frequency": row.get("selected_PCA_frequency")}, "selection_protocol": "train-only nested CV"})
        e["metrics"]["train_cv"] = metric_block(row)
        if method == "mean":
            e["threshold"] = {"source": "dev macro-F1 search", "value": selected["selected_dev_threshold"], "tuned_on": "dev"}
            e["metrics"]["dev"] = metric_block(selected["dev_metrics_at_selected_threshold"])
            e["conclusion"].update({"selected": True, "selection_reason": "Mean aggregation had the highest train nested-CV ROC-AUC and remained simplest.", "main_finding": "Aggregation was not the main bottleneck; mean remained selected.", "next_step": "Assess timing features."})
            e["artifacts"].update({"model": rel(run_dir / "models/selected_aggregation_pipeline.joblib"), "predictions": rel(run_dir / "selected_aggregation_dev_predictions.csv")})
        else:
            e["conclusion"].update({"selected": False, "main_finding": f"{method} aggregation did not improve over mean.", "rejection_reason": "Lower or practically negligible train nested-CV performance versus mean."})
        e["integrity"].update({"test_result_viewed": False, "test_provenance": ""})
        e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json"), "manifest": rel(run_dir / "aggregation_manifest.csv")})
        entries.append(e)
    return entries


def parse_phase1d(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "evaluation_report.json")
    selected = report["selected_dev_evaluation"]
    entries: list[dict[str, Any]] = []
    for row in report["nested_cv_summary"]:
        name = row["model_name"]
        e = empty_entry(f"phase1d_{name}", "Phase 1D", f"Timing fusion model: {name}", report["status"], run_dir)
        apply_phase1_common(e)
        e["data"]["test_n"] = None
        feature_family = "timing" if name == "timing_only" else "wavlm_embedding" if name == "wavlm_only" else "wavlm_plus_timing"
        e["representation"].update({"feature_family": feature_family, "model": "microsoft/wavlm-base-plus" if name != "timing_only" else "", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b" if name != "timing_only" else "", "layer": "layer_09" if name != "timing_only" else None, "participant_aggregation": "mean" if name != "timing_only" else "", "dimension": row.get("representation_dimension"), "timing_feature_count": 28 if name != "wavlm_only" else None, "fusion": "early feature concatenation" if name == "early_fusion" else "none"})
        e["classifier"].update({"name": "LogisticRegression", "pipeline": "leakage-safe preprocessing + LogisticRegression", "class_weight": "balanced", "C": 100.0 if name == "timing_only" else 0.001 if name == "wavlm_only" else None, "other_parameters": {"selected_C_frequency": row.get("selected_C_frequency")}, "selection_protocol": "train-only nested CV"})
        e["metrics"]["train_cv"] = metric_block(row)
        if name == "timing_only":
            e["threshold"] = {"source": "dev macro-F1 search", "value": selected["selected_dev_threshold"], "tuned_on": "dev"}
            e["metrics"]["dev"] = metric_block(selected["dev_metrics_at_selected_threshold"])
            e["conclusion"].update({"selected": True, "selection_reason": "Highest train nested-CV ROC-AUC among timing/WavLM/fusion.", "main_finding": "Compact speech-timing features added useful signal on DAIC guided interview.", "limitations": ["Requires guided-interview timing structure."]})
            e["artifacts"].update({"model": rel(run_dir / "models/selected_phase1d_pipeline.joblib"), "predictions": rel(run_dir / "selected_phase1d_dev_predictions.csv")})
        elif name == "early_fusion":
            e["conclusion"].update({"selected": False, "main_finding": "Early feature fusion underperformed.", "rejection_reason": "Train ROC-AUC below timing-only and WavLM-only."})
        else:
            e["conclusion"].update({"selected": False, "main_finding": "WavLM-only reproduced Phase 1C result within tolerance.", "rejection_reason": "Timing-only was stronger under Phase 1D selection."})
        e["integrity"].update({"test_result_viewed": False, "test_provenance": ""})
        e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json"), "manifest": rel(run_dir / "speech_timing_features.csv") if name != "wavlm_only" else rel("outputs/phase1_daic/phase1c_aggregation_runs/20260622_135800/aggregation_manifest.csv")})
        entries.append(e)
    return entries


def parse_phase1e(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "evaluation_report.json")
    selected = report["selected_dev_evaluation"]
    entries: list[dict[str, Any]] = []
    for row in report["nested_cv_summary"]:
        name = row["model_name"]
        e = empty_entry(f"phase1e_{name}", "Phase 1E", f"Late-fusion protocol: {name}", report["status"], run_dir)
        apply_phase1_common(e)
        e["data"]["test_n"] = None
        family = "timing" if name == "timing_only" else "wavlm_embedding" if name == "wavlm_only" else "probability_late_fusion"
        e["representation"].update({"feature_family": family, "model": "microsoft/wavlm-base-plus" if name != "timing_only" else "", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b" if name != "timing_only" else "", "layer": "layer_09" if name != "timing_only" else None, "participant_aggregation": "mean" if name != "timing_only" else "", "dimension": row.get("representation_dimension"), "timing_feature_count": 28 if name != "wavlm_only" else None, "fusion": name if name in {"weighted_late_fusion", "logistic_stacking"} else "none"})
        e["classifier"].update({"name": "LogisticRegression" if name != "weighted_late_fusion" else "Weighted probability fusion", "pipeline": "leakage-safe probability-level protocol", "class_weight": "balanced" if name != "weighted_late_fusion" else "", "C": 100.0 if name == "timing_only" else 0.001 if name == "wavlm_only" else None, "other_parameters": {"selected_timing_C_frequency": row.get("selected_timing_C_frequency"), "selected_wavlm_C_frequency": row.get("selected_wavlm_C_frequency"), "selected_weight_frequency": row.get("selected_weight_frequency"), "selected_stacker_C_frequency": row.get("selected_stacker_C_frequency")}, "selection_protocol": "train-only nested CV"})
        e["metrics"]["train_cv"] = metric_block(row)
        if name == "timing_only":
            e["threshold"] = {"source": "dev macro-F1 search", "value": selected["selected_dev_threshold"], "tuned_on": "dev"}
            e["metrics"]["dev"] = metric_block(selected["dev_metrics_at_selected_threshold"])
            e["conclusion"].update({"selected": True, "selection_reason": "Best train nested-CV ROC-AUC; fusion failed practical improvement rule.", "main_finding": "Fusion did not improve reliability; timing-only retained.", "limitations": ["Guided-interview-dependent."]})
            e["artifacts"].update({"model": rel(run_dir / "models/timing_base_pipeline.joblib"), "predictions": rel(run_dir / "selected_phase1e_dev_predictions.csv")})
        else:
            reason = "Fusion did not improve over timing-only." if "fusion" in name or "stacking" in name else "Timing-only was selected."
            e["conclusion"].update({"selected": False, "main_finding": f"{name} evaluated under leakage-safe Phase 1E protocol.", "rejection_reason": reason})
        e["integrity"].update({"test_result_viewed": False, "test_provenance": ""})
        e["artifacts"].update({"report_json": rel(run_dir / "evaluation_report.json")})
        entries.append(e)
    return entries


def parse_phase1f(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE1_FINAL_REPORT.json")
    candidates = report["frozen_candidates"]
    entries: list[dict[str, Any]] = []
    for key, exp_id, name in [
        ("candidate_A_wavlm_free_speech", "phase1f_frozen_wavlm", "Frozen DAIC WavLM free-speech candidate"),
        ("candidate_B_timing_guided_interview", "phase1f_frozen_timing", "Frozen DAIC guided-interview timing candidate"),
    ]:
        cand = candidates[key]
        test = cand["provisional_local_test_metrics"]
        e = empty_entry(exp_id, "Phase 1F", name, report["phase1_status"], run_dir)
        apply_phase1_common(e)
        if key.endswith("free_speech"):
            e["representation"].update({"feature_family": "wavlm_embedding", "model": cand["model"], "model_commit": cand["model_commit"], "layer": cand["layer"], "window_sec": 10, "overlap": 0.5, "window_pooling": "temporal mean", "participant_aggregation": "mean", "dimension": 768})
            e["classifier"].update({"name": "LogisticRegression", "pipeline": cand["classifier"], "class_weight": "balanced", "C": 0.001, "selection_protocol": "frozen after Phase 1C"})
            threshold = cand["frozen_dev_threshold"]
            pred = rel(run_dir / "wavlm_final_test_predictions.csv")
            model_path = rel("outputs/phase1_daic/phase1c_aggregation_runs/20260622_135800/models/selected_aggregation_pipeline.joblib")
            selected = True
            main = "Free-speech-compatible candidate closed for external validation, but provisional local-test AUC was near chance."
            warnings = ["Not deployment-ready; external validation required."]
        else:
            e["representation"].update({"feature_family": "timing", "timing_feature_count": 28, "dimension": 28})
            e["classifier"].update({"name": "LogisticRegression", "pipeline": cand["classifier"], "class_weight": "balanced", "C": 100.0, "selection_protocol": "frozen after Phase 1E"})
            threshold = cand["frozen_dev_threshold"]
            pred = rel(run_dir / "timing_final_test_predictions.csv")
            model_path = rel("outputs/phase1_daic/phase1e_late_fusion_runs/20260622_145240/models/timing_base_pipeline.joblib")
            selected = True
            main = "Guided-interview timing candidate performed better on provisional local test, but depends on standardized interaction timing."
            warnings = ["Not free-speech-compatible.", "Not deployment-ready; external validation required."]
        e["threshold"] = {"source": "frozen DAIC dev threshold", "value": threshold, "tuned_on": "dev before final test"}
        e["metrics"]["test"] = metric_block(test)
        e["integrity"].update({"test_provenance": report["evaluation_status"], "test_result_viewed": True, "decisions_made_after_test_view": ["Phase 1 closed; no post-test model modification."], "leakage_status": "test viewed once for Phase 1 closure; no post-test tuning recorded"})
        e["conclusion"].update({"selected": selected, "selection_reason": "Frozen Phase 1 candidate.", "main_finding": main, "limitations": ["test provenance is provisional local labeled test"], "warnings": warnings, "next_step": "External validation"})
        e["artifacts"].update({"report_json": rel(run_dir / "PHASE1_FINAL_REPORT.json"), "report_markdown": rel(run_dir / "PHASE1_FINAL_REPORT.md") if (run_dir / "PHASE1_FINAL_REPORT.md").exists() else "", "predictions": pred, "model": model_path})
        entries.append(e)
    return entries


def parse_phase2a(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE2A_AUDIT_REPORT.json")
    e = empty_entry("phase2a_android_audit", "Phase 2A", "Androids Corpus preparation and compatibility audit", report["status"], run_dir)
    e["data"].update({"dataset_id": "androids_corpus", "task": "inventory and compatibility audit", "language": "Italian", "train_n": None, "dev_n": None, "test_n": None, "class_counts": {"HC": 54, "PT": 64}, "split_protocol": "audit only; no modeling"})
    e["representation"].update({"feature_family": "audit", "fusion": "none"})
    e["classifier"].update({"name": "none", "pipeline": "none", "selection_protocol": "not applicable"})
    e["metrics"]["test"] = {}
    e["metrics"]["confidence_intervals"] = {}
    e["integrity"].update({"test_provenance": "not a model evaluation", "external_validation": False, "zero_shot": False, "test_result_viewed": False, "leakage_status": "no modeling or prediction performed"})
    e["conclusion"].update({"main_finding": "Android interview WavLM preprocessing is reproducible; timing model direct transfer is invalid because response-latency features are unavailable.", "selected": False, "limitations": ["85 participants have cross-task fold inconsistency", "DAIC PHQ8_Binary and Android clinical-group labels are related but not identical"], "warnings": [f"Timing compatibility: {report['timing_exactly_compatible_feature_count']} exact, 3 approximate, 8 unavailable"], "next_step": "Phase 2B zero-shot WavLM interview evaluation"})
    e["artifacts"].update({"report_json": rel(run_dir / "PHASE2A_AUDIT_REPORT.json"), "report_markdown": rel(run_dir / "PHASE2A_AUDIT_REPORT.md"), "manifest": rel(run_dir / "android_participant_manifest.csv")})
    return [e]


def parse_phase2b(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.json")
    e = empty_entry("phase2b_daic_to_android_zero_shot_wavlm", "Phase 2B", "Frozen DAIC-to-Android zero-shot WavLM interview", report["status"], run_dir)
    e["data"].update({"dataset_id": "androids_corpus", "task": "interview participant speech", "language": "Italian", "train_n": 0, "dev_n": 0, "test_n": report["eligible_interview_participants"], "class_counts": report["class_counts"], "split_protocol": "external zero-shot; every eligible Android interview participant evaluated once", "excluded": report["excluded_participants"]})
    e["representation"].update({"feature_family": "wavlm_embedding", "model": "microsoft/wavlm-base-plus", "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b", "layer": "layer_09", "window_sec": 10, "overlap": 0.5, "window_pooling": "temporal mean", "participant_aggregation": "mean", "dimension": 768})
    e["classifier"].update({"name": "Frozen DAIC LogisticRegression", "pipeline": "Frozen Phase 1C StandardScaler + LogisticRegression", "class_weight": "balanced", "C": 0.001, "selection_protocol": "selected on DAIC train/dev only; no Android fitting"})
    e["threshold"] = {"source": "frozen DAIC dev threshold", "value": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["threshold"], "tuned_on": "DAIC dev only"}
    e["metrics"]["test"] = {
        **metric_block(report["zero_shot_metrics"]["primary"]),
        "balanced_accuracy": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["balanced_accuracy"],
        "macro_f1": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["macro_f1"],
        "sensitivity": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["sensitivity"],
        "specificity": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["specificity"],
        "threshold": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["threshold"],
        "tn": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["tn"],
        "fp": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["fp"],
        "fn": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["fn"],
        "tp": report["zero_shot_metrics"]["at_frozen_daic_threshold"]["tp"],
    }
    ci = {}
    for row in report["bootstrap_intervals"]:
        if row["metric"] in {"roc_auc", "pr_auc"}:
            ci[row["metric"]] = {"p2_5": row["percentile_2_5"], "median": row["percentile_50"], "p97_5": row["percentile_97_5"]}
    e["metrics"]["confidence_intervals"] = ci
    e["integrity"].update({"test_provenance": "Android clinical patient/control labels; external cross-corpus evaluation target differs from DAIC PHQ8_Binary", "external_validation": True, "zero_shot": True, "test_result_viewed": True, "target_model_refitted": False, "target_threshold_refitted": False, "decisions_made_after_test_view": ["No Android adaptation performed; Phase 2C planned to separate calibration and boundary shift."], "leakage_status": "no Android training, scaler fitting, threshold tuning, or recalibration"})
    e["conclusion"].update({"main_finding": "Ranking transferred modestly, but probability scale and frozen threshold did not: all Android participants were predicted negative.", "selected": False, "selection_reason": "External validation result, not model selection.", "limitations": ["DAIC English PHQ8_Binary and Android Italian clinical-group labels are not equivalent", "threshold transfer failed", "probabilities are shifted low"], "warnings": ["Do not claim clinical validation."], "next_step": "Phase 2C Android adaptation analysis"})
    e["artifacts"].update({"report_json": rel(run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.json"), "report_markdown": rel(run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.md"), "predictions": rel(run_dir / "android_interview_zero_shot_predictions.csv"), "manifest": rel(run_dir / "android_interview_window_manifest.csv")})
    env = read_json(run_dir / "phase2b_run_metadata.json") if (run_dir / "phase2b_run_metadata.json").exists() else {}
    e["environment"].update({"python": env.get("python_version"), "pytorch": env.get("pytorch_version"), "transformers": env.get("transformers_version"), "scikit_learn": env.get("sklearn_version"), "cuda": env.get("cuda_version"), "gpu": env.get("gpu_model"), "git_commit": None})
    return [e]


def parse_phase2c(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.json")
    metadata = read_json(run_dir / "phase2c_run_metadata.json") if (run_dir / "phase2c_run_metadata.json").exists() else {}
    by_protocol = {row["protocol"]: row for row in report["protocol_metrics"]}
    protocol_map = [
        (
            "A_frozen_daic_at_frozen_threshold",
            "phase2c_protocol_a_frozen_daic_reference",
            "Protocol A frozen DAIC zero-shot reference",
            "Frozen DAIC StandardScaler + LogisticRegression",
            False,
            False,
            True,
            "Frozen DAIC threshold; no Android adaptation",
        ),
        (
            "B_threshold_only",
            "phase2c_protocol_b_threshold_only",
            "Protocol B threshold-only adaptation",
            "Frozen DAIC StandardScaler + LogisticRegression",
            False,
            True,
            False,
            "Fold-specific Android training-fold threshold adaptation",
        ),
        (
            "C_frozen_score_calibration",
            "phase2c_protocol_c_score_calibration",
            "Protocol C frozen-score calibration",
            "Frozen DAIC score plus fold-selected calibrator",
            False,
            False,
            False,
            "Fold-selected intercept/Platt/isotonic calibration; threshold 0.5",
        ),
        (
            "D_android_lr_threshold_adapted",
            "phase2c_protocol_d_android_lr",
            "Protocol D Android-trained logistic regression",
            "Android fold-trained StandardScaler + LogisticRegression",
            True,
            True,
            False,
            "Official Android five-fold inner C and threshold selection",
        ),
    ]
    entries: list[dict[str, Any]] = []
    for protocol_key, exp_id, name, pipeline, target_refit, threshold_refit, zero_shot, selection_protocol in protocol_map:
        metrics = by_protocol[protocol_key]
        e = empty_entry(exp_id, "Phase 2C", name, report["status"], run_dir)
        e["parent_id"] = "P2C_ANDROID_DOMAIN_SHIFT_001"
        e["data"].update(
            {
                "dataset_id": "androids_corpus",
                "task": "interview participant speech",
                "language": "Italian",
                "train_n": None if protocol_key.startswith("A_") else "official five-fold outer training folds",
                "dev_n": None,
                "test_n": report["data_summary"]["participants"],
                "class_counts": report["data_summary"]["class_counts"],
                "split_protocol": "official Android interview five-fold participant-independent protocol",
                "excluded": [],
            }
        )
        e["representation"].update(
            {
                "feature_family": "wavlm_embedding",
                "model": "microsoft/wavlm-base-plus",
                "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b",
                "layer": "layer_09",
                "window_sec": 10,
                "overlap": 0.5,
                "window_pooling": "temporal mean",
                "participant_aggregation": "mean",
                "dimension": 768,
                "fusion": "none",
            }
        )
        selected_c = None
        if protocol_key == "D_android_lr_threshold_adapted":
            selected_c = report.get("selected_C_distribution")
        elif protocol_key.startswith("A_") or protocol_key == "B_threshold_only":
            selected_c = 0.001
        e["classifier"].update(
            {
                "name": "LogisticRegression" if protocol_key.startswith("D_") else "Frozen DAIC LogisticRegression/calibrator",
                "pipeline": pipeline,
                "class_weight": "balanced" if protocol_key != "C_frozen_score_calibration" else "not applicable to calibrator",
                "C": selected_c,
                "other_parameters": {
                    "calibration_method_distribution": report.get("calibration_method_distribution") if protocol_key == "C_frozen_score_calibration" else None,
                    "selected_C_distribution": report.get("selected_C_distribution") if protocol_key == "D_android_lr_threshold_adapted" else None,
                },
                "selection_protocol": selection_protocol,
            }
        )
        threshold_value = metrics.get("threshold")
        e["threshold"].update({"source": selection_protocol, "value": threshold_value, "tuned_on": "Android outer-training folds only" if threshold_refit else "DAIC or fixed 0.5"})
        e["metrics"]["test"].update(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "positive_precision": metrics["positive_class_precision"],
                "negative_predictive_value": metrics["negative_predictive_value"],
                "brier_score": metrics["brier_score"],
                "threshold": threshold_value,
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
        e["metrics"]["confidence_intervals"].update(
            {
                "source": rel(run_dir / "phase2c_bootstrap_intervals.csv"),
                "bootstrap_replicates": 5000,
                "paired_comparisons": rel(run_dir / "phase2c_paired_comparisons.csv"),
            }
        )
        e["integrity"].update(
            {
                "test_provenance": "Android clinical patient/control labels under official interview folds",
                "external_validation": True,
                "zero_shot": zero_shot,
                "test_result_viewed": True,
                "target_model_refitted": target_refit,
                "target_threshold_refitted": threshold_refit,
                "test_used_for_selection": False,
                "decisions_made_after_test_view": [
                    "Phase 2C is diagnostic; no Phase 1 frozen candidate was altered.",
                    "No reading speech, WavLM extraction, timing features, or new model family was introduced.",
                ],
                "leakage_status": report["leakage_checks"]["status"],
            }
        )
        e["conclusion"].update(
            {
                "main_finding": report["interpretation"]["dominant_failure_source"],
                "selected": False,
                "selection_reason": "Diagnostic protocol result; not a new frozen cross-corpus candidate.",
                "limitations": [
                    "Android labels and DAIC PHQ8_Binary are related but not identical targets.",
                    "Within-Android Protocol D estimates target-domain boundary shift, not zero-shot clinical validity.",
                ],
                "warnings": ["Do not claim clinical validation.", "Repeated external-result inspection must be treated as diagnostic history."],
                "next_step": "Design any future Android adaptation protocol separately before execution.",
            }
        )
        e["artifacts"].update(
            {
                "report_json": rel(run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.json"),
                "report_markdown": rel(run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.md"),
                "predictions": "",
                "manifest": "",
            }
        )
        e["environment"].update(
            {
                "python": metadata.get("python_version"),
                "scikit_learn": metadata.get("sklearn_version"),
                "git_commit": metadata.get("git_commit"),
            }
        )
        entries.append(e)
    return entries


def parse_phase2d(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE2D_ANDROID_READING_REPORT.json")
    metadata = read_json(run_dir / "phase2d_run_metadata.json") if (run_dir / "phase2d_run_metadata.json").exists() else {}
    by_protocol = {row["protocol"]: row for row in report["protocol_metrics"]}
    task_rows = {row["task"]: row for row in report["task_comparison"]}
    entries: list[dict[str, Any]] = []

    common_data = {
        "dataset_id": "androids_corpus",
        "task": "reading speech",
        "language": "Italian",
        "train_n": None,
        "dev_n": None,
        "test_n": report["data_summary"]["eligible_reading_participants"],
        "class_counts": report["data_summary"]["class_counts"],
        "split_protocol": "official Android reading five-fold participant-independent protocol",
        "excluded": [],
    }
    common_representation = {
        "feature_family": "wavlm_embedding",
        "model": "microsoft/wavlm-base-plus",
        "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "layer": "layer_09",
        "window_sec": 10,
        "overlap": 0.5,
        "window_pooling": "temporal mean",
        "participant_aggregation": "mean",
        "dimension": 768,
        "fusion": "none",
    }

    protocol_map = [
        (
            "A_zero_shot_daic_at_frozen_threshold",
            "phase2d_android_reading_zero_shot_daic",
            "Frozen DAIC zero-shot Android reading",
            "Frozen Phase 1C StandardScaler + LogisticRegression",
            0.001,
            False,
            False,
            True,
            "Frozen DAIC dev threshold; no Android reading fitting",
            report["zero_shot_reading"]["metrics_at_frozen_threshold"]["threshold"],
            "DAIC dev only",
        ),
        (
            "B_android_lr_threshold_adapted",
            "phase2d_android_reading_logistic_regression",
            "Android-trained reading logistic regression",
            "Official reading-fold StandardScaler + LogisticRegression",
            report["android_trained_reading"]["selected_C_distribution"],
            True,
            True,
            False,
            "Official reading five-fold outer evaluation with leave-one-training-fold-out inner C selection",
            None,
            "Android reading outer-training folds only",
        ),
    ]
    for protocol_key, exp_id, name, pipeline, c_value, target_refit, threshold_refit, zero_shot, selection_protocol, threshold_value, tuned_on in protocol_map:
        metrics = by_protocol[protocol_key]
        e = empty_entry(exp_id, "Phase 2D", name, report["status"], run_dir)
        e["data"].update(common_data)
        if target_refit:
            e["data"]["train_n"] = "four official reading folds per outer evaluation"
        e["representation"].update(common_representation)
        e["classifier"].update(
            {
                "name": "LogisticRegression" if target_refit else "Frozen DAIC LogisticRegression",
                "pipeline": pipeline,
                "class_weight": "balanced",
                "C": c_value,
                "other_parameters": {
                    "selected_C_distribution": report["android_trained_reading"].get("selected_C_distribution") if target_refit else None,
                    "selected_threshold_distribution": report["android_trained_reading"].get("selected_threshold_distribution") if target_refit else None,
                },
                "selection_protocol": selection_protocol,
            }
        )
        e["threshold"].update({"source": selection_protocol, "value": threshold_value, "tuned_on": tuned_on})
        e["metrics"]["test"].update(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "positive_precision": metrics["positive_class_precision"],
                "negative_predictive_value": metrics["negative_predictive_value"],
                "brier_score": metrics["brier_score"],
                "threshold": metrics.get("threshold"),
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
        e["metrics"]["confidence_intervals"].update(
            {
                "source": rel(run_dir / "phase2d_bootstrap_intervals.csv"),
                "bootstrap_replicates": 5000,
                "paired_comparisons": rel(run_dir / "phase2d_paired_comparisons.csv"),
            }
        )
        e["integrity"].update(
            {
                "test_provenance": "Android clinical patient/control labels under official reading folds",
                "external_validation": True,
                "zero_shot": zero_shot,
                "test_result_viewed": True,
                "target_model_refitted": target_refit,
                "target_threshold_refitted": threshold_refit,
                "test_used_for_selection": False,
                "decisions_made_after_test_view": [
                    "Phase 2D is a pre-specified reading benchmark and task comparison.",
                    "No Phase 1 model, Phase 2C model, WavLM layer, or aggregation rule was altered.",
                ],
                "leakage_status": report["leakage_checks"]["status"],
            }
        )
        main_finding = (
            "Frozen DAIC reading transfer retained modest ranking but threshold transfer failed."
            if zero_shot
            else f"Android-trained reading model achieved {metrics['roc_auc']:.4f} ROC-AUC under official reading folds."
        )
        e["conclusion"].update(
            {
                "main_finding": main_finding,
                "selected": False,
                "selection_reason": "Completed Phase 2D diagnostic protocol; not a new frozen cross-corpus candidate.",
                "limitations": [
                    "Android clinical group labels and DAIC PHQ8_Binary are related but not identical.",
                    "Reading and interview participant sets/folds differ slightly; task comparison is descriptive.",
                ],
                "warnings": ["Do not claim clinical validation.", "Repeated external-result inspection must be treated as diagnostic history."],
                "next_step": "Use Phase 2D only as a recorded benchmark unless a new protocol is frozen.",
            }
        )
        e["artifacts"].update(
            {
                "report_json": rel(run_dir / "PHASE2D_ANDROID_READING_REPORT.json"),
                "report_markdown": rel(run_dir / "PHASE2D_ANDROID_READING_REPORT.md"),
                "predictions": "",
                "manifest": "",
            }
        )
        e["environment"].update(
            {
                "python": metadata.get("python_version"),
                "pytorch": metadata.get("pytorch_version"),
                "transformers": metadata.get("transformers_version"),
                "scikit_learn": metadata.get("sklearn_version"),
                "cuda": metadata.get("cuda_version"),
                "gpu": metadata.get("gpu_model"),
                "git_commit": metadata.get("git_commit"),
            }
        )
        entries.append(e)

    comparison = empty_entry(
        "phase2d_android_interview_vs_reading_comparison",
        "Phase 2D",
        "Android interview-versus-reading comparison",
        report["status"],
        run_dir,
    )
    comparison["data"].update(
        {
            "dataset_id": "androids_corpus",
            "task": "reading versus interview speech",
            "language": "Italian",
            "train_n": None,
            "dev_n": None,
            "test_n": {
                "reading": report["data_summary"]["eligible_reading_participants"],
                "interview": task_rows["Android interview"]["participant_count"],
            },
            "class_counts": {
                "reading": report["data_summary"]["class_counts"],
                "interview": task_rows["Android interview"]["class_counts"],
            },
            "split_protocol": "descriptive comparison of Phase 2D reading Protocol B and Phase 2C interview Protocol D",
            "excluded": [],
        }
    )
    comparison["representation"].update(common_representation)
    comparison["classifier"].update(
        {
            "name": "Comparison of Android-trained logistic-regression protocols",
            "pipeline": "Reading Phase 2D Protocol B versus interview Phase 2C Protocol D",
            "class_weight": "balanced",
            "C": {
                "reading": report["android_trained_reading"]["selected_C_distribution"],
                "interview": task_rows["Android interview"]["selected_C_distribution"],
            },
            "selection_protocol": "descriptive task comparison; no pooled task training",
        }
    )
    reading_row = task_rows["Android reading"]
    interview_row = task_rows["Android interview"]
    comparison["metrics"]["test"].update(
        {
            "roc_auc": reading_row["roc_auc"],
            "reading_roc_auc": reading_row["roc_auc"],
            "interview_roc_auc": interview_row["roc_auc"],
            "pr_auc": reading_row["pr_auc"],
            "reading_pr_auc": reading_row["pr_auc"],
            "interview_pr_auc": interview_row["pr_auc"],
            "balanced_accuracy": reading_row["balanced_accuracy"],
            "macro_f1": reading_row["macro_f1"],
            "sensitivity": reading_row["sensitivity"],
            "specificity": reading_row["specificity"],
            "brier_score": reading_row["brier_score"],
        }
    )
    comparison["metrics"]["confidence_intervals"].update(
        {
            "source": rel(run_dir / "phase2d_bootstrap_intervals.csv"),
            "paired_comparisons": rel(run_dir / "phase2d_paired_comparisons.csv"),
            "task_comparison": rel(run_dir / "phase2d_task_comparison.csv"),
        }
    )
    comparison["integrity"].update(
        {
            "test_provenance": "Android clinical patient/control labels under official task-specific folds",
            "external_validation": True,
            "zero_shot": False,
            "test_result_viewed": True,
            "target_model_refitted": True,
            "target_threshold_refitted": True,
            "test_used_for_selection": False,
            "decisions_made_after_test_view": ["No pooled task model was trained.", "No Phase 2C interview model was rerun."],
            "leakage_status": report["leakage_checks"]["status"],
        }
    )
    comparison["conclusion"].update(
        {
            "main_finding": report["interpretation"]["narrative"],
            "selected": False,
            "selection_reason": "Task comparison record only.",
            "limitations": [
                "Reading and interview participant sets/folds are not identical.",
                "Task comparison is descriptive and not a clinical validation claim.",
            ],
            "warnings": ["Do not claim clinical validation."],
            "next_step": "Do not proceed without freezing a new protocol.",
        }
    )
    comparison["artifacts"].update(
        {
            "report_json": rel(run_dir / "PHASE2D_ANDROID_READING_REPORT.json"),
            "report_markdown": rel(run_dir / "PHASE2D_ANDROID_READING_REPORT.md"),
            "manifest": rel(run_dir / "phase2d_task_comparison.csv"),
        }
    )
    comparison["environment"].update(entries[-1]["environment"])
    entries.append(comparison)
    return entries


def parse_phase2e(run_dir: Path) -> list[dict[str, Any]]:
    report = read_json(run_dir / "PHASE2E_CROSS_TASK_REPORT.json")
    metadata = read_json(run_dir / "phase2e_run_metadata.json") if (run_dir / "phase2e_run_metadata.json").exists() else {}
    by_protocol = {(row["protocol"], row["threshold_variant"]): row for row in report["protocol_metrics"]}
    primary = "threshold_adapted"
    data_summary = report["data_summary"]
    selected_params = {row["source_model"]: row for row in report.get("selected_parameter_summary", [])}
    common_representation = {
        "feature_family": "wavlm_embedding",
        "model": "microsoft/wavlm-base-plus",
        "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "layer": "layer_09",
        "window_sec": 10,
        "overlap": 0.5,
        "window_pooling": "temporal mean",
        "participant_aggregation": "mean",
        "dimension": 768,
        "fusion": "none",
    }
    protocol_map = [
        (
            "matched_reading_within_task",
            "phase2e_matched_reading_within_task",
            "Phase 2E matched reading within-task",
            "reading speech",
            "reading_only",
            "reading",
            "reading",
            "Fold-matched reading within-task reference under participant-coherent folds.",
        ),
        (
            "matched_interview_within_task",
            "phase2e_matched_interview_within_task",
            "Phase 2E matched interview within-task",
            "interview speech",
            "interview_only",
            "interview",
            "interview",
            "Fold-matched interview within-task reference under participant-coherent folds.",
        ),
        (
            "reading_to_interview",
            "phase2e_reading_to_interview_transfer",
            "Phase 2E reading-to-interview transfer",
            "reading to interview transfer",
            "reading_only",
            "reading",
            "interview",
            "Reading-trained classifier evaluated on interview speech from held-out participants.",
        ),
        (
            "interview_to_reading",
            "phase2e_interview_to_reading_transfer",
            "Phase 2E interview-to-reading transfer",
            "interview to reading transfer",
            "interview_only",
            "interview",
            "reading",
            "Interview-trained classifier evaluated on reading speech from held-out participants.",
        ),
        (
            "pooled_to_reading",
            "phase2e_pooled_to_reading",
            "Phase 2E pooled-task training to reading",
            "pooled task training to reading",
            "pooled",
            "reading plus interview",
            "reading",
            "Pooled reading/interview training evaluated on held-out reading speech.",
        ),
        (
            "pooled_to_interview",
            "phase2e_pooled_to_interview",
            "Phase 2E pooled-task training to interview",
            "pooled task training to interview",
            "pooled",
            "reading plus interview",
            "interview",
            "Pooled reading/interview training evaluated on held-out interview speech.",
        ),
    ]
    entries: list[dict[str, Any]] = []
    for protocol, exp_id, name, task, source_model, source_task, target_task, finding in protocol_map:
        metrics = by_protocol[(protocol, primary)]
        e = empty_entry(exp_id, "Phase 2E", name, report["status"], run_dir)
        e["data"].update(
            {
                "dataset_id": "androids_corpus",
                "task": task,
                "language": "Italian",
                "train_n": "four participant-coherent folds per outer evaluation",
                "dev_n": None,
                "test_n": data_summary["paired_participants"],
                "class_counts": data_summary["class_counts"],
                "split_protocol": "new Phase 2E participant-coherent five-fold split over paired reading/interview participants",
                "excluded": {
                    "reading_only": data_summary.get("reading_only_participants"),
                    "interview_only": data_summary.get("interview_only_participants"),
                },
            }
        )
        e["representation"].update(common_representation)
        e["classifier"].update(
            {
                "name": "LogisticRegression",
                "pipeline": "StandardScaler + balanced L2 LogisticRegression",
                "class_weight": "balanced",
                "C": selected_params.get(source_model, {}).get("selected_C_distribution"),
                "other_parameters": {
                    "source_task": source_task,
                    "target_task": target_task,
                    "threshold_distribution": selected_params.get(source_model, {}),
                },
                "selection_protocol": "participant-coherent outer folds with leave-one-training-fold-out inner C selection",
            }
        )
        e["threshold"].update({"source": "outer-training inner OOF macro-F1", "value": None, "tuned_on": "outer-training participants only"})
        e["metrics"]["test"].update(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "positive_precision": metrics["positive_class_precision"],
                "negative_predictive_value": metrics["negative_predictive_value"],
                "brier_score": metrics["brier_score"],
                "threshold": metrics.get("threshold"),
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
        e["metrics"]["confidence_intervals"].update(
            {
                "source": rel(run_dir / "phase2e_bootstrap_intervals.csv"),
                "paired_comparisons": rel(run_dir / "phase2e_paired_comparisons.csv"),
                "transfer_gaps": rel(run_dir / "phase2e_transfer_gap_analysis.csv"),
            }
        )
        e["integrity"].update(
            {
                "test_provenance": "Android clinical patient/control labels under Phase 2E participant-coherent folds",
                "external_validation": True,
                "zero_shot": False,
                "test_result_viewed": True,
                "target_model_refitted": True,
                "target_threshold_refitted": True,
                "test_used_for_selection": False,
                "decisions_made_after_test_view": [
                    "Phase 2E used a frozen protocol and did not alter Phase 1, Phase 2C, or Phase 2D models.",
                    "No new WavLM extraction, audio processing, layer selection, or model family was introduced.",
                ],
                "leakage_status": report["leakage_checks"]["status"],
            }
        )
        e["conclusion"].update(
            {
                "main_finding": finding,
                "selected": False,
                "selection_reason": "Participant-safe cross-task diagnostic protocol; not a new frozen deployment candidate.",
                "limitations": [
                    "Phase 2E folds differ from official task-specific Phase 2C and Phase 2D folds.",
                    "Android clinical grouping is not the same target as DAIC PHQ8_Binary.",
                ],
                "warnings": ["Do not claim clinical validation.", "Cross-task findings are diagnostic and task-specific."],
                "next_step": "Do not proceed without freezing a new protocol.",
            }
        )
        e["artifacts"].update(
            {
                "report_json": rel(run_dir / "PHASE2E_CROSS_TASK_REPORT.json"),
                "report_markdown": rel(run_dir / "PHASE2E_CROSS_TASK_REPORT.md"),
                "predictions": "",
                "manifest": "",
            }
        )
        e["environment"].update(
            {
                "python": metadata.get("python_version"),
                "scikit_learn": metadata.get("sklearn_version"),
                "git_commit": metadata.get("git_commit"),
            }
        )
        entries.append(e)

    interp = report["interpretation"]
    conclusion = empty_entry(
        "phase2e_cross_task_transfer_conclusion",
        "Phase 2E",
        "Phase 2E cross-task transfer conclusion",
        report["status"],
        run_dir,
    )
    conclusion["data"].update(
        {
            "dataset_id": "androids_corpus",
            "task": "reading-interview cross-task comparison",
            "language": "Italian",
            "train_n": "participant-coherent cross-validation",
            "dev_n": None,
            "test_n": data_summary["paired_participants"],
            "class_counts": data_summary["class_counts"],
            "split_protocol": "new Phase 2E participant-coherent five-fold split over paired participants",
            "excluded": {
                "reading_only": data_summary.get("reading_only_participants"),
                "interview_only": data_summary.get("interview_only_participants"),
            },
        }
    )
    conclusion["representation"].update(common_representation)
    conclusion["classifier"].update(
        {
            "name": "Comparison of participant-safe logistic-regression protocols",
            "pipeline": "matched within-task, cross-task, and pooled-task evaluations",
            "class_weight": "balanced",
            "C": {row["source_model"]: row.get("selected_C_distribution") for row in report.get("selected_parameter_summary", [])},
            "selection_protocol": "pre-specified Phase 2E participant-coherent protocol",
        }
    )
    conclusion["metrics"]["test"].update(
        {
            "roc_auc": interp.get("reading_to_interview_auc"),
            "reading_to_interview_auc": interp.get("reading_to_interview_auc"),
            "interview_to_reading_auc": interp.get("interview_to_reading_auc"),
            "matched_reading_auc": interp.get("matched_reading_auc"),
            "matched_interview_auc": interp.get("matched_interview_auc"),
            "pooled_to_reading_auc": interp.get("pooled_to_reading_auc"),
            "pooled_to_interview_auc": interp.get("pooled_to_interview_auc"),
        }
    )
    conclusion["metrics"]["confidence_intervals"].update(
        {
            "source": rel(run_dir / "phase2e_bootstrap_intervals.csv"),
            "paired_comparisons": rel(run_dir / "phase2e_paired_comparisons.csv"),
            "transfer_gaps": rel(run_dir / "phase2e_transfer_gap_analysis.csv"),
        }
    )
    conclusion["integrity"].update(
        {
            "test_provenance": "Android clinical patient/control labels under Phase 2E participant-coherent folds",
            "external_validation": True,
            "zero_shot": False,
            "test_result_viewed": True,
            "target_model_refitted": True,
            "target_threshold_refitted": True,
            "test_used_for_selection": False,
            "decisions_made_after_test_view": ["No Phase 1, Phase 2C, or Phase 2D frozen decision was changed."],
            "leakage_status": report["leakage_checks"]["status"],
        }
    )
    conclusion["conclusion"].update(
        {
            "main_finding": f"Phase 2E interpretation category: {interp['category']}.",
            "selected": False,
            "selection_reason": "Conclusion record only.",
            "limitations": [
                "Fold-matched results are not replacements for official Phase 2C/2D task-specific benchmarks.",
                "No clinical-validity claim is made.",
            ],
            "warnings": ["Repeated external-result inspection must remain documented."],
            "next_step": "Freeze any future protocol before execution.",
        }
    )
    conclusion["artifacts"].update(
        {
            "report_json": rel(run_dir / "PHASE2E_CROSS_TASK_REPORT.json"),
            "report_markdown": rel(run_dir / "PHASE2E_CROSS_TASK_REPORT.md"),
            "manifest": rel(run_dir / "phase2e_protocol_comparison.csv"),
        }
    )
    conclusion["environment"].update(entries[-1]["environment"])
    entries.append(conclusion)
    return entries


def parse_run(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(rel(run_dir))
    if run_dir == OFFICIAL_RUNS["phase1a"]:
        return parse_phase1a(run_dir)
    if run_dir == OFFICIAL_RUNS["phase1b"]:
        return parse_phase1b(run_dir)
    if run_dir == OFFICIAL_RUNS["phase1c"]:
        return parse_phase1c(run_dir)
    if run_dir == OFFICIAL_RUNS["phase1d"]:
        return parse_phase1d(run_dir)
    if run_dir == OFFICIAL_RUNS["phase1e"]:
        return parse_phase1e(run_dir)
    if run_dir == OFFICIAL_RUNS["phase1f"]:
        return parse_phase1f(run_dir)
    if run_dir == OFFICIAL_RUNS["phase2a"]:
        return parse_phase2a(run_dir)
    if run_dir == OFFICIAL_RUNS["phase2b"]:
        return parse_phase2b(run_dir)
    if (run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.json").exists():
        return parse_phase2c(run_dir)
    if (run_dir / "PHASE2D_ANDROID_READING_REPORT.json").exists():
        return parse_phase2d(run_dir)
    if (run_dir / "PHASE2E_CROSS_TASK_REPORT.json").exists():
        return parse_phase2e(run_dir)
    return parse_generic_run(run_dir)


def parse_generic_run(run_dir: Path) -> list[dict[str, Any]]:
    report_json = next((p for p in run_dir.glob("*REPORT.json")), None) or next((p for p in run_dir.glob("evaluation_report.json")), None)
    if report_json is None:
        raise SystemExit(f"No supported report JSON found in {run_dir}")
    exp_id = "generic_" + run_dir.name
    e = empty_entry(exp_id, "Unknown", f"Generic imported run {run_dir.name}", "imported_needs_manual_review", run_dir)
    e["conclusion"].update({"main_finding": "Imported by generic parser; review and enrich this entry.", "warnings": ["Generic parser preserves path but cannot infer full scientific context."]})
    e["artifacts"]["report_json"] = rel(report_json)
    return [e]


def merge_entries(existing: list[dict[str, Any]], new_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["id"]: deepcopy(row) for row in existing}
    for new in new_entries:
        old = by_id.get(new["id"])
        if old is None:
            by_id[new["id"]] = new
            continue
        if old.get("run_directory") != new.get("run_directory") or old.get("status") != new.get("status"):
            history = old.get("history", [])
            old_copy = deepcopy(old)
            old_copy.pop("history", None)
            history.append({"recorded_at": utc_now(), "previous_entry": old_copy, "reason": "registry entry corrected or run replaced"})
            new["revision"] = int(old.get("revision", 1)) + 1
            new["history"] = history
        else:
            new["revision"] = old.get("revision", 1)
            new["history"] = old.get("history", [])
        by_id[new["id"]] = new
    return [by_id[key] for key in sorted(by_id)]


def build_official_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in OFFICIAL_RUNS.values():
        if not path.exists():
            raise SystemExit(f"Missing official run directory: {path}")
        entries.extend(parse_run(path))
    return entries


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    experiments = registry.get("experiments", [])
    ids = [row.get("id") for row in experiments]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate experiment IDs detected.")
    authoritative_runs = {row.get("run_directory") for row in experiments}
    if "outputs/phase2_android/phase2a_preparation_runs/20260622_154008" in authoritative_runs:
        errors.append("Superseded Phase 2A run is present as authoritative.")
    for exp in experiments:
        for field in ["id", "phase", "name", "status", "run_directory"]:
            if not exp.get(field):
                errors.append(f"{exp.get('id', '<missing>')} missing required identity field {field}.")
        if exp.get("conclusion", {}).get("selected") and "failed" in str(exp.get("status", "")).lower():
            errors.append(f"{exp['id']} is selected but marked failed.")
        if exp.get("metrics", {}).get("test") and not exp.get("integrity", {}).get("test_provenance"):
            errors.append(f"{exp['id']} has test metrics without test provenance.")
        if exp.get("integrity", {}).get("zero_shot") and not exp.get("integrity", {}).get("external_validation"):
            errors.append(f"{exp['id']} is zero-shot but not marked external validation.")
        for path_field in ["run_directory"]:
            value = exp.get(path_field)
            if isinstance(value, str) and (Path(value).is_absolute() or ":" in value[:3]):
                errors.append(f"{exp['id']} has non-relative {path_field}: {value}")
        for artifact_name, value in exp.get("artifacts", {}).items():
            if value and (Path(str(value)).is_absolute() or ":" in str(value)[:3]):
                errors.append(f"{exp['id']} artifact {artifact_name} is not project-relative: {value}")
        validate_metrics(exp.get("metrics", {}), exp.get("id", ""), errors)
        validate_paper_interpretation(exp, errors)
    official_dirs = [str(path) for path in OFFICIAL_RUNS.values()]
    if len(official_dirs) != len(set(official_dirs)):
        errors.append("Official run directory list contains duplicates.")
    if len(experiments) < 28:
        warnings.append(f"Expected at least 28 experiment/protocol entries; found {len(experiments)}.")
    return {"status": "passed" if not errors else "failed", "checked_at": utc_now(), "errors": errors, "warnings": warnings}


def validate_metrics(metrics: dict[str, Any], exp_id: str, errors: list[str]) -> None:
    metric_keys = {
        "roc_auc",
        "roc_auc_std",
        "pr_auc",
        "balanced_accuracy",
        "macro_f1",
        "sensitivity",
        "specificity",
        "positive_precision",
        "negative_predictive_value",
        "threshold",
        "p2_5",
        "median",
        "p97_5",
    }

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in metric_keys and isinstance(value, (int, float)) and not (0 <= float(value) <= 1):
                    errors.append(f"{exp_id} metric {prefix + key} outside [0,1]: {value}")
                walk(value, prefix + key + ".")
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)

    walk(metrics)


def validate_paper_interpretation(exp: dict[str, Any], errors: list[str]) -> None:
    exp_id = exp.get("id", "<missing>")
    p = exp.get("paper_interpretation")
    if not isinstance(p, dict):
        errors.append(f"{exp_id} missing paper_interpretation block.")
        return

    missing = [field for field in PAPER_INTERPRETATION_FIELDS if field not in p]
    if missing:
        errors.append(f"{exp_id} missing paper interpretation fields: {', '.join(missing)}.")

    required_nonempty = [
        "result_scope",
        "training_dataset",
        "evaluation_dataset",
        "evaluation_protocol",
        "participant_level_split",
        "paper_ready_result_sentence",
        "clinical_status",
    ]
    for field in required_nonempty:
        if p.get(field) in (None, "", []):
            errors.append(f"{exp_id} has empty paper interpretation field {field}.")

    for field in [
        "target_labels_used_for_training",
        "target_labels_used_for_model_selection",
        "target_labels_used_for_threshold_selection",
        "external_validation",
        "cross_corpus",
        "zero_shot",
        "target_domain_refitting",
        "calibration_performed",
        "threshold_adaptation_performed",
    ]:
        if not isinstance(p.get(field), bool):
            errors.append(f"{exp_id} paper interpretation field {field} must be boolean.")

    if p.get("result_scope") not in RESULT_SCOPES:
        errors.append(f"{exp_id} has unknown result_scope: {p.get('result_scope')}.")

    if not p.get("allowed_claims"):
        errors.append(f"{exp_id} needs at least one allowed paper claim.")
    if not p.get("prohibited_claims"):
        errors.append(f"{exp_id} needs at least one prohibited paper claim.")

    scope = p.get("result_scope")
    if scope in {
        "methodological_ablation",
        "within_corpus_cross_validation",
        "within_corpus_held_out_test",
        "provisional_local_labeled_test",
        "cross_task_transfer",
        "pooled_task_within_corpus_cross_validation",
        "data_quality_audit",
    } and p.get("external_validation"):
        errors.append(f"{exp_id} marks a non-external scope as external validation.")

    if scope == "cross_corpus_zero_shot_external_evaluation" and not (p.get("external_validation") and p.get("cross_corpus") and p.get("zero_shot")):
        errors.append(f"{exp_id} zero-shot external scope is missing external/cross-corpus/zero-shot flags.")

    if p.get("target_domain_refitting") and p.get("zero_shot"):
        errors.append(f"{exp_id} marks target-domain refitting as zero-shot.")
    if p.get("threshold_adaptation_performed") and p.get("zero_shot"):
        errors.append(f"{exp_id} marks target-label threshold adaptation as pure zero-shot.")

    paper_text = " ".join(
        [
            str(p.get("paper_ready_result_sentence", "")),
            " ".join(str(item) for item in p.get("allowed_claims", [])),
            str(p.get("evaluation_protocol", "")),
            str(p.get("participant_level_split", "")),
        ]
    ).lower()
    if scope == "provisional_local_labeled_test" and ("official avec" in paper_text or "official_avec" in paper_text):
        errors.append(f"{exp_id} describes provisional DAIC testing as official AVEC test evaluation.")
    if p.get("clinical_status") != "clinically_validated" and any(term in paper_text for term in ["diagnostic", "diagnosis", "clinical deployment"]):
        errors.append(f"{exp_id} uses diagnostic wording despite not being clinically validated.")
    if scope == "cross_task_transfer" and any(term in paper_text for term in ["cross-corpus", "cross-dataset"]):
        errors.append(f"{exp_id} describes cross-task transfer as cross-corpus transfer.")


def summary_value(exp: dict[str, Any], section: str, key: str) -> Any:
    return get_path(exp, "metrics", section, key)


def best_metric(exp: dict[str, Any], key: str) -> Any:
    for section in ["test", "dev", "train_cv"]:
        value = summary_value(exp, section, key)
        if value not in (None, ""):
            return value
    return None


def csv_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry["experiments"]:
        rep = exp["representation"]
        clf = exp["classifier"]
        paper = exp.get("paper_interpretation", {})
        rows.append(
            {
                "experiment_id": exp["id"],
                "phase": exp["phase"],
                "experiment_name": exp["name"],
                "status": exp["status"],
                "result_scope": paper.get("result_scope"),
                "dataset": exp["data"].get("dataset_id"),
                "task": exp["data"].get("task"),
                "representation": rep.get("feature_family"),
                "wavlm_layer": rep.get("layer"),
                "aggregation": rep.get("participant_aggregation"),
                "classifier": clf.get("name"),
                "selected_C": clf.get("C"),
                "train_cv_auc": summary_value(exp, "train_cv", "roc_auc"),
                "train_cv_auc_std": summary_value(exp, "train_cv", "roc_auc_std"),
                "dev_auc": summary_value(exp, "dev", "roc_auc"),
                "test_auc": summary_value(exp, "test", "roc_auc"),
                "pr_auc": best_metric(exp, "pr_auc"),
                "balanced_accuracy": best_metric(exp, "balanced_accuracy"),
                "macro_f1": best_metric(exp, "macro_f1"),
                "sensitivity": best_metric(exp, "sensitivity"),
                "specificity": best_metric(exp, "specificity"),
                "threshold": exp["threshold"].get("value") if exp["threshold"].get("value") is not None else best_metric(exp, "threshold"),
                "evaluation_type": "external_zero_shot" if exp["integrity"].get("zero_shot") else "provisional_test" if exp["metrics"].get("test") else "train_dev_or_audit",
                "clinical_status": paper.get("clinical_status"),
                "selected": exp["conclusion"].get("selected"),
                "main_finding": exp["conclusion"].get("main_finding"),
                "paper_ready_result_sentence": paper.get("paper_ready_result_sentence"),
                "limitation": "; ".join(exp["conclusion"].get("limitations", [])[:2]),
                "run_directory": exp["run_directory"],
            }
        )
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def md_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def paper_matrix_path() -> Path:
    return project_root() / "docs" / "PAPER_RESULTS_INTERPRETATION_MATRIX.md"


def generate_paper_matrix(registry: dict[str, Any]) -> str:
    lines = [
        "# Paper Results Interpretation Matrix",
        "",
        "_Generated automatically from `outputs/experiment_registry/experiment_registry.json` by `update_registry.py`._",
        "",
        "This matrix separates methodological ablations, within-corpus estimates, provisional local test results, zero-shot external validation, target-domain adaptation and cross-task transfer. It is intended to prevent paper text from overstating what each completed experiment can support.",
        "",
        "| Experiment ID | Phase | Experiment | Scope | Train -> Evaluate | Protocol | Primary metric | Paper-ready interpretation | Allowed claim | Do not claim | Clinical status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp in registry.get("experiments", []):
        p = exp.get("paper_interpretation", {})
        metric_name = p.get("primary_metric_name")
        metric_value = fmt(p.get("primary_metric_value"), 4)
        metric = f"{metric_name}: {metric_value}" if metric_name and metric_value else md_cell(metric_name)
        train_eval = f"{p.get('training_dataset', '')} / {p.get('training_task', '')} -> {p.get('evaluation_dataset', '')} / {p.get('evaluation_task', '')}"
        allowed = p.get("allowed_claims", [""])[0] if p.get("allowed_claims") else ""
        prohibited = p.get("prohibited_claims", [""])[0] if p.get("prohibited_claims") else ""
        lines.append(
            "| "
            + " | ".join(
                md_cell(value)
                for value in [
                    exp.get("id"),
                    exp.get("phase"),
                    exp.get("name"),
                    p.get("result_scope"),
                    train_eval,
                    p.get("evaluation_protocol"),
                    metric,
                    p.get("paper_ready_result_sentence"),
                    allowed,
                    prohibited,
                    p.get("clinical_status"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def generate_markdown(registry: dict[str, Any]) -> str:
    rows = csv_rows(registry)
    selected = [exp for exp in registry["experiments"] if exp["conclusion"].get("selected")]
    rejected = [exp for exp in registry["experiments"] if exp["conclusion"].get("rejection_reason")]
    lines = [
        "# DepressionFromVoice Experiment Log",
        "",
        "_Generated automatically from `experiment_registry.json` by `update_registry.py`._",
        "",
        "## 1. Current Research Status",
        "",
        f"- Phase 1 status: `{registry['project_status']['phase1_status']}`",
        f"- Phase 2 status: `{registry['project_status']['phase2_status']}`",
        "- Current candidates are research candidates only; no universal best model is declared.",
        f"- Accuracy-hunting warning: {ACCURACY_HUNTING_WARNING}",
        "",
        "## 2. Dataset Summary",
        "",
        "| Dataset | Participants | Labels | Language | Notes |",
        "|---|---:|---|---|---|",
    ]
    d1 = registry["datasets"]["daic_woz_phase1_clean"]
    d2 = registry["datasets"]["androids_corpus"]
    lines.append(f"| DAIC-WOZ Phase 1 clean | {d1['usable_participants']} | {d1['labels']} | {d1['language']} | 17,058 participant-only 10s windows; provisional local labeled test |")
    lines.append(f"| Androids Corpus | {d2['participants']} | {d2['labels']} | {d2['language']} | 112 reading, 116 interview, 874 clips; {d2['fold_warning']} |")
    lines.extend(["", "## 3. Chronological Experiment Table", "", "| Phase | Method | Train AUC | Dev AUC | Test/External AUC | Decision | Key Finding |", "|---|---|---:|---:|---:|---|---|"])
    for row in rows:
        decision = "selected" if row["selected"] in {True, "True", "true"} else "not selected"
        if row["phase"] == "Phase 2A":
            decision = "audit"
        lines.append(
            f"| {row['phase']} | {row['experiment_name']} | {fmt(row['train_cv_auc'])} | {fmt(row['dev_auc'])} | {fmt(row['test_auc'])} | {decision} | {row['main_finding']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Guidance For Paper Reporting",
            "",
            "| Experiment | Scope | External? | Zero-shot? | Paper-ready interpretation | Do Not Claim |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for exp in registry["experiments"]:
        p = exp.get("paper_interpretation", {})
        prohibited = "; ".join(p.get("prohibited_claims", [])[:2])
        lines.append(
            f"| {md_cell(exp['name'])} | {md_cell(p.get('result_scope'))} | {fmt(p.get('external_validation'))} | {fmt(p.get('zero_shot'))} | {md_cell(p.get('paper_ready_result_sentence'))} | {md_cell(prohibited)} |"
        )
    lines.extend(["", "## 5. Selected And Rejected Approaches", "", "### Selected / Frozen Research Candidates", "", "| Experiment | Why It Was Kept | Limitation |", "|---|---|---|"])
    for exp in selected:
        lines.append(f"| {exp['name']} | {exp['conclusion'].get('selection_reason', '')} | {'; '.join(exp['conclusion'].get('limitations', [])[:2])} |")
    lines.extend(["", "### Rejected Or Not Advanced", "", "| Experiment | Reason |", "|---|---|"])
    for exp in rejected:
        lines.append(f"| {exp['name']} | {exp['conclusion'].get('rejection_reason')} |")
    lines.extend(
        [
            "",
            "## 6. Current Conclusions",
            "",
            "- WavLM layer_09 is the frozen free-speech DAIC representation.",
            "- Mean participant aggregation was retained; aggregation variants did not appear to be the main bottleneck.",
            "- DAIC timing features are useful inside guided-interview DAIC conditions but do not directly transfer to Android because required response-latency features are unavailable.",
            "- Phase 2B showed Android ranking signal for frozen DAIC WavLM, but the DAIC probability scale and threshold failed externally.",
            "",
            "## 7. Important Limitations",
            "",
            "- DAIC test labels are provisional local labeled test labels, not confirmed official AVEC test-label provenance.",
            "- DAIC PHQ8_Binary and Android clinical patient/control status are related but not equivalent.",
            "- Android cross-task folds are inconsistent for 85 participants; keep task protocols explicit.",
            "- Repeated inspection of the same test/external data must not drive model development.",
            "",
            "## 8. Next Planned Experiment",
            "",
            "| Planned phase | Protocols | Purpose |",
            "|---|---|---|",
        ]
    )
    for plan in registry["planned_experiments"]:
        lines.append(f"| {plan['phase']} | {'; '.join(plan['protocols'])} | {'; '.join(plan['purpose'])} |")
    return "\n".join(lines) + "\n"


def save_registry(registry: dict[str, Any]) -> None:
    registry["last_updated"] = utc_now()
    apply_paper_interpretations(registry)
    registry["validation"] = validate_registry(registry)
    write_json(registry_dir() / "experiment_registry.json", registry)
    write_csv(registry_dir() / "experiment_summary.csv", csv_rows(registry))
    (registry_dir() / "EXPERIMENT_LOG.md").write_text(generate_markdown(registry), encoding="utf-8")
    paper_matrix_path().write_text(generate_paper_matrix(registry), encoding="utf-8")


def validate_existing_artifacts(registry: dict[str, Any], validation: dict[str, Any]) -> None:
    summary_path = registry_dir() / "experiment_summary.csv"
    log_path = registry_dir() / "EXPERIMENT_LOG.md"
    matrix_path = paper_matrix_path()
    expected_ids = [str(exp.get("id", "")) for exp in registry.get("experiments", [])]

    if not summary_path.exists():
        validation["errors"].append(f"Missing registry summary: {summary_path}")
    else:
        try:
            summary_rows = read_csv(summary_path)
            summary_ids = [row.get("experiment_id", "") for row in summary_rows]
            if summary_ids != expected_ids:
                validation["errors"].append("experiment_summary.csv experiment order does not match experiment_registry.json.")
        except (csv.Error, OSError, UnicodeDecodeError) as exc:
            validation["errors"].append(f"Could not read experiment_summary.csv: {exc}")

    if not log_path.exists():
        validation["errors"].append(f"Missing experiment log: {log_path}")
    else:
        try:
            log_text = log_path.read_text(encoding="utf-8")
            if not log_text.startswith("# DepressionFromVoice Experiment Log"):
                validation["errors"].append("EXPERIMENT_LOG.md does not look like the generated experiment log.")
        except (OSError, UnicodeDecodeError) as exc:
            validation["errors"].append(f"Could not read EXPERIMENT_LOG.md: {exc}")

    if not matrix_path.exists():
        validation["errors"].append(f"Missing paper interpretation matrix: {matrix_path}")
    else:
        try:
            matrix_text = matrix_path.read_text(encoding="utf-8")
            if not matrix_text.startswith("# Paper Results Interpretation Matrix"):
                validation["errors"].append("PAPER_RESULTS_INTERPRETATION_MATRIX.md does not look like the generated paper matrix.")
            missing_ids = [exp_id for exp_id in expected_ids if exp_id and exp_id not in matrix_text]
            if missing_ids:
                validation["errors"].append("PAPER_RESULTS_INTERPRETATION_MATRIX.md is missing experiment IDs: " + ", ".join(missing_ids[:5]))
        except (OSError, UnicodeDecodeError) as exc:
            validation["errors"].append(f"Could not read PAPER_RESULTS_INTERPRETATION_MATRIX.md: {exc}")

    validation["status"] = "passed" if not validation["errors"] else "failed"


def print_validation(registry: dict[str, Any], validation: dict[str, Any]) -> None:
    print(f"registry={registry_dir() / 'experiment_registry.json'}")
    print(f"experiments={len(registry.get('experiments', []))}")
    print(f"validation_status={validation['status']}")
    if validation["errors"]:
        print("errors=" + "; ".join(validation["errors"]))
    if validation["warnings"]:
        print("warnings=" + "; ".join(validation["warnings"]))


def validate_registry_read_only() -> int:
    registry = load_registry()
    validation = validate_registry(registry)
    validate_existing_artifacts(registry, validation)
    print_validation(registry, validation)
    return 0 if validation["status"] == "passed" else 1


def load_registry() -> dict[str, Any]:
    path = registry_dir() / "experiment_registry.json"
    if path.exists():
        return read_json(path)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="Scan official completed run directories and regenerate registry artifacts.")
    parser.add_argument("--run-dir", default="", help="Add or update one run directory.")
    parser.add_argument("--validate", action="store_true", help="Validate existing registry artifacts without modifying them.")
    args = parser.parse_args()

    if not (args.scan or args.run_dir or args.validate):
        parser.error("Use --scan, --run-dir <path>, or --validate.")

    if args.validate and not (args.scan or args.run_dir):
        raise SystemExit(validate_registry_read_only())

    registry = base_registry(load_registry())
    if args.scan:
        registry["experiments"] = merge_entries(registry.get("experiments", []), build_official_entries())
    if args.run_dir:
        run_path = Path(args.run_dir)
        if run_path.is_absolute():
            try:
                run_path = run_path.resolve().relative_to(project_root().resolve())
            except ValueError:
                pass
        registry["experiments"] = merge_entries(registry.get("experiments", []), parse_run(run_path))
    save_registry(registry)
    validation = registry["validation"]
    print_validation(registry, validation)
    if validation["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
