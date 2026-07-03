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
    "selected",
    "main_finding",
    "limitation",
    "run_directory",
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
                "phase2_status": "phase2b_zero_shot_external_validation_completed",
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
        rows.append(
            {
                "experiment_id": exp["id"],
                "phase": exp["phase"],
                "experiment_name": exp["name"],
                "status": exp["status"],
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
                "selected": exp["conclusion"].get("selected"),
                "main_finding": exp["conclusion"].get("main_finding"),
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
    lines.extend(["", "## 4. Selected And Rejected Approaches", "", "### Selected / Frozen Research Candidates", "", "| Experiment | Why It Was Kept | Limitation |", "|---|---|---|"])
    for exp in selected:
        lines.append(f"| {exp['name']} | {exp['conclusion'].get('selection_reason', '')} | {'; '.join(exp['conclusion'].get('limitations', [])[:2])} |")
    lines.extend(["", "### Rejected Or Not Advanced", "", "| Experiment | Reason |", "|---|---|"])
    for exp in rejected:
        lines.append(f"| {exp['name']} | {exp['conclusion'].get('rejection_reason')} |")
    lines.extend(
        [
            "",
            "## 5. Current Conclusions",
            "",
            "- WavLM layer_09 is the frozen free-speech DAIC representation.",
            "- Mean participant aggregation was retained; aggregation variants did not appear to be the main bottleneck.",
            "- DAIC timing features are useful inside guided-interview DAIC conditions but do not directly transfer to Android because required response-latency features are unavailable.",
            "- Phase 2B showed Android ranking signal for frozen DAIC WavLM, but the DAIC probability scale and threshold failed externally.",
            "",
            "## 6. Important Limitations",
            "",
            "- DAIC test labels are provisional local labeled test labels, not confirmed official AVEC test-label provenance.",
            "- DAIC PHQ8_Binary and Android clinical patient/control status are related but not equivalent.",
            "- Android cross-task folds are inconsistent for 85 participants; keep task protocols explicit.",
            "- Repeated inspection of the same test/external data must not drive model development.",
            "",
            "## 7. Next Planned Experiment",
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
    registry["validation"] = validate_registry(registry)
    write_json(registry_dir() / "experiment_registry.json", registry)
    write_csv(registry_dir() / "experiment_summary.csv", csv_rows(registry))
    (registry_dir() / "EXPERIMENT_LOG.md").write_text(generate_markdown(registry), encoding="utf-8")


def load_registry() -> dict[str, Any]:
    path = registry_dir() / "experiment_registry.json"
    if path.exists():
        return read_json(path)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="Scan official completed run directories and regenerate registry artifacts.")
    parser.add_argument("--run-dir", default="", help="Add or update one run directory.")
    parser.add_argument("--validate", action="store_true", help="Validate and regenerate summary/log from the existing registry.")
    args = parser.parse_args()

    if not (args.scan or args.run_dir or args.validate):
        parser.error("Use --scan, --run-dir <path>, or --validate.")

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
    print(f"registry={registry_dir() / 'experiment_registry.json'}")
    print(f"experiments={len(registry['experiments'])}")
    print(f"validation_status={validation['status']}")
    if validation["errors"]:
        print("errors=" + "; ".join(validation["errors"]))


if __name__ == "__main__":
    main()
