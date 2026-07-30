"""Compute Android reading-versus-interview MMD^2 for IEEE TCSS paper audit.

This script is intentionally paper-scoped: it reads existing participant-level
WavLM embedding records and writes aggregate audit outputs only. It does not
rerun audio preprocessing, WavLM extraction, classifier training, or registry
updates.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2.phase2c_domain_shift import cosine_distance, rbf_mmd2_biased  # noqa: E402


EXPECTED_DIM = 768
READING_MANIFEST = PROJECT_ROOT / "outputs/phase2_android/phase2d_reading_runs/20260704_125931/android_reading_participant_embedding_manifest.csv"
INTERVIEW_MANIFEST = PROJECT_ROOT / "outputs/phase2_android/phase2b_zero_shot_interview_runs/20260622_162548/android_interview_participant_embedding_manifest.csv"
PHASE2D_TASK_SHIFT_REPORT = PROJECT_ROOT / "outputs/phase2_android/phase2d_reading_runs/20260704_125931/phase2d_task_shift_report.json"
PHASE2C_DOMAIN_REPORT = PROJECT_ROOT / "outputs/phase2_android/phase2c_domain_shift_runs/20260703_125614/daic_android_domain_shift_report.json"
OUT_DIR = PROJECT_ROOT / "paper/ieee_tcss/mmd2"
JSON_OUT = OUT_DIR / "android_reading_interview_mmd2.json"
CSV_OUT = OUT_DIR / "android_reading_interview_mmd2.csv"
SUMMARY_OUT = OUT_DIR / "android_reading_interview_mmd2_summary.md"
AUDIT_OUT = PROJECT_ROOT / "paper/ieee_tcss/ANDROID_READING_INTERVIEW_MMD2_AUDIT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def rel_to_root(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def participant_sort_key(pid: str) -> tuple[str, int, str]:
    match = re.match(r"([A-Za-z]+)_?(\d+)$", pid)
    if not match:
        return (pid, -1, pid)
    return (match.group(1), int(match.group(2)), pid)


def load_manifest_embeddings(path: Path, expected_task: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    loaded: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row["participant_id"]
        if row.get("aggregation_status", "success") != "success":
            continue
        if row.get("task") != expected_task:
            raise ValueError(f"Unexpected task for {pid}: {row.get('task')}")
        if row.get("source_embedding_layer") != "layer_09":
            raise ValueError(f"Unexpected WavLM layer for {pid}: {row.get('source_embedding_layer')}")
        if int(row.get("embedding_dimension", "0")) != EXPECTED_DIM:
            raise ValueError(f"Unexpected embedding dimension for {pid}: {row.get('embedding_dimension')}")
        embedding_path = PROJECT_ROOT / row["participant_embedding_path"]
        embedding = np.load(embedding_path).astype(np.float32)
        if embedding.shape != (EXPECTED_DIM,):
            raise ValueError(f"Bad embedding shape for {pid}: {embedding.shape}")
        if not np.isfinite(embedding).all() or np.all(embedding == 0):
            raise ValueError(f"Invalid embedding values for {pid}")
        loaded[pid] = {"row": row, "embedding": embedding}
    return loaded


def matrix_for_ids(loaded: dict[str, dict[str, Any]], ids: list[str]) -> np.ndarray:
    return np.stack([loaded[pid]["embedding"] for pid in ids], axis=0).astype(np.float32)


def summary_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def feature_shift(reading: np.ndarray, interview: np.ndarray) -> dict[str, float | int]:
    interview_std = np.std(interview, axis=0, ddof=0)
    safe_std = np.where(interview_std > 1e-12, interview_std, np.nan)
    standardized = (np.mean(reading, axis=0) - np.mean(interview, axis=0)) / safe_std
    abs_shift = np.abs(standardized)
    return {
        "standardization_reference": "interview_std_ddof0",
        "mean_absolute_standardized_shift": float(np.nanmean(abs_shift)),
        "median_absolute_standardized_shift": float(np.nanmedian(abs_shift)),
        "features_shifted_more_than_one_interview_std": int(np.nansum(abs_shift > 1.0)),
        "fraction_features_shifted_more_than_one_interview_std": float(np.nansum(abs_shift > 1.0) / reading.shape[1]),
    }


def comparison_result(
    name: str,
    sample_selection: str,
    reading: np.ndarray,
    interview: np.ndarray,
) -> dict[str, Any]:
    mmd2, settings = rbf_mmd2_biased(reading, interview)
    reading_mean = np.mean(reading, axis=0)
    interview_mean = np.mean(interview, axis=0)
    return {
        "comparison": name,
        "sample_selection": sample_selection,
        "reading_count": int(reading.shape[0]),
        "interview_count": int(interview.shape[0]),
        "embedding_dimension": int(reading.shape[1]),
        "scaling": "none_for_mmd2_raw_participant_embeddings",
        "mmd2": mmd2,
        **settings,
        "task_centroid_cosine_distance": cosine_distance(reading_mean, interview_mean),
        "embedding_norms": {
            "reading": summary_stats(np.linalg.norm(reading, axis=1)),
            "interview": summary_stats(np.linalg.norm(interview, axis=1)),
        },
        "standardized_feature_shift_using_interview_std": feature_shift(reading, interview),
    }


def write_csv_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "sample_selection",
        "reading_count",
        "interview_count",
        "embedding_dimension",
        "scaling",
        "kernel",
        "estimator",
        "sigma2",
        "gamma",
        "mmd2",
        "task_centroid_cosine_distance",
        "mean_absolute_standardized_shift",
        "features_shifted_more_than_one_interview_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            shift = result["standardized_feature_shift_using_interview_std"]
            writer.writerow(
                {
                    "comparison": result["comparison"],
                    "sample_selection": result["sample_selection"],
                    "reading_count": result["reading_count"],
                    "interview_count": result["interview_count"],
                    "embedding_dimension": result["embedding_dimension"],
                    "scaling": result["scaling"],
                    "kernel": result["kernel"],
                    "estimator": result["estimator"],
                    "sigma2": result["sigma2"],
                    "gamma": result["gamma"],
                    "mmd2": result["mmd2"],
                    "task_centroid_cosine_distance": result["task_centroid_cosine_distance"],
                    "mean_absolute_standardized_shift": shift["mean_absolute_standardized_shift"],
                    "features_shifted_more_than_one_interview_std": shift["features_shifted_more_than_one_interview_std"],
                }
            )


def markdown_summary(payload: dict[str, Any]) -> str:
    primary = payload["results"]["primary_paired_common_participants"]
    sensitivity = payload["results"]["sensitivity_all_available_participants"]
    return f"""# Android Reading/Interview MMD2 Summary

Computed at: `{payload['created_at']}`

## Primary Result

| Item | Value |
| --- | --- |
| Comparison | Android reading versus Android interview |
| Sample selection | Paired participants with both tasks |
| Reading participants | {primary['reading_count']} |
| Interview participants | {primary['interview_count']} |
| Embedding dimension | {primary['embedding_dimension']} |
| Kernel | {primary['kernel']} |
| Estimator | {primary['estimator']} |
| Scaling for MMD2 | {primary['scaling']} |
| Sigma squared | {primary['sigma2']:.12g} |
| Gamma | {primary['gamma']:.12g} |
| MMD2 | {primary['mmd2']:.12g} |
| Centroid cosine distance | {primary['task_centroid_cosine_distance']:.12g} |

The primary MMD2 value is methodologically compatible with the DAIC-to-Android MMD2 computation because it uses the same participant-level frozen WavLM layer-9 embeddings, the same raw embedding scale, the same biased RBF MMD2 estimator and the same median squared-distance bandwidth rule. It is not the same experimental comparison: DAIC-to-Android compares corpora, whereas this estimate compares paired Android reading and interview tasks.

## Sensitivity Result

| Item | Value |
| --- | --- |
| Sample selection | All available reading and interview participants |
| Reading participants | {sensitivity['reading_count']} |
| Interview participants | {sensitivity['interview_count']} |
| MMD2 | {sensitivity['mmd2']:.12g} |

The all-available estimate is provided as a sensitivity check. The paired-participant estimate is preferred for Figure 5 because the existing Android reading/interview task-shift metrics in Phase 2D were computed over common participants with both task recordings.
"""


def audit_markdown(payload: dict[str, Any]) -> str:
    primary = payload["results"]["primary_paired_common_participants"]
    sensitivity = payload["results"]["sensitivity_all_available_participants"]
    method = payload["recovered_daic_android_mmd2_method"]
    return f"""# Android Reading/Interview MMD2 Audit

Stage: Android reading/interview MMD2 analysis for IEEE TCSS

Branch: `paper/android-reading-interview-mmd2-v1`

## Source Records

| Record | Path |
| --- | --- |
| Reading participant manifest | `{payload['source_files']['reading_manifest']}` |
| Interview participant manifest | `{payload['source_files']['interview_manifest']}` |
| Phase 2D task-shift report | `{payload['source_files']['phase2d_task_shift_report']}` |
| Phase 2C DAIC-to-Android domain-shift report | `{payload['source_files']['phase2c_domain_shift_report']}` |

## Recovered DAIC-to-Android MMD2 Method

| Method Detail | Value |
| --- | --- |
| Existing DAIC-to-Android MMD2 | {method['daic_to_android_mmd2']:.12g} |
| Kernel | {method['kernel']} |
| Estimator | {method['estimator']} |
| Bandwidth rule | {method['bandwidth_rule']} |
| Scaling for MMD2 | {method['scaling']} |
| Feature level | {method['feature_level']} |
| Embedding dimension | {method['embedding_dimension']} |

## Computed Android Reading/Interview MMD2

| Result | Reading n | Interview n | Dimension | MMD2 | Sigma2 | Gamma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary paired common participants | {primary['reading_count']} | {primary['interview_count']} | {primary['embedding_dimension']} | {primary['mmd2']:.12g} | {primary['sigma2']:.12g} | {primary['gamma']:.12g} |
| Sensitivity all available participants | {sensitivity['reading_count']} | {sensitivity['interview_count']} | {sensitivity['embedding_dimension']} | {sensitivity['mmd2']:.12g} | {sensitivity['sigma2']:.12g} | {sensitivity['gamma']:.12g} |

## Comparability Assessment

The primary paired-participant Android reading/interview MMD2 is directly comparable in estimator, kernel, bandwidth rule, feature dimensionality, participant-level aggregation and raw embedding scale to the DAIC-to-Android MMD2 value. It is not a like-for-like corpus-transfer result because the DAIC-to-Android value compares DAIC train participants with Android interview participants, whereas this value compares reading and interview task distributions inside Android for participants with both task recordings.

Primary value for paper use: `{primary['mmd2']:.4f}`.

## Limitations

- The value is a new aggregate analysis computed from existing feature records, not a new classifier result.
- It does not change any DAIC-to-Android MMD2 value, classifier metric, label, dataset count or table value.
- Participant-level embeddings are read locally, but the outputs written here are aggregate audit records only.
- The paired-participant estimate is preferred for Figure 5 because existing Android task-shift metrics were computed on common participants.

## Scope Controls

- No audio preprocessing was run.
- No WavLM extraction was run.
- No classifier training or evaluation was run.
- No experiment registry record was modified.
- No references, author details, clinical claims, deployment claims or speaker-disentanglement claims were changed.
"""


def main() -> int:
    reading = load_manifest_embeddings(READING_MANIFEST, "reading")
    interview = load_manifest_embeddings(INTERVIEW_MANIFEST, "interview")
    common_ids = sorted(set(reading) & set(interview), key=participant_sort_key)
    reading_ids = sorted(reading, key=participant_sort_key)
    interview_ids = sorted(interview, key=participant_sort_key)

    if len(common_ids) != 110:
        raise ValueError(f"Expected 110 common reading/interview participants; found {len(common_ids)}")

    reading_common = matrix_for_ids(reading, common_ids)
    interview_common = matrix_for_ids(interview, common_ids)
    reading_all = matrix_for_ids(reading, reading_ids)
    interview_all = matrix_for_ids(interview, interview_ids)

    if reading_common.shape[1] != interview_common.shape[1]:
        raise ValueError(f"Primary dimension mismatch: {reading_common.shape} vs {interview_common.shape}")
    if reading_all.shape[1] != interview_all.shape[1]:
        raise ValueError(f"Sensitivity dimension mismatch: {reading_all.shape} vs {interview_all.shape}")

    phase2d_task = read_json(PHASE2D_TASK_SHIFT_REPORT)
    phase2c_domain = read_json(PHASE2C_DOMAIN_REPORT)
    phase2c_mmd = phase2c_domain["maximum_mean_discrepancy"]
    primary = comparison_result(
        "android_reading_vs_interview",
        "paired_common_participants_with_both_tasks",
        reading_common,
        interview_common,
    )
    sensitivity = comparison_result(
        "android_reading_vs_interview",
        "all_available_task_participants",
        reading_all,
        interview_all,
    )

    payload = {
        "created_at": utc_now(),
        "status": "computed",
        "source_files": {
            "reading_manifest": rel_to_root(READING_MANIFEST),
            "interview_manifest": rel_to_root(INTERVIEW_MANIFEST),
            "phase2d_task_shift_report": rel_to_root(PHASE2D_TASK_SHIFT_REPORT),
            "phase2c_domain_shift_report": rel_to_root(PHASE2C_DOMAIN_REPORT),
        },
        "recovered_daic_android_mmd2_method": {
            "daic_to_android_mmd2": float(phase2c_mmd["mmd2"]),
            "kernel": phase2c_mmd["kernel"],
            "estimator": phase2c_mmd["estimator"],
            "sigma2": float(phase2c_mmd["sigma2"]),
            "gamma": float(phase2c_mmd["gamma"]),
            "bandwidth_rule": "median_nonzero_squared_pairwise_distance_on_combined_samples",
            "scaling": "none_for_mmd2_raw_participant_embeddings",
            "feature_level": "participant_level_arithmetic_mean_wavlm_layer_09_embeddings",
            "embedding_dimension": int(phase2c_domain["embedding_dimension"]),
        },
        "phase2d_existing_task_shift_anchor": {
            "common_reading_interview_participants": int(phase2d_task["common_reading_interview_participants"]),
            "task_centroid_cosine_distance": float(phase2d_task["task_centroid_cosine_distance"]),
            "mean_absolute_standardized_shift_using_interview_std": float(
                phase2d_task["standardized_feature_shift_using_interview_std"]["mean_absolute_shift"]
            ),
            "features_shifted_more_than_one_interview_std": int(
                phase2d_task["standardized_feature_shift_using_interview_std"]["features_shifted_more_than_one_interview_std"]
            ),
        },
        "results": {
            "primary_paired_common_participants": primary,
            "sensitivity_all_available_participants": sensitivity,
        },
        "comparability_to_daic_android_mmd2": {
            "same_estimator_kernel_bandwidth_feature_level_and_raw_scale": True,
            "like_for_like_experimental_contrast": False,
            "recommended_for_figure5_android_task_shift": True,
            "recommended_value": float(primary["mmd2"]),
            "recommended_value_rounded_4dp": f"{primary['mmd2']:.4f}",
        },
        "scope_controls": {
            "audio_preprocessing_run": False,
            "wavlm_extraction_run": False,
            "classifier_training_run": False,
            "registry_modified": False,
            "existing_frozen_metrics_changed": False,
        },
    }

    write_json(JSON_OUT, payload)
    write_csv_summary(CSV_OUT, [primary, sensitivity])
    SUMMARY_OUT.write_text(markdown_summary(payload), encoding="utf-8", newline="\n")
    AUDIT_OUT.write_text(audit_markdown(payload), encoding="utf-8", newline="\n")
    print(f"primary_mmd2={primary['mmd2']:.12g}")
    print(f"sensitivity_mmd2={sensitivity['mmd2']:.12g}")
    print(f"json={rel_to_root(JSON_OUT)}")
    print(f"csv={rel_to_root(CSV_OUT)}")
    print(f"summary={rel_to_root(SUMMARY_OUT)}")
    print(f"audit={rel_to_root(AUDIT_OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
