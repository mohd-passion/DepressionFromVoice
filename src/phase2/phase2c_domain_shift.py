"""Label-independent DAIC-to-Android embedding-shift diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
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


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - (np.dot(a, b) / denom))


def rbf_mmd2_biased(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float | str]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    combined = np.vstack([x, y])
    diff = combined[:, None, :] - combined[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    nonzero = sqdist[sqdist > 0]
    sigma2 = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / (2.0 * sigma2) if sigma2 > 0 else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(d * d, axis=2))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    value = float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))
    return value, {"kernel": "rbf", "estimator": "biased", "sigma2": sigma2, "gamma": gamma}


def embedding_shift_report(
    daic_train_embeddings: np.ndarray,
    android_embeddings: np.ndarray,
    daic_scores: np.ndarray,
    android_scores: np.ndarray,
    daic_ids: list[str],
    android_ids: list[str],
) -> dict[str, Any]:
    daic = np.asarray(daic_train_embeddings, dtype=float)
    android = np.asarray(android_embeddings, dtype=float)
    if daic.shape[1] != android.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: DAIC {daic.shape}, Android {android.shape}")
    daic_mean = np.mean(daic, axis=0)
    android_mean = np.mean(android, axis=0)
    daic_std = np.std(daic, axis=0, ddof=0)
    android_std = np.std(android, axis=0, ddof=0)
    safe_daic_std = np.where(daic_std > 1e-12, daic_std, np.nan)
    standardized_shift = (android_mean - daic_mean) / safe_daic_std
    abs_shift = np.abs(standardized_shift)
    shifted_mask = abs_shift > 1.0
    mmd2, mmd_settings = rbf_mmd2_biased(daic, android)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    daic_pca = pca.fit_transform(daic)
    android_pca = pca.transform(android)

    top_shift_indices = np.argsort(np.nan_to_num(abs_shift, nan=-1.0))[::-1][:20]
    feature_rows = []
    for idx in top_shift_indices:
        feature_rows.append(
            {
                "feature_index": int(idx),
                "daic_mean": float(daic_mean[idx]),
                "android_mean": float(android_mean[idx]),
                "mean_shift": float(android_mean[idx] - daic_mean[idx]),
                "daic_std": float(daic_std[idx]),
                "android_std": float(android_std[idx]),
                "std_shift": float(android_std[idx] - daic_std[idx]),
                "standardized_mean_shift": float(standardized_shift[idx]),
            }
        )

    return {
        "daic_train_count": int(daic.shape[0]),
        "android_interview_count": int(android.shape[0]),
        "embedding_dimension": int(daic.shape[1]),
        "embedding_norms": {
            "daic_train": _summary(np.linalg.norm(daic, axis=1)),
            "android_interview": _summary(np.linalg.norm(android, axis=1)),
        },
        "centroid_cosine_distance": cosine_distance(daic_mean, android_mean),
        "per_feature_shift": {
            "mean_absolute_standardized_shift": float(np.nanmean(abs_shift)),
            "median_absolute_standardized_shift": float(np.nanmedian(abs_shift)),
            "features_shifted_more_than_one_daic_std": int(np.nansum(shifted_mask)),
            "fraction_features_shifted_more_than_one_daic_std": float(np.nansum(shifted_mask) / daic.shape[1]),
            "top_abs_standardized_shift_features": feature_rows,
        },
        "frozen_daic_classifier_score_distributions": {
            "daic_train": _summary(daic_scores),
            "android_interview": _summary(android_scores),
        },
        "pca_fitted_on_daic_train_only": {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "daic_train_coordinate_summary": {"pc1": _summary(daic_pca[:, 0]), "pc2": _summary(daic_pca[:, 1])},
            "android_coordinate_summary": {"pc1": _summary(android_pca[:, 0]), "pc2": _summary(android_pca[:, 1])},
            "participant_coordinates": [
                {"corpus": "DAIC_train", "participant_id": str(pid), "pc1": float(coord[0]), "pc2": float(coord[1])}
                for pid, coord in zip(daic_ids, daic_pca)
            ]
            + [
                {"corpus": "Android_interview", "participant_id": str(pid), "pc1": float(coord[0]), "pc2": float(coord[1])}
                for pid, coord in zip(android_ids, android_pca)
            ],
        },
        "maximum_mean_discrepancy": {"mmd2": mmd2, **mmd_settings},
    }


def summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus, stats in report["embedding_norms"].items():
        rows.append({"metric_group": "embedding_norm", "metric": corpus, **stats})
    for corpus, stats in report["frozen_daic_classifier_score_distributions"].items():
        rows.append({"metric_group": "frozen_daic_score", "metric": corpus, **stats})
    rows.append(
        {
            "metric_group": "centroid",
            "metric": "cosine_distance",
            "value": report["centroid_cosine_distance"],
        }
    )
    rows.append(
        {
            "metric_group": "feature_shift",
            "metric": "mean_absolute_standardized_shift",
            "value": report["per_feature_shift"]["mean_absolute_standardized_shift"],
        }
    )
    rows.append(
        {
            "metric_group": "feature_shift",
            "metric": "features_shifted_more_than_one_daic_std",
            "value": report["per_feature_shift"]["features_shifted_more_than_one_daic_std"],
        }
    )
    rows.append(
        {
            "metric_group": "mmd",
            "metric": "rbf_mmd2_biased",
            "value": report["maximum_mean_discrepancy"]["mmd2"],
            "sigma2": report["maximum_mean_discrepancy"]["sigma2"],
            "gamma": report["maximum_mean_discrepancy"]["gamma"],
        }
    )
    for row in report["per_feature_shift"]["top_abs_standardized_shift_features"]:
        out = {"metric_group": "top_feature_shift", "metric": f"feature_{row['feature_index']}"}
        out.update(row)
        rows.append(out)
    return rows
