from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "outputs" / "experiment_registry" / "experiment_registry.json"
SUMMARY_PATH = ROOT / "outputs" / "experiment_registry" / "experiment_summary.csv"
EXPERIMENT_LOG_PATH = ROOT / "outputs" / "experiment_registry" / "EXPERIMENT_LOG.md"
PAPER_MATRIX_PATH = ROOT / "docs" / "PAPER_RESULTS_INTERPRETATION_MATRIX.md"
UPDATE_REGISTRY_PATH = ROOT / "outputs" / "experiment_registry" / "update_registry.py"
SUPERSEDED_PHASE2A = "outputs/phase2_android/phase2a_preparation_runs/20260622_154008"
FORBIDDEN_SUFFIXES = {".wav", ".mp3", ".flac", ".mp4", ".zip", ".7z", ".rar", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".onnx"}
METRIC_KEYS = {
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


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from walk_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_values(item)


def is_relative_project_path(value: str) -> bool:
    return not Path(value).is_absolute() and not re.match(r"^[A-Za-z]:\\", value)


class RegistryTests(unittest.TestCase):
    def test_registry_exists_and_validates(self) -> None:
        registry = load_registry()
        self.assertEqual(registry["validation"]["status"], "passed")
        self.assertEqual(registry["validation"]["errors"], [])
        self.assertGreaterEqual(len(registry["experiments"]), 28)
        self.assertTrue(SUMMARY_PATH.exists())

    def test_registry_validation_is_read_only(self) -> None:
        registry_files = [REGISTRY_PATH, SUMMARY_PATH, EXPERIMENT_LOG_PATH, PAPER_MATRIX_PATH]
        before = {path: sha256_file(path) for path in registry_files}
        result = subprocess.run(
            [sys.executable, str(UPDATE_REGISTRY_PATH), "--validate"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = {path: sha256_file(path) for path in registry_files}
        self.assertEqual(before, after)

    def test_unique_experiment_ids(self) -> None:
        experiments = load_registry()["experiments"]
        ids = [exp["id"] for exp in experiments]
        self.assertEqual(len(ids), len(set(ids)))

    def test_metric_ranges(self) -> None:
        for exp in load_registry()["experiments"]:
            for key, value in walk_values(exp.get("metrics", {})):
                if key in METRIC_KEYS and isinstance(value, (int, float)):
                    self.assertGreaterEqual(value, 0.0, f"{exp['id']} {key}")
                    self.assertLessEqual(value, 1.0, f"{exp['id']} {key}")

    def test_project_relative_artifact_paths(self) -> None:
        for exp in load_registry()["experiments"]:
            self.assertTrue(is_relative_project_path(exp["run_directory"]), exp["id"])
            for name, value in exp.get("artifacts", {}).items():
                if value:
                    self.assertTrue(is_relative_project_path(value), f"{exp['id']} {name}: {value}")

    def test_superseded_phase2a_not_authoritative(self) -> None:
        registry = load_registry()
        authoritative = {exp["run_directory"].replace("\\", "/") for exp in registry["experiments"]}
        self.assertNotIn(SUPERSEDED_PHASE2A.replace("\\", "/"), authoritative)
        superseded = {row["run_directory"].replace("\\", "/") for row in registry["superseded_runs"]}
        self.assertIn(SUPERSEDED_PHASE2A.replace("\\", "/"), superseded)

    def test_external_zero_shot_marked_external_validation(self) -> None:
        exp = next(row for row in load_registry()["experiments"] if row["id"] == "phase2b_daic_to_android_zero_shot_wavlm")
        self.assertTrue(exp["integrity"]["zero_shot"])
        self.assertTrue(exp["integrity"]["external_validation"])
        self.assertFalse(exp["integrity"]["target_model_refitted"])
        self.assertFalse(exp["integrity"]["target_threshold_refitted"])

    def test_expected_embedding_dimensions_recorded(self) -> None:
        registry = load_registry()
        wavlm_entries = [exp for exp in registry["experiments"] if exp["representation"].get("feature_family") == "wavlm_embedding"]
        self.assertTrue(wavlm_entries)
        for exp in wavlm_entries:
            if exp["representation"].get("dimension") is not None:
                aggregation = exp["representation"].get("participant_aggregation")
                expected = 1536 if aggregation == "mean_std_pca" else 768
                self.assertEqual(exp["representation"]["dimension"], expected, exp["id"])


class RepositorySafetyTests(unittest.TestCase):
    def test_compact_records_do_not_contain_restricted_artifacts(self) -> None:
        records = ROOT / "records" / "runs"
        self.assertTrue(records.exists())
        for path in records.rglob("*"):
            if not path.is_file():
                continue
            self.assertNotIn(path.suffix.lower(), FORBIDDEN_SUFFIXES, str(path))
            self.assertFalse(path.name.lower().endswith("predictions.csv"), str(path))

    def test_configs_parse_as_json(self) -> None:
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))

    def test_no_absolute_windows_paths_in_source_or_config(self) -> None:
        scan_roots = [ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "tests", ROOT / ".github"]
        pattern = re.compile(r"[A-Za-z]:\\")
        for base in scan_roots:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".py", ".json", ".yml", ".yaml", ".md", ".txt"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    self.assertIsNone(pattern.search(text), str(path))

    def test_no_participant_prediction_files_in_trackable_records(self) -> None:
        for path in (ROOT / "records" / "runs").rglob("*predictions*.csv"):
            self.fail(f"Participant prediction file should not be in compact records: {path}")


if __name__ == "__main__":
    unittest.main()
