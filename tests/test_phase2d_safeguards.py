import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phase2"))

from phase2c_metrics import assert_metric_range, select_threshold
from phase2d_protocols import assert_reading_fold_integrity, assert_unique_oof_predictions
from run_phase2d_android_reading import HOP_SAMPLES, TARGET_SAMPLES


class Phase2DReadingSafeguardTests(unittest.TestCase):
    def test_reading_fold_integrity_rejects_duplicate_participant(self) -> None:
        participant_ids = ["HC_01", "HC_01", "PT_01", "PT_02", "PT_03"]
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        labels = np.array([0, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            assert_reading_fold_integrity(participant_ids, folds, labels, ["reading"] * 5)

    def test_reading_fold_integrity_rejects_interview_task(self) -> None:
        participant_ids = ["HC_01", "HC_02", "PT_01", "PT_02", "PT_03", "HC_03", "PT_04", "HC_04", "PT_05", "HC_05"]
        folds = ["fold1", "fold1", "fold2", "fold2", "fold3", "fold3", "fold4", "fold4", "fold5", "fold5"]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        tasks = ["reading"] * 9 + ["interview"]
        with self.assertRaises(ValueError):
            assert_reading_fold_integrity(participant_ids, folds, labels, tasks)

    def test_reading_fold_integrity_accepts_clean_five_folds(self) -> None:
        participant_ids = ["HC_01", "PT_01", "HC_02", "PT_02", "HC_03", "PT_03", "HC_04", "PT_04", "HC_05", "PT_05"]
        folds = ["fold1", "fold1", "fold2", "fold2", "fold3", "fold3", "fold4", "fold4", "fold5", "fold5"]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        summaries = assert_reading_fold_integrity(participant_ids, folds, labels, ["reading"] * 10)
        self.assertEqual(len(summaries), 5)
        self.assertEqual(sum(item.outer_test_count for item in summaries), 10)

    def test_unique_oof_prediction_enforcement(self) -> None:
        rows = [
            {"protocol": "B_android_lr_at_0_5", "participant_id": "HC_01"},
            {"protocol": "B_android_lr_at_0_5", "participant_id": "PT_01"},
        ]
        assert_unique_oof_predictions(rows, {"HC_01", "PT_01"}, "B_android_lr_at_0_5")
        rows.append({"protocol": "B_android_lr_at_0_5", "participant_id": "PT_01"})
        with self.assertRaises(ValueError):
            assert_unique_oof_predictions(rows, {"HC_01", "PT_01"}, "B_android_lr_at_0_5")

    def test_threshold_selection_tie_breaking_runs(self) -> None:
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.2, 0.4, 0.6, 0.8])
        selected = select_threshold(labels, probabilities)
        self.assertTrue(0.0 <= selected.threshold <= 1.0)
        self.assertTrue(0.0 <= selected.macro_f1 <= 1.0)

    def test_protocol_config_parses_and_excludes_interview_fusion(self) -> None:
        config = json.loads((ROOT / "configs" / "phase2d_android_reading.json").read_text(encoding="utf-8"))
        self.assertEqual(config["wavlm_layer"], "layer_09")
        self.assertEqual(config["embedding_dimension"], 768)
        self.assertEqual(config["target_samples"], 160000)
        self.assertTrue(config["forbidden"]["use_interview_audio"])
        self.assertTrue(config["forbidden"]["combine_reading_and_interview_features"])

    def test_window_constants_match_locked_preprocessing(self) -> None:
        self.assertEqual(TARGET_SAMPLES, 160000)
        self.assertEqual(HOP_SAMPLES, 80000)

    def test_metric_range_validation(self) -> None:
        assert_metric_range("roc_auc", 0.75)
        with self.assertRaises(ValueError):
            assert_metric_range("roc_auc", -0.01)


if __name__ == "__main__":
    unittest.main()
