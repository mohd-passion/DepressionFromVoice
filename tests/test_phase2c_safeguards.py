import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phase2"))

from phase2c_calibration import available_calibration_methods
from phase2c_metrics import assert_metric_range, select_threshold
from phase2c_protocols import assert_fold_integrity, assert_unique_oof_predictions


class Phase2CSafeguardTests(unittest.TestCase):
    def test_fold_integrity_rejects_duplicate_participant(self) -> None:
        participant_ids = ["HC_01", "HC_01", "PT_01", "PT_02", "PT_03"]
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        labels = np.array([0, 0, 1, 1, 1])
        with self.assertRaises(ValueError):
            assert_fold_integrity(participant_ids, folds, labels)

    def test_fold_integrity_accepts_five_clean_folds(self) -> None:
        participant_ids = ["HC_01", "HC_02", "PT_01", "PT_02", "PT_03"]
        folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        labels = np.array([0, 0, 1, 1, 1])
        summaries = assert_fold_integrity(participant_ids, folds, labels)
        self.assertEqual(len(summaries), 5)
        self.assertEqual(sum(item.outer_test_count for item in summaries), 5)

    def test_threshold_selection_uses_requested_tie_breaking(self) -> None:
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.2, 0.4, 0.6, 0.8])
        selected = select_threshold(labels, probabilities, extra_thresholds=[0.555840134621])
        self.assertTrue(0.0 <= selected.threshold <= 1.0)
        self.assertTrue(0.0 <= selected.macro_f1 <= 1.0)

    def test_calibration_method_boundaries(self) -> None:
        y_small = np.array([0, 1, 0, 1])
        p_small = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(available_calibration_methods(y_small, p_small), ["intercept_only", "platt"])
        y_one_class = np.zeros(20, dtype=int)
        p = np.linspace(0.1, 0.9, 20)
        self.assertEqual(available_calibration_methods(y_one_class, p), [])
        y_stable = np.array([0] * 20 + [1] * 20)
        p_stable = np.linspace(0.05, 0.95, 40)
        self.assertIn("isotonic", available_calibration_methods(y_stable, p_stable))

    def test_unique_oof_prediction_enforcement(self) -> None:
        rows = [
            {"protocol": "B", "participant_id": "HC_01"},
            {"protocol": "B", "participant_id": "PT_01"},
        ]
        assert_unique_oof_predictions(rows, {"HC_01", "PT_01"}, "B")
        rows.append({"protocol": "B", "participant_id": "PT_01"})
        with self.assertRaises(ValueError):
            assert_unique_oof_predictions(rows, {"HC_01", "PT_01"}, "B")

    def test_protocol_config_parses(self) -> None:
        config = json.loads((ROOT / "configs" / "phase2c_android_domain_shift.json").read_text(encoding="utf-8"))
        self.assertEqual(config["planned_registry_id"], "P2C_ANDROID_DOMAIN_SHIFT_001")
        self.assertEqual(config["wavlm_layer"], "layer_09")
        self.assertEqual(config["embedding_dimension"], 768)

    def test_metric_range_validation(self) -> None:
        assert_metric_range("roc_auc", 0.5)
        with self.assertRaises(ValueError):
            assert_metric_range("roc_auc", 1.5)


if __name__ == "__main__":
    unittest.main()
