import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phase2"))

from phase2e_protocols import (
    EXPECTED_DIM,
    assert_coherent_fold_integrity,
    assert_no_participant_overlap,
    assert_oof_prediction_completeness,
    assert_paired_identity,
    make_participant_coherent_folds,
)


class Phase2ECrossTaskSafeguardTests(unittest.TestCase):
    def test_participant_coherent_folds_are_deterministic_and_balanced(self) -> None:
        labels = np.array([0] * 10 + [1] * 10)
        folds_a = make_participant_coherent_folds(labels, seed=20260704)
        folds_b = make_participant_coherent_folds(labels, seed=20260704)
        self.assertEqual(folds_a, folds_b)
        summaries = assert_coherent_fold_integrity([f"P{i:02d}" for i in range(20)], labels, folds_a)
        self.assertEqual(len(summaries), 5)
        self.assertTrue(all(row.hc_count > 0 and row.pt_count > 0 for row in summaries))
        self.assertEqual(sum(row.outer_test_count for row in summaries), 20)

    def test_paired_identity_rejects_duplicate_participant(self) -> None:
        labels = np.array([0, 1])
        embeddings = np.ones((2, EXPECTED_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            assert_paired_identity(["HC_01", "HC_01"], labels, embeddings, embeddings)

    def test_paired_identity_rejects_bad_embedding_dimension(self) -> None:
        labels = np.array([0, 1])
        reading = np.ones((2, EXPECTED_DIM - 1), dtype=np.float32)
        interview = np.ones((2, EXPECTED_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            assert_paired_identity(["HC_01", "PT_01"], labels, reading, interview)

    def test_paired_identity_rejects_all_zero_vector(self) -> None:
        labels = np.array([0, 1])
        reading = np.ones((2, EXPECTED_DIM), dtype=np.float32)
        interview = np.ones((2, EXPECTED_DIM), dtype=np.float32)
        reading[1, :] = 0.0
        with self.assertRaises(ValueError):
            assert_paired_identity(["HC_01", "PT_01"], labels, reading, interview)

    def test_outer_leakage_guard_rejects_overlap(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_participant_overlap({"HC_01", "PT_01"}, {"PT_01"}, "synthetic")

    def test_oof_completeness_rejects_duplicate_prediction(self) -> None:
        rows = [
            {"protocol": "reading_to_interview", "threshold_variant": "threshold_0_5", "participant_id": "HC_01"},
            {"protocol": "reading_to_interview", "threshold_variant": "threshold_0_5", "participant_id": "PT_01"},
            {"protocol": "reading_to_interview", "threshold_variant": "threshold_0_5", "participant_id": "PT_01"},
        ]
        with self.assertRaises(ValueError):
            assert_oof_prediction_completeness(rows, {"HC_01", "PT_01"}, "reading_to_interview", "threshold_0_5")

    def test_oof_completeness_accepts_one_prediction_per_participant(self) -> None:
        rows = [
            {"protocol": "reading_to_interview", "threshold_variant": "threshold_0_5", "participant_id": "HC_01"},
            {"protocol": "reading_to_interview", "threshold_variant": "threshold_0_5", "participant_id": "PT_01"},
        ]
        assert_oof_prediction_completeness(rows, {"HC_01", "PT_01"}, "reading_to_interview", "threshold_0_5")


if __name__ == "__main__":
    unittest.main()
