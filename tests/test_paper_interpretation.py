from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "outputs" / "experiment_registry" / "experiment_registry.json"
PAPER_MATRIX_PATH = ROOT / "docs" / "PAPER_RESULTS_INTERPRETATION_MATRIX.md"

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

NON_EXTERNAL_SCOPES = {
    "methodological_ablation",
    "within_corpus_cross_validation",
    "within_corpus_held_out_test",
    "provisional_local_labeled_test",
    "cross_task_transfer",
    "pooled_task_within_corpus_cross_validation",
    "data_quality_audit",
}

BOOLEAN_FIELDS = {
    "target_labels_used_for_training",
    "target_labels_used_for_model_selection",
    "target_labels_used_for_threshold_selection",
    "external_validation",
    "cross_corpus",
    "zero_shot",
    "target_domain_refitting",
    "calibration_performed",
    "threshold_adaptation_performed",
}


def load_experiments() -> list[dict]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["experiments"]


class PaperInterpretationTests(unittest.TestCase):
    def test_all_experiments_have_paper_interpretation_fields(self) -> None:
        for exp in load_experiments():
            with self.subTest(exp=exp["id"]):
                paper = exp.get("paper_interpretation")
                self.assertIsInstance(paper, dict)
                self.assertIn(paper["result_scope"], RESULT_SCOPES)
                self.assertTrue(paper["evaluation_protocol"])
                self.assertTrue(paper["paper_ready_result_sentence"])
                self.assertTrue(paper["allowed_claims"])
                self.assertTrue(paper["prohibited_claims"])
                self.assertTrue(paper["clinical_status"])
                for field in BOOLEAN_FIELDS:
                    self.assertIsInstance(paper[field], bool, field)

    def test_external_validation_flags_match_result_scope(self) -> None:
        for exp in load_experiments():
            paper = exp["paper_interpretation"]
            with self.subTest(exp=exp["id"]):
                if paper["result_scope"] in NON_EXTERNAL_SCOPES:
                    self.assertFalse(paper["external_validation"])
                if paper["result_scope"] == "cross_corpus_zero_shot_external_evaluation":
                    self.assertTrue(paper["external_validation"])
                    self.assertTrue(paper["cross_corpus"])
                    self.assertTrue(paper["zero_shot"])

    def test_zero_shot_is_not_mixed_with_target_adaptation(self) -> None:
        for exp in load_experiments():
            paper = exp["paper_interpretation"]
            with self.subTest(exp=exp["id"]):
                if paper["zero_shot"]:
                    self.assertFalse(paper["target_domain_refitting"])
                    self.assertFalse(paper["threshold_adaptation_performed"])

    def test_provisional_daic_test_is_not_called_official_avec(self) -> None:
        for exp in load_experiments():
            paper = exp["paper_interpretation"]
            if paper["result_scope"] != "provisional_local_labeled_test":
                continue
            text = " ".join(
                [
                    paper["paper_ready_result_sentence"],
                    paper["evaluation_protocol"],
                    " ".join(paper["allowed_claims"]),
                ]
            ).lower()
            self.assertNotIn("official avec", text, exp["id"])
            self.assertNotIn("official_avec", text, exp["id"])

    def test_cross_task_transfer_is_not_called_cross_corpus(self) -> None:
        for exp in load_experiments():
            paper = exp["paper_interpretation"]
            if paper["result_scope"] != "cross_task_transfer":
                continue
            text = " ".join(
                [
                    paper["paper_ready_result_sentence"],
                    paper["evaluation_protocol"],
                    " ".join(paper["allowed_claims"]),
                ]
            ).lower()
            self.assertNotIn("cross-corpus", text, exp["id"])
            self.assertNotIn("cross-dataset", text, exp["id"])

    def test_research_results_do_not_make_diagnostic_claims(self) -> None:
        for exp in load_experiments():
            paper = exp["paper_interpretation"]
            if paper["clinical_status"] == "clinically_validated":
                continue
            text = " ".join([paper["paper_ready_result_sentence"], " ".join(paper["allowed_claims"])]).lower()
            self.assertNotIn("diagnostic", text, exp["id"])
            self.assertNotIn("diagnosis", text, exp["id"])
            self.assertNotIn("clinical deployment", text, exp["id"])

    def test_paper_matrix_lists_all_experiment_ids(self) -> None:
        matrix_text = PAPER_MATRIX_PATH.read_text(encoding="utf-8")
        self.assertTrue(matrix_text.startswith("# Paper Results Interpretation Matrix"))
        for exp in load_experiments():
            self.assertIn(exp["id"], matrix_text)


if __name__ == "__main__":
    unittest.main()
