from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
VALIDATOR = ROOT / "paper" / "validate_manuscript.py"
MANUSCRIPT = PAPER / "MANUSCRIPT_DRAFT.md"

STAGE2C_FILES = [
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2C.md",
    PAPER / "CROSS_CORPUS_METHODS_LITERATURE_TABLE.md",
    PAPER / "DOMAIN_SHIFT_CONCEPT_TABLE.md",
    PAPER / "CALIBRATION_METHODS_TABLE.md",
]

STAGE2D_FILES = [
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2D.md",
    PAPER / "RESPONSIBLE_AI_EVIDENCE_TABLE.md",
    PAPER / "CLINICAL_READINESS_MATRIX.md",
]


class ManuscriptValidationTests(unittest.TestCase):
    def test_manuscript_validator_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(VALIDATOR)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("manuscript validation passed", result.stdout)

    def test_stage2c_files_exist(self) -> None:
        for path in STAGE2C_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"Missing {path}")

    def test_stage2d_files_exist(self) -> None:
        for path in STAGE2D_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"Missing {path}")

    def test_stage2c_tables_contain_expected_methods(self) -> None:
        cross_corpus = (PAPER / "CROSS_CORPUS_METHODS_LITERATURE_TABLE.md").read_text(encoding="utf-8")
        self.assertIn("Alghowinem", cross_corpus)
        self.assertIn("Ravi", cross_corpus)
        self.assertIn("Seneviratne", cross_corpus)

        domain_shift = (PAPER / "DOMAIN_SHIFT_CONCEPT_TABLE.md").read_text(encoding="utf-8")
        for concept in [
            "Covariate shift",
            "Prior probability shift",
            "Concept or conditional shift",
            "Label-definition shift",
            "Task shift",
            "Language or cultural shift",
            "Score calibration shift",
            "Threshold shift",
            "Classifier-boundary shift",
        ]:
            with self.subTest(concept=concept):
                self.assertIn(concept, domain_shift)

        calibration = (PAPER / "CALIBRATION_METHODS_TABLE.md").read_text(encoding="utf-8")
        for method in [
            "No calibration",
            "Intercept-only recalibration",
            "Logistic calibration / Platt scaling",
            "Isotonic regression",
            "Temperature scaling",
            "Threshold-only adaptation",
        ]:
            with self.subTest(method=method):
                self.assertIn(method, calibration)

    def test_stage2c_manuscript_preserves_transfer_claim_safeguards(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        for phrase in [
            "ROC-AUC 0.6412",
            "ROC-AUC 0.8675",
            "0.5558",
            "predicted every Android interview participant as negative",
            "target-domain threshold adaptation, not pure zero-shot evaluation",
            "target-domain score calibration rather than zero-shot evaluation",
            "not DAIC-to-Android model transfer",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_stage2d_manuscript_preserves_clinical_safeguards(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("[REFERENCE NEEDED:", text)
        for phrase in [
            "research-only",
            "not diagnostic tools",
            "No prospective deployment study was performed",
            "No demographic fairness audit was completed",
            "No clinical utility study was completed",
            "source datasets are not redistributed",
            "raw audio",
            "participant-level artifacts",
            "paper-experiment-freeze-v1",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_stage2d_responsible_ai_table_contains_required_issues(self) -> None:
        text = (PAPER / "RESPONSIBLE_AI_EVIDENCE_TABLE.md").read_text(encoding="utf-8")
        for issue in [
            "Mental-health data sensitivity",
            "Speech privacy",
            "Voice re-identification",
            "Linguistic bias",
            "Demographic bias",
            "Dataset imbalance",
            "Subgroup evaluation",
            "External validation",
            "Prospective clinical validation",
            "Calibration in target populations",
            "Screening versus diagnosis",
            "Human oversight",
            "Deployment monitoring",
            "Informed consent",
        ]:
            with self.subTest(issue=issue):
                self.assertIn(issue, text)

    def test_stage2d_clinical_readiness_matrix_preserves_statuses(self) -> None:
        text = (PAPER / "CLINICAL_READINESS_MATRIX.md").read_text(encoding="utf-8")
        self.assertIn("Project is not clinically deployment-ready.", text)
        expected = {
            "Retrospective technical validation": "completed",
            "Participant-level evaluation": "completed",
            "Cross-corpus zero-shot evaluation": "completed",
            "Target-domain calibration": "partially_completed",
            "Independent external validation of Android-trained models": "not_completed",
            "Subgroup fairness analysis": "not_completed",
            "Prospective clinical validation": "not_completed",
            "Comparison with clinician assessment": "not_completed",
            "Clinical utility evaluation": "not_completed",
            "Privacy and governance assessment": "partially_completed",
            "Deployment monitoring and drift management": "not_completed",
        }
        for criterion, status in expected.items():
            with self.subTest(criterion=criterion):
                self.assertIn(f"| {criterion} | {status} |", text)


if __name__ == "__main__":
    unittest.main()
