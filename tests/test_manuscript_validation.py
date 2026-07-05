from __future__ import annotations

import csv
import hashlib
import re
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

STAGE3_FILES = [
    PAPER / "FINAL_NUMERICAL_AUDIT.csv",
    PAPER / "FINAL_RESULT_SCOPE_AUDIT.md",
    PAPER / "FINAL_CITATION_AUDIT.csv",
    PAPER / "REFERENCE_CLEANUP_REPORT.md",
    PAPER / "TERMINOLOGY_STYLE_GUIDE.md",
    PAPER / "MANUSCRIPT_STRUCTURE_AUDIT.md",
    PAPER / "REDUNDANCY_EDIT_LOG.md",
    PAPER / "CONTRIBUTION_CLAIM_AUDIT.md",
    PAPER / "FINAL_TABLE_AUDIT.md",
    PAPER / "FIGURE_SPECIFICATIONS.md",
    PAPER / "SUPPLEMENTARY_MATERIAL_PLAN.md",
    PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md",
    PAPER / "DATA_CODE_AVAILABILITY_AUDIT.md",
    PAPER / "FINAL_CLINICAL_CLAIM_AUDIT.md",
    PAPER / "SUBMISSION_READINESS_CHECKLIST.md",
]

REGISTRY = ROOT / "outputs" / "experiment_registry" / "experiment_registry.json"
REGISTRY_VALIDATOR = ROOT / "outputs" / "experiment_registry" / "update_registry.py"


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

    def test_stage3_files_exist(self) -> None:
        for path in STAGE3_FILES:
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

    def test_stage3_numerical_audit_covers_major_metrics(self) -> None:
        with (PAPER / "FINAL_NUMERICAL_AUDIT.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 80)
        self.assertTrue(all(row["correction required"].lower() == "no" for row in rows))
        values = {row["reported value"] for row in rows}
        for value in [
            "0.6412",
            "0.8675",
            "0.8796",
            "0.8475",
            "0.8501",
            "0.9058",
            "0.9135",
            "550 of 768",
            "292 of 768",
            "0.4752",
            "0.6659",
            "0.1216",
        ]:
            with self.subTest(value=value):
                self.assertIn(value, values)

    def test_stage3_result_scope_audit_contains_required_protocols(self) -> None:
        text = (PAPER / "FINAL_RESULT_SCOPE_AUDIT.md").read_text(encoding="utf-8").lower()
        for phrase in [
            "daic methodological development",
            "provisional local labeled daic test",
            "cross-corpus zero-shot external evaluation",
            "target-domain threshold adaptation",
            "target-domain score calibration",
            "within-android interview nested cross-validation",
            "within-android reading nested cross-validation",
            "cross-task transfer within android",
            "pooled-task within-android evaluation",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_stage3_citation_audit_covers_all_manuscript_keys(self) -> None:
        manuscript = MANUSCRIPT.read_text(encoding="utf-8")
        cited_keys = set(re.findall(r"@([A-Za-z][A-Za-z0-9_:-]*)", manuscript))
        with (PAPER / "FINAL_CITATION_AUDIT.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        audited_keys = {row["citation key"] for row in rows}
        self.assertLessEqual(cited_keys, audited_keys)
        self.assertTrue(all(row["BibTeX entry exists"] == "yes" for row in rows))
        self.assertTrue(all(row["citation appropriate"] == "yes" for row in rows))

    def test_stage3_figure_specifications_are_complete(self) -> None:
        text = (PAPER / "FIGURE_SPECIFICATIONS.md").read_text(encoding="utf-8")
        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5", "Figure S1", "Figure S2"]:
            with self.subTest(label=label):
                self.assertIn(label, text)
        for field in [
            "Exact source files",
            "Required columns",
            "Data transformations",
            "Prohibited interpretation",
            "Preferred vector output format",
        ]:
            with self.subTest(field=field):
                self.assertIn(field, text)

    def test_stage3_submission_checklist_uses_allowed_statuses(self) -> None:
        text = (PAPER / "SUBMISSION_READINESS_CHECKLIST.md").read_text(encoding="utf-8")
        self.assertIn("`ready_for_author_review`", text)
        allowed = {
            "complete",
            "needs_author_input",
            "needs_formatting",
            "needs_figure",
            "needs_journal_selection",
            "not_applicable",
        }
        for line in text.splitlines():
            separator = line.replace("|", "").replace(" ", "").strip()
            if not line.startswith("| ") or line.startswith("| Item ") or set(separator) <= {"-", ":"}:
                continue
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            self.assertIn(cells[1], allowed, line)

    def test_stage3_terminology_and_clinical_restrictions(self) -> None:
        terminology = (PAPER / "TERMINOLOGY_STYLE_GUIDE.md").read_text(encoding="utf-8")
        for phrase in [
            "within-corpus nested cross-validation",
            "research classification model",
            "diagnostic system",
            "external accuracy",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, terminology)

        clinical = (PAPER / "FINAL_CLINICAL_CLAIM_AUDIT.md").read_text(encoding="utf-8")
        for phrase in [
            "| Diagnosis | no |",
            "| Clinical validation | no |",
            "| Deployment readiness | no |",
            "| Fairness across demographic groups | no |",
            "| Universal generalization | no |",
            "| Prospective utility | no |",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, clinical)

    def test_stage3_experiment_metrics_remain_unchanged(self) -> None:
        evidence = (PAPER / "PAPER_EVIDENCE_MAP.md").read_text(encoding="utf-8")
        for phrase in [
            "0.6412 | phase2b_daic_to_android_zero_shot_wavlm",
            "0.8675 | phase2c_protocol_d_android_lr",
            "0.8796 | phase2d_android_reading_logistic_regression",
            "0.8475 | phase2e_reading_to_interview_transfer",
            "0.8501 | phase2e_interview_to_reading_transfer",
            "0.9058 | phase2e_pooled_to_reading",
            "0.9135 | phase2e_pooled_to_interview",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, evidence)

    def test_registry_validation_is_read_only(self) -> None:
        before = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
        result = subprocess.run(
            [sys.executable, str(REGISTRY_VALIDATOR), "--validate"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        after = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, after)
        self.assertIn("experiments=42", result.stdout)


if __name__ == "__main__":
    unittest.main()
