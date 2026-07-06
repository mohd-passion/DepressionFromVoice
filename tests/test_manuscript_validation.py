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

STAGE3B_FILES = [
    PAPER / "AUTHOR_INPUT_FORM.md",
    PAPER / "AUTHOR_DECISION_LOG.md",
    PAPER / "MANUSCRIPT_AUTHOR_REVIEW_CHECKLIST.md",
    PAPER / "JOURNAL_CANDIDATE_MATRIX.md",
    PAPER / "JOURNAL_SELECTION_CRITERIA.md",
    PAPER / "JOURNAL_SHORTLIST.md",
    PAPER / "SUBMISSION_STRATEGY.md",
    PAPER / "JOURNAL_FIT_GAP_ANALYSIS.md",
    PAPER / "MANUSCRIPT_AUTHOR_REVIEW_COPY.md",
    PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md",
]

STAGE3C_FILES = [
    PAPER / "AUTHOR_PREFERENCES_PRELIMINARY.md",
    PAPER / "PROVISIONAL_TARGET_JOURNAL_DECISION.md",
    PAPER / "JOURNAL_ADAPTATION_PLAN.md",
    PAPER / "AUTHOR_FINAL_QUESTIONS.md",
]

STAGE3D_FILES = [
    PAPER / "BSPC_GUIDELINES_VERIFICATION.md",
    PAPER / "BSPC_ADAPTATION_PLAN.md",
    PAPER / "BSPC_MANUSCRIPT_DRAFT.md",
    PAPER / "BSPC_HIGHLIGHTS.md",
    PAPER / "BSPC_GRAPHICAL_ABSTRACT_PLAN.md",
    PAPER / "BSPC_DECLARATIONS_DRAFT.md",
    PAPER / "BSPC_FIGURE_PRODUCTION_PLAN.md",
    PAPER / "BSPC_TABLE_FORMATTING_PLAN.md",
    PAPER / "BSPC_REFERENCE_FORMATTING_PLAN.md",
    PAPER / "BSPC_COVER_LETTER_DRAFT.md",
]

STAGE3E_FIGURE_OUTPUTS = [
    PAPER / "figures" / "output" / "figure1_study_design_protocol_overview.svg",
    PAPER / "figures" / "output" / "figure1_study_design_protocol_overview.png",
    PAPER / "figures" / "output" / "figure2_daic_phase1_key_results.svg",
    PAPER / "figures" / "output" / "figure2_daic_phase1_key_results.png",
    PAPER / "figures" / "output" / "figure3_daic_android_transfer_adaptation.svg",
    PAPER / "figures" / "output" / "figure3_daic_android_transfer_adaptation.png",
    PAPER / "figures" / "output" / "figure4_android_cross_task_results.svg",
    PAPER / "figures" / "output" / "figure4_android_cross_task_results.png",
    PAPER / "figures" / "output" / "figure5_domain_shift_boundary_summary.svg",
    PAPER / "figures" / "output" / "figure5_domain_shift_boundary_summary.png",
]

STAGE3E_FILES = [
    PAPER / "BSPC_FIGURE_CAPTIONS.md",
    PAPER / "BSPC_FIGURE_DATA_AUDIT.csv",
    PAPER / "figures" / "README.md",
    PAPER / "figures" / "scripts" / "generate_bspc_figures.py",
    PAPER / "figures" / "source" / "bspc_figure_source_values.csv",
    PAPER / "figures" / "output" / "figure_generation_log.txt",
    *STAGE3E_FIGURE_OUTPUTS,
]

REGISTRY = ROOT / "outputs" / "experiment_registry" / "experiment_registry.json"
REGISTRY_VALIDATOR = ROOT / "outputs" / "experiment_registry" / "update_registry.py"
MANUSCRIPT_SHA256 = "e0a89872e7244bf4aac47a055c055c590bd5394dea63f0516bf1e804d170d97c"


def first_markdown_table(path: Path) -> tuple[list[str], list[list[str]]]:
    header: list[str] = []
    rows: list[list[str]] = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table and rows:
                break
            continue
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if not header:
            header = cells
            in_table = True
            continue
        rows.append(cells)
    return header, rows


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

    def test_stage3b_files_exist(self) -> None:
        for path in STAGE3B_FILES:
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
            "figures_generated_awaiting_author_review",
            "not_applicable",
        }
        for line in text.splitlines():
            separator = line.replace("|", "").replace(" ", "").strip()
            if not line.startswith("| ") or line.startswith("| Item ") or set(separator) <= {"-", ":"}:
                continue
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            self.assertIn(cells[1], allowed, line)

    def test_stage3b_author_input_form_contains_required_fields(self) -> None:
        text = (PAPER / "AUTHOR_INPUT_FORM.md").read_text(encoding="utf-8")
        for field in [
            "Final working title",
            "Running title",
            "Manuscript article type",
            "Target journal, once selected",
            "Full name",
            "Preferred publication name",
            "Author order",
            "Highest qualification",
            "Department",
            "Institution",
            "Institutional email",
            "ORCID",
            "Corresponding-author status",
            "Funding",
            "Competing interests",
            "Ethics approval or exemption",
            "Consent to participate",
            "Consent for publication",
            "Data availability",
            "Code availability",
            "Use of AI-assisted writing tools",
            "Prior presentation or preprint status",
        ]:
            with self.subTest(field=field):
                self.assertIn(field, text)
        self.assertIn("Unverified Author Context", text)
        self.assertIn("Mohammad Ilyas", text)
        self.assertIn(
            "Final affiliation wording | Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India | confirmed for Stage 3E draft |",
            text,
        )
        for role in [
            "Conceptualization",
            "Methodology",
            "Software",
            "Validation",
            "Formal analysis",
            "Investigation",
            "Resources",
            "Data curation",
            "Writing - original draft",
            "Writing - review and editing",
            "Visualization",
            "Supervision",
            "Project administration",
            "Funding acquisition",
        ]:
            with self.subTest(role=role):
                self.assertIn(role, text)

    def test_stage3b_decision_log_covers_required_topics(self) -> None:
        text = (PAPER / "AUTHOR_DECISION_LOG.md").read_text(encoding="utf-8")
        for topic in [
            "Final title",
            "Author order",
            "Corresponding author",
            "Journal choice",
            "Article type",
            "Open-access preference",
            "Maximum APC",
            "Figure count",
            "Supplement scope",
            "Ethics wording",
            "AI-assistance disclosure",
            "Preprint preference",
        ]:
            with self.subTest(topic=topic):
                self.assertIn(topic, text)
        self.assertIn("Do not select final journal until author preferences are supplied", text)

    def test_stage3b_journal_matrix_structure_and_sources(self) -> None:
        header, rows = first_markdown_table(PAPER / "JOURNAL_CANDIDATE_MATRIX.md")
        expected_columns = {
            "journal",
            "publisher",
            "scope fit",
            "speech/affective-computing fit",
            "mental-health fit",
            "methodological fit",
            "article type",
            "word limit",
            "figure/table constraints",
            "open-access model",
            "APC",
            "preprint policy",
            "data/code policy",
            "likely manuscript changes",
            "main advantage",
            "main risk",
            "official source",
            "verification date",
        }
        self.assertLessEqual(expected_columns, set(header))
        self.assertGreaterEqual(len(rows), 6)
        self.assertLessEqual(len(rows), 10)
        text = (PAPER / "JOURNAL_CANDIDATE_MATRIX.md").read_text(encoding="utf-8")
        for journal in [
            "Biomedical Signal Processing and Control",
            "PLOS Digital Health",
            "Frontiers in Digital Health",
            "Computer Speech & Language",
            "Artificial Intelligence in Medicine",
            "Speech Communication",
        ]:
            with self.subTest(journal=journal):
                self.assertIn(journal, text)
        for row in rows:
            row_data = dict(zip(header, row))
            self.assertEqual(row_data["verification date"], "2026-07-06")
            self.assertIn("https://", row_data["official source"])
        self.assertIn("not_verified", text)

    def test_stage3b_verification_log_is_complete(self) -> None:
        header, rows = first_markdown_table(PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md")
        expected_columns = {
            "journal",
            "publisher",
            "information checked",
            "official source",
            "date checked",
            "status",
            "ambiguity",
            "action needed before submission",
        }
        self.assertLessEqual(expected_columns, set(header))
        self.assertGreaterEqual(len(rows), 6)
        statuses = {"verified", "partially_verified", "not_verified"}
        for row in rows:
            row_data = dict(zip(header, row))
            self.assertEqual(row_data["date checked"], "2026-07-06")
            self.assertIn("https://", row_data["official source"])
            self.assertIn(row_data["status"], statuses)
        text = (PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md").read_text(encoding="utf-8")
        self.assertIn("not_verified", text)

    def test_stage3b_avoids_fabricated_journal_metrics(self) -> None:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in STAGE3B_FILES)
        self.assertNotRegex(
            combined,
            r"\b(?:impact factor|citescore|acceptance rate|acceptance-rate|quartile)\s*[:=]\s*\d",
        )
        self.assertIn("No candidate should be described as guaranteed acceptance", combined)

    def test_stage3c_required_files_exist(self) -> None:
        for path in STAGE3C_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), path)

    def test_stage3c_author_preferences_and_stage3e_author_inputs_are_recorded(self) -> None:
        preferences = (PAPER / "AUTHOR_PREFERENCES_PRELIMINARY.md").read_text(encoding="utf-8")
        for phrase in [
            "SCI/SCIE mandatory | No",
            "Q1 mandatory | No",
            "Scopus acceptable if SCI/SCIE unavailable | Yes",
            "Maximum APC | not specified",
            "Must have no-APC subscription route | Yes",
            "Open access mandatory | No",
            "highest quartile where possible",
            "ORCID | not available/not provided",
            "Additional authors | no additional authors provided at this stage",
            "Treat `N/A` as not yet provided",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, preferences)

        author_input = (PAPER / "AUTHOR_INPUT_FORM.md").read_text(encoding="utf-8")
        for phrase in [
            "Funding | No funding was received for this study.",
            "Competing interests | The authors declare no competing interests.",
            "Ethics approval or exemption | This study used previously collected datasets obtained under their original access terms.",
            "Consent to participate | [AUTHOR CONFIRMATION REQUIRED: consent statement for use of DAIC-WOZ and Androids Corpus data]",
            "Consent for publication | [AUTHOR CONFIRMATION REQUIRED: consent statement for use of DAIC-WOZ and Androids Corpus data]",
            "Acknowledgements | [AUTHOR CONFIRMATION REQUIRED: acknowledgements]",
            "Use of AI-assisted writing tools | [AUTHOR CONFIRMATION REQUIRED: AI-assisted writing/tool-use disclosure. ChatGPT/Codex was used",
            "Prior presentation or preprint status | Preprint decision: undecided.",
            "Institutional email | mohd.passion.cse@mitmoradabad.edu.in",
            "Institutional email | [AUTHOR CONFIRMATION REQUIRED: Dr. Vikram Singh email not provided]",
            "ORCID | not available",
            "ORCID | [AUTHOR CONFIRMATION REQUIRED: Dr. Vikram Singh ORCID not provided]",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, author_input)
        self.assertNotIn("Use of AI-assisted writing tools | No", author_input)
        self.assertNotIn("No external funding", author_input)
        self.assertNotRegex(author_input, r"orcid\.org/|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")

    def test_stage3c_target_is_provisional_not_final(self) -> None:
        decision = (PAPER / "PROVISIONAL_TARGET_JOURNAL_DECISION.md").read_text(encoding="utf-8")
        for phrase in [
            "Decision status: `provisional_target_selected`",
            "Final submission status: `not_selected`",
            "Biomedical Signal Processing and Control",
            "This is not a final author decision",
            "needs_reverification_before_submission",
            "awaiting_author_confirmation_for_target_journal",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, decision)
        self.assertNotIn("Final submission status: `selected`", decision)

        log = (PAPER / "AUTHOR_DECISION_LOG.md").read_text(encoding="utf-8")
        self.assertIn("Biomedical Signal Processing and Control is provisional only", log)
        self.assertIn("[AUTHOR DECISION REQUIRED] before formatting/submission", log)

    def test_stage3c_no_unverified_journal_metric_or_apc_claims(self) -> None:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in STAGE3C_FILES)
        self.assertNotRegex(
            combined,
            r"\b(?:impact factor|citescore|acceptance rate|acceptance-rate|quartile)\s*[:=]\s*\d",
        )
        self.assertNotRegex(combined, r"\b(?:USD|CHF|EUR|INR)\s*[\d,]+")
        self.assertIn("No current impact factor", combined)

    def test_stage3c_author_final_questions_count(self) -> None:
        questions = re.findall(
            r"^\d+\. ",
            (PAPER / "AUTHOR_FINAL_QUESTIONS.md").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertEqual(len(questions), 17)

    def test_stage3d_required_files_exist(self) -> None:
        for path in STAGE3D_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), path)

    def test_stage3d_bspc_manuscript_preserves_metrics_and_limits(self) -> None:
        text = (PAPER / "BSPC_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
        self.assertIn("Submission status: journal-specific preparation only; not submitted.", text)
        self.assertIn("Article type: Full paper [AUTHOR CONFIRMATION REQUIRED]", text)
        self.assertNotIn("[REFERENCE NEEDED:", text)

        abstract = re.search(r"## Abstract\s+(.*?)\s+## Keywords", text, flags=re.DOTALL)
        self.assertIsNotNone(abstract)
        self.assertLessEqual(len(re.findall(r"\b\S+\b", abstract.group(1))), 250)

        keywords = re.search(r"## Keywords\s+(.*?)\s+## 1\. Introduction", text, flags=re.DOTALL)
        self.assertIsNotNone(keywords)
        self.assertLessEqual(len([item for item in keywords.group(1).split(";") if item.strip()]), 7)

        for phrase in [
            "ROC-AUC 0.6412",
            "ROC-AUC 0.8675",
            "ROC-AUC 0.8796",
            "0.8475",
            "0.8501",
            "0.9058",
            "0.9135",
            "550 shifted WavLM features",
            "292 of 768 features",
            "0.4752",
            "0.6659",
            "0.1216",
            "not clinical validation",
            "not diagnostic tools",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_stage3e_declarations_apply_confirmed_items_and_keep_unresolved_markers(self) -> None:
        text = (PAPER / "BSPC_DECLARATIONS_DRAFT.md").read_text(encoding="utf-8")
        for section in [
            "## Funding",
            "## Competing Interests",
            "## Ethics Approval",
            "## Consent To Participate",
            "## Consent For Publication",
            "## Data Availability",
            "## Code Availability",
            "## Author Contributions",
            "## Acknowledgements",
            "## Declaration Of Generative AI And AI-Assisted Technologies In The Writing Process",
            "## Preprint Statement",
        ]:
            with self.subTest(section=section):
                self.assertIn(section, text)
        for phrase in [
            "No funding was received for this study.",
            "The authors declare no competing interests.",
            "This study used previously collected datasets obtained under their original access terms.",
            "Author-review note: this wording must be checked against institutional policy and the dataset licence terms before submission.",
            "[AUTHOR CONFIRMATION REQUIRED: consent statement for use of DAIC-WOZ and Androids Corpus data]",
            "[AUTHOR CONFIRMATION REQUIRED: acknowledgements]",
            "[AUTHOR CONFIRMATION REQUIRED: AI-assisted writing/tool-use disclosure. ChatGPT/Codex was used",
            "[AUTHOR CONFIRMATION REQUIRED: supervision, methodology review, writing - review and editing, or other CRediT roles]",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        self.assertNotIn("No AI", text)

    def test_stage3e_figure_package_outputs_and_audit_are_present(self) -> None:
        for path in STAGE3E_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), path)

        captions = (PAPER / "BSPC_FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
        self.assertIn("figures_generated_awaiting_author_review", captions)
        self.assertEqual(captions.count("Prohibited interpretation"), 5)
        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
            self.assertIn(label, captions)

        with (PAPER / "BSPC_FIGURE_DATA_AUDIT.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 30)
        self.assertEqual({row["figure"] for row in rows}, {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"})
        self.assertLessEqual({row["match status"] for row in rows}, {"match", "not_applicable"})
        values = {(row["figure"], row["metric"], row["plotted value"]) for row in rows}
        for required in [
            ("Figure 2", "ROC-AUC", "0.5065"),
            ("Figure 3", "ROC-AUC", "0.8675"),
            ("Figure 4", "ROC-AUC", "0.9135"),
            ("Figure 5", "DAIC-to-Android MMD^2", "0.6091"),
            ("Figure 5", "Cross-target corr.", "0.5555"),
        ]:
            with self.subTest(required=required):
                self.assertIn(required, values)

        for path in STAGE3E_FIGURE_OUTPUTS:
            if path.suffix == ".png":
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater(path.stat().st_size, 10_000)
            else:
                text = path.read_text(encoding="utf-8")
                self.assertIn("<svg", text)
                self.assertIn("</svg>", text)

    def test_stage3d_no_invented_author_or_submission_details(self) -> None:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in STAGE3D_FILES)
        self.assertNotRegex(combined, r"orcid\.org/|\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
        self.assertNotIn("Submission status: submitted", combined)
        self.assertNotIn("ready_for_submission", combined)
        self.assertIn("DOCX/PDF or submission", (PAPER / "PAPER_WRITING_STATUS.md").read_text(encoding="utf-8"))

    def test_stage3d_bspc_guideline_verification_is_official_source_only(self) -> None:
        text = (PAPER / "BSPC_GUIDELINES_VERIFICATION.md").read_text(encoding="utf-8")
        self.assertIn("No non-official source was used", text)
        self.assertIn("https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control", text)
        self.assertIn("https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors", text)
        self.assertIn("Subscription route", text)
        self.assertIn("not_verified", text)

    def test_stage3b_manuscript_metrics_are_preserved(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(), MANUSCRIPT_SHA256)
        citation_markers = re.findall(r"@([A-Za-z][A-Za-z0-9_:-]*)", text)
        self.assertEqual(len(citation_markers), 99)
        self.assertEqual(len(set(citation_markers)), 28)
        self.assertNotIn("[REFERENCE NEEDED:", text)

        status = (PAPER / "PAPER_WRITING_STATUS.md").read_text(encoding="utf-8")
        for phrase in [
            "Manuscript draft word count: 5309",
            "Experiment registry records: 42",
            "`awaiting_author_confirmation_for_target_journal`",
            "Provisional first target: Biomedical Signal Processing and Control",
            "Final journal not selected",
            "Figures not generated",
            "Journal formatting not applied",
            "Manuscript not submitted",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, status)

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
