from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

BIB_PATH = PAPER / "references.bib"
MANUSCRIPT_PATH = PAPER / "MANUSCRIPT_DRAFT.md"
AUDIT_PATH = PAPER / "CITATION_PLACEHOLDER_AUDIT.csv"

DOCS_WITH_CITATIONS = [
    MANUSCRIPT_PATH,
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2A.md",
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2B.md",
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2C.md",
    PAPER / "DEPRESSION_SPEECH_EVIDENCE_TABLE.md",
    PAPER / "READING_INTERVIEW_LITERATURE_COMPARISON.md",
    PAPER / "CROSS_CORPUS_METHODS_LITERATURE_TABLE.md",
    PAPER / "DOMAIN_SHIFT_CONCEPT_TABLE.md",
    PAPER / "CALIBRATION_METHODS_TABLE.md",
    PAPER / "CLAIM_REFERENCE_MAP_STAGE2D.md",
    PAPER / "RESPONSIBLE_AI_EVIDENCE_TABLE.md",
    PAPER / "CLINICAL_READINESS_MATRIX.md",
    PAPER / "REFERENCE_VERIFICATION_REPORT.md",
    PAPER / "LITERATURE_SOURCE_INVENTORY.md",
    PAPER / "PAPER_REFERENCE_NEEDS.md",
]

STAGE3_REQUIRED_FILES = [
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

STAGE3B_REQUIRED_FILES = [
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

REQUIRED_MANUSCRIPT_SECTIONS = {
    "## Abstract",
    "## Keywords",
    "## 1. Introduction",
    "## 2. Related Work",
    "## 3. Materials and Methods",
    "## 4. Results",
    "## 5. Discussion",
    "### 5.8 Ethical and Clinical Considerations",
    "## 6. Limitations",
    "## 7. Conclusion",
    "## 8. Data and Code Availability",
}

STAGE2B_PLACEHOLDERS = {
    "[REFERENCE NEEDED: depression and speech as behavioral signal]",
    "[REFERENCE NEEDED: acoustic and prosodic correlates of depression]",
    "[REFERENCE NEEDED: handcrafted acoustic features in depression detection]",
    "[REFERENCE NEEDED: reading versus spontaneous speech in clinical speech analysis]",
    "[REFERENCE NEEDED: survey of speech-based depression detection]",
    "[REFERENCE NEEDED: evaluation variability in depression speech studies]",
    "[REFERENCE NEEDED: acoustic feature sets for depression speech]",
    "[REFERENCE NEEDED: reading speech and spontaneous speech comparison]",
}

STAGE2C_PLACEHOLDERS = {
    "[REFERENCE NEEDED: self-supervised speech representation learning]",
    "[REFERENCE NEEDED: wav2vec 2.0 and related SSL speech models]",
    "[REFERENCE NEEDED: cross-corpus depression detection]",
    "[REFERENCE NEEDED: domain shift and calibration in machine learning]",
    "[REFERENCE NEEDED: domain adaptation and dataset shift taxonomy]",
    "[REFERENCE NEEDED: classifier boundary shift under dataset shift]",
    "[REFERENCE NEEDED: prior cross-corpus findings in depression speech]",
    "[REFERENCE NEEDED: SSL transfer in paralinguistic or affective speech]",
}

STAGE2D_PLACEHOLDERS = {
    "[REFERENCE NEEDED: depression burden and prevalence]",
    "[REFERENCE NEEDED: ethical and clinical limits of speech-based mental-health models]",
}

STAGE2C_TABLE_MIN_ROWS = {
    PAPER / "CROSS_CORPUS_METHODS_LITERATURE_TABLE.md": 4,
    PAPER / "DOMAIN_SHIFT_CONCEPT_TABLE.md": 9,
    PAPER / "CALIBRATION_METHODS_TABLE.md": 5,
}

STAGE2D_TABLE_MIN_ROWS = {
    PAPER / "RESPONSIBLE_AI_EVIDENCE_TABLE.md": 14,
    PAPER / "CLINICAL_READINESS_MATRIX.md": 11,
}

REQUIRED_STAGE2C_PHRASES = {
    "target-domain threshold adaptation, not pure zero-shot evaluation",
    "target-domain score calibration rather than zero-shot evaluation",
    "not DAIC-to-Android model transfer",
    "ROC-AUC 0.6412",
    "ROC-AUC 0.8675",
    "0.5558",
    "predicted every Android interview participant as negative",
}

REQUIRED_STAGE2D_MANUSCRIPT_PHRASES = {
    "research-only",
    "not diagnostic tools",
    "No prospective deployment study was performed",
    "No demographic fairness audit was completed",
    "No clinical utility study was completed",
    "source datasets are not redistributed",
    "paper-experiment-freeze-v1",
    "raw audio",
    "participant-level artifacts",
}

REQUIRED_RESPONSIBLE_AI_ISSUES = {
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
}

REQUIRED_STAGE3_SCOPE_TERMS = {
    "DAIC methodological development",
    "provisional local labeled DAIC test",
    "cross-corpus zero-shot external evaluation",
    "target-domain threshold adaptation",
    "target-domain score calibration",
    "within-Android interview nested cross-validation",
    "within-Android reading nested cross-validation",
    "Cross-task transfer within Android",
    "Pooled-task within-Android evaluation",
}

REQUIRED_STAGE3_NUMERIC_VALUES = {
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
}

VALID_CHECKLIST_STATUSES = {
    "complete",
    "needs_author_input",
    "needs_formatting",
    "needs_figure",
    "needs_journal_selection",
    "not_applicable",
}

REQUIRED_FRONT_MATTER_PLACEHOLDERS = {
    "[AUTHOR INPUT REQUIRED: author names and order]",
    "[AUTHOR INPUT REQUIRED: affiliations]",
    "[AUTHOR INPUT REQUIRED: funding statement or \"No external funding\" if accurate]",
    "[AUTHOR INPUT REQUIRED: ethics/IRB statement for use of DAIC-WOZ and Androids Corpus data]",
    "[AUTHOR INPUT REQUIRED: disclose AI-assisted drafting/auditing if required by target journal or institution]",
}

REQUIRED_AUTHOR_INPUT_FIELDS = {
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
    "City",
    "State or region",
    "Country",
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
}

REQUIRED_CREDIT_ROLES = {
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
}

REQUIRED_DECISION_TOPICS = {
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
}

REQUIRED_AUTHOR_REVIEW_SECTIONS = {
    "Title",
    "Abstract",
    "Introduction",
    "Related work",
    "Datasets",
    "Preprocessing",
    "WavLM representation",
    "Evaluation protocols",
    "Results",
    "Discussion",
    "Limitations",
    "Ethical considerations",
    "Conclusion",
    "Data availability",
    "Code availability",
    "Tables",
    "Figures",
    "Supplementary material",
    "References",
}

AUTHOR_REVIEW_STATUSES = {
    "not_reviewed",
    "reviewed_no_change",
    "reviewed_changes_needed",
    "approved",
}

REQUIRED_AUTHOR_REVIEW_MARKERS = {
    "[AUTHOR DECISION REQUIRED: final title]",
    "[AUTHOR CONFIRMATION REQUIRED: author names, author order and affiliations]",
    "[AUTHOR DECISION REQUIRED: corresponding author]",
    "[AUTHOR DECISION REQUIRED: target journal and article type]",
    "[AUTHOR CONFIRMATION REQUIRED: data availability wording]",
    "[AUTHOR CONFIRMATION REQUIRED: code availability wording]",
    "[AUTHOR CONFIRMATION REQUIRED: funding]",
    "[AUTHOR CONFIRMATION REQUIRED: ethics wording]",
    "[AUTHOR CONFIRMATION REQUIRED: AI-assistance disclosure]",
    "[AUTHOR DECISION REQUIRED: preprint preference]",
}

REQUIRED_JOURNAL_MATRIX_COLUMNS = {
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

REQUIRED_VERIFICATION_LOG_COLUMNS = {
    "journal",
    "publisher",
    "information checked",
    "official source",
    "date checked",
    "status",
    "ambiguity",
    "action needed before submission",
}

REQUIRED_JOURNALS_REVIEWED = {
    "Biomedical Signal Processing and Control",
    "PLOS Digital Health",
    "Frontiers in Digital Health",
    "Computer Speech & Language",
    "Artificial Intelligence in Medicine",
    "Speech Communication",
}

JOURNAL_FIT_VALUES = {"excellent", "strong", "moderate", "weak"}

PROHIBITED_UNVERIFIED_JOURNAL_METRIC_TERMS = {
    "impact factor",
    "citescore",
    "acceptance rate",
    "quartile",
}

EXPECTED_AUTHORITATIVE_MANUSCRIPT_METRICS = {
    "citation_markers": 99,
    "unique_citation_keys": 28,
    "reference_placeholders": 0,
}

REQUIRED_CLINICAL_READINESS_CRITERIA = {
    "Retrospective technical validation": "completed",
    "Participant-level evaluation": "completed",
    "Cross-corpus zero-shot evaluation": "completed",
    "Target-domain calibration": "partially_completed",
    "Independent external validation of Android-trained models": "not_completed",
    "Subgroup fairness analysis": "not_completed",
    "Prospective clinical validation": "not_completed",
    "Comparison with clinician assessment": "not_completed",
    "Screening workflow evaluation": "not_completed",
    "Clinical utility evaluation": "not_completed",
    "Privacy and governance assessment": "partially_completed",
    "Deployment monitoring and drift management": "not_completed",
}

REQUIRED_CALIBRATION_TABLE_PHRASES = {
    "Threshold-only adaptation does not alter predicted probabilities.",
    "Calibration can improve probability scores without improving ranking.",
}

PROHIBITED_CLINICAL_OVERCLAIM_PHRASES = {
    "clinically validated",
    "diagnostic system",
    "diagnoses depression",
    "deployment-ready",
    "generalizes to all populations",
    "bias-free",
    "fair across demographics",
}

NEGATION_OR_LIMITING_MARKERS = {
    "not",
    "no ",
    "without",
    "unmet",
    "would require",
    "does not",
    "do not",
    "not ready",
    "not clinically",
}

REQUIRED_AUDIT_COLUMNS = {
    "placeholder_id",
    "placeholder_text",
    "resolution_status",
    "support_level",
    "source_limitations",
    "notes",
    "citation_key",
}

REQUIRED_NUMERICAL_AUDIT_COLUMNS = {
    "manuscript section",
    "table or paragraph",
    "reported value",
    "metric or quantity",
    "official phase",
    "experiment ID",
    "source file",
    "authoritative value",
    "match status",
    "correction required",
    "notes",
}

REQUIRED_CITATION_AUDIT_COLUMNS = {
    "manuscript section",
    "claim",
    "citation key",
    "BibTeX entry exists",
    "source verified",
    "direct support",
    "partial support",
    "contextual support",
    "primary or secondary source",
    "citation appropriate",
    "action required",
    "notes",
}


def bib_keys_and_dois() -> tuple[set[str], list[str], list[str]]:
    text = BIB_PATH.read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)
    dois = re.findall(r"doi\s*=\s*\{([^}]+)\}", text, flags=re.IGNORECASE)
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    duplicate_dois = sorted({doi.lower() for doi in dois if dois.count(doi) > 1})
    return set(keys), duplicate_keys, duplicate_dois


def citations_in_text(text: str) -> set[str]:
    return set(re.findall(r"@([A-Za-z][A-Za-z0-9_:-]*)", text))


def audit_rows() -> list[dict[str, str]]:
    with AUDIT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_AUDIT_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Audit CSV missing columns: {sorted(missing_columns)}")
        return list(reader)


def csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path.relative_to(ROOT)} missing columns: {sorted(missing_columns)}")
        return list(reader)


def citation_keys_from_audit(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def markdown_table_rows(path: Path) -> int:
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        if stripped.lower().startswith("| row "):
            continue
        rows += 1
    return rows


def markdown_table_data_rows(path: Path) -> int:
    rows = 0
    in_table = False
    saw_header = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            saw_header = False
            continue
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        if not in_table:
            in_table = True
            saw_header = True
            continue
        if saw_header:
            rows += 1
    return rows


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
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        if not header:
            header = cells
            in_table = True
            continue
        rows.append(cells)
    return header, rows


def sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def validate() -> list[str]:
    errors: list[str] = []
    bib_keys, duplicate_keys, duplicate_dois = bib_keys_and_dois()
    if duplicate_keys:
        errors.append(f"Duplicate BibTeX keys: {duplicate_keys}")
    if duplicate_dois:
        errors.append(f"Duplicate BibTeX DOIs: {duplicate_dois}")

    for path in DOCS_WITH_CITATIONS:
        if not path.exists():
            errors.append(f"Missing paper file: {path.relative_to(ROOT)}")
            continue
        missing = sorted(citations_in_text(path.read_text(encoding="utf-8")).difference(bib_keys))
        if missing:
            errors.append(f"{path.relative_to(ROOT)} cites keys missing from references.bib: {missing}")

    manuscript_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_MANUSCRIPT_SECTIONS:
        if section not in manuscript_text:
            errors.append(f"Manuscript missing required section: {section}")

    for path in STAGE3_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3A package file: {path.relative_to(ROOT)}")

    for path in STAGE3B_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3B package file: {path.relative_to(ROOT)}")

    manuscript_placeholders = set(re.findall(r"\[REFERENCE NEEDED: [^\]]+\]", manuscript_text))
    rows = audit_rows()
    audit_by_text = {row["placeholder_text"]: row for row in rows}
    for placeholder in manuscript_placeholders:
        row = audit_by_text.get(placeholder)
        if row is None:
            errors.append(f"Manuscript placeholder lacks audit row: {placeholder}")
        elif row["resolution_status"] != "unresolved":
            errors.append(f"Manuscript still contains resolved audit placeholder: {placeholder}")

    for placeholder in STAGE2B_PLACEHOLDERS:
        row = audit_by_text.get(placeholder)
        if row is None:
            errors.append(f"Stage 2B placeholder lacks audit row: {placeholder}")
            continue
        if placeholder in manuscript_placeholders:
            errors.append(f"Stage 2B placeholder still present in manuscript: {placeholder}")
        if row["resolution_status"] != "resolved":
            errors.append(f"Stage 2B placeholder not marked resolved in audit: {placeholder}")

    for placeholder in STAGE2C_PLACEHOLDERS:
        row = audit_by_text.get(placeholder)
        if row is None:
            errors.append(f"Stage 2C placeholder lacks audit row: {placeholder}")
            continue
        if placeholder in manuscript_placeholders:
            errors.append(f"Stage 2C placeholder still present in manuscript: {placeholder}")
        if row["resolution_status"] != "resolved":
            errors.append(f"Stage 2C placeholder not marked resolved in audit: {placeholder}")

    for placeholder in STAGE2D_PLACEHOLDERS:
        row = audit_by_text.get(placeholder)
        if row is None:
            errors.append(f"Stage 2D placeholder lacks audit row: {placeholder}")
            continue
        if placeholder in manuscript_placeholders:
            errors.append(f"Stage 2D placeholder still present in manuscript: {placeholder}")
        if row["resolution_status"] != "resolved":
            errors.append(f"Stage 2D placeholder not marked resolved in audit: {placeholder}")

    if manuscript_placeholders:
        errors.append(f"Manuscript still contains unresolved citation placeholders: {sorted(manuscript_placeholders)}")

    obvious_metric_placeholders = re.findall(r"\b(?:TBD|TODO|INSERT|0\.XXXX|X\.XXXX|XX\.XX)\b", manuscript_text)
    if obvious_metric_placeholders:
        errors.append(f"Manuscript contains obvious placeholder metrics/text: {sorted(set(obvious_metric_placeholders))}")

    for row in rows:
        if row["resolution_status"] != "resolved":
            continue
        keys = citation_keys_from_audit(row["citation_key"])
        if not keys:
            errors.append(f"Resolved audit row lacks citation key: {row['placeholder_id']}")
        for key in keys:
            if key not in bib_keys:
                errors.append(f"Audit row {row['placeholder_id']} uses unknown citation key: {key}")
        if not row["support_level"].strip():
            errors.append(f"Resolved audit row lacks support level: {row['placeholder_id']}")
        if not row["source_limitations"].strip():
            errors.append(f"Resolved audit row lacks source limitations: {row['placeholder_id']}")

    if "Because no bibliography was inserted" in manuscript_text:
        errors.append("Manuscript still contains obsolete no-bibliography wording.")

    evidence_rows = markdown_table_rows(PAPER / "DEPRESSION_SPEECH_EVIDENCE_TABLE.md")
    if evidence_rows < 9:
        errors.append(f"Depression speech evidence table has too few data rows: {evidence_rows}")

    reading_rows = markdown_table_rows(PAPER / "READING_INTERVIEW_LITERATURE_COMPARISON.md")
    if reading_rows < 4:
        errors.append(f"Reading/interview comparison table has too few data rows: {reading_rows}")

    for path, minimum_rows in STAGE2C_TABLE_MIN_ROWS.items():
        if not path.exists():
            errors.append(f"Missing Stage 2C table: {path.relative_to(ROOT)}")
            continue
        rows = markdown_table_data_rows(path)
        if rows < minimum_rows:
            errors.append(
                f"{path.relative_to(ROOT)} has too few data rows: {rows}; expected at least {minimum_rows}"
            )

    for path, minimum_rows in STAGE2D_TABLE_MIN_ROWS.items():
        if not path.exists():
            errors.append(f"Missing Stage 2D table: {path.relative_to(ROOT)}")
            continue
        rows = markdown_table_data_rows(path)
        if rows < minimum_rows:
            errors.append(
                f"{path.relative_to(ROOT)} has too few data rows: {rows}; expected at least {minimum_rows}"
            )

    for phrase in REQUIRED_STAGE2C_PHRASES:
        if phrase not in manuscript_text:
            errors.append(f"Manuscript missing Stage 2C safeguard phrase: {phrase}")

    for phrase in REQUIRED_STAGE2D_MANUSCRIPT_PHRASES:
        if phrase not in manuscript_text:
            errors.append(f"Manuscript missing Stage 2D safeguard phrase: {phrase}")

    calibration_text = (PAPER / "CALIBRATION_METHODS_TABLE.md").read_text(encoding="utf-8")
    for phrase in REQUIRED_CALIBRATION_TABLE_PHRASES:
        if phrase not in calibration_text:
            errors.append(f"Calibration table missing safeguard phrase: {phrase}")

    responsible_text = (PAPER / "RESPONSIBLE_AI_EVIDENCE_TABLE.md").read_text(encoding="utf-8")
    for issue in REQUIRED_RESPONSIBLE_AI_ISSUES:
        if issue not in responsible_text:
            errors.append(f"Responsible-AI table missing issue: {issue}")

    readiness_text = (PAPER / "CLINICAL_READINESS_MATRIX.md").read_text(encoding="utf-8")
    for criterion, status in REQUIRED_CLINICAL_READINESS_CRITERIA.items():
        pattern = rf"\|\s*{re.escape(criterion)}\s*\|\s*{status}\s*\|"
        if not re.search(pattern, readiness_text):
            errors.append(f"Clinical readiness matrix missing criterion/status: {criterion} -> {status}")
    if "Project is not clinically deployment-ready." not in readiness_text:
        errors.append("Clinical readiness matrix must explicitly state the project is not clinically deployment-ready.")

    if (PAPER / "FINAL_NUMERICAL_AUDIT.csv").exists():
        numeric_rows = csv_rows(PAPER / "FINAL_NUMERICAL_AUDIT.csv", REQUIRED_NUMERICAL_AUDIT_COLUMNS)
        if len(numeric_rows) < 80:
            errors.append(f"Final numerical audit has too few rows: {len(numeric_rows)}")
        bad_numeric_rows = [
            row["reported value"]
            for row in numeric_rows
            if row["correction required"].strip().lower() not in {"no", "none"}
        ]
        if bad_numeric_rows:
            errors.append(f"Final numerical audit still requires corrections: {bad_numeric_rows[:5]}")
        numeric_text = (PAPER / "FINAL_NUMERICAL_AUDIT.csv").read_text(encoding="utf-8")
        for value in REQUIRED_STAGE3_NUMERIC_VALUES:
            if value not in numeric_text:
                errors.append(f"Final numerical audit missing required value: {value}")

    if (PAPER / "FINAL_CITATION_AUDIT.csv").exists():
        citation_rows = csv_rows(PAPER / "FINAL_CITATION_AUDIT.csv", REQUIRED_CITATION_AUDIT_COLUMNS)
        cited_keys = citations_in_text(manuscript_text)
        audited_keys = {row["citation key"] for row in citation_rows}
        missing_audited_keys = sorted(cited_keys.difference(audited_keys))
        if missing_audited_keys:
            errors.append(f"Final citation audit missing manuscript keys: {missing_audited_keys}")
        inappropriate = [
            row["citation key"]
            for row in citation_rows
            if row["citation appropriate"].strip().lower() != "yes"
        ]
        if inappropriate:
            errors.append(f"Final citation audit contains inappropriate citation rows: {inappropriate[:5]}")

    if (PAPER / "FINAL_RESULT_SCOPE_AUDIT.md").exists():
        scope_text = (PAPER / "FINAL_RESULT_SCOPE_AUDIT.md").read_text(encoding="utf-8")
        for term in REQUIRED_STAGE3_SCOPE_TERMS:
            if term.lower() not in scope_text.lower():
                errors.append(f"Final result-scope audit missing term: {term}")

    if (PAPER / "FIGURE_SPECIFICATIONS.md").exists():
        figure_text = (PAPER / "FIGURE_SPECIFICATIONS.md").read_text(encoding="utf-8")
        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5", "Figure S1", "Figure S2"]:
            if label not in figure_text:
                errors.append(f"Figure specifications missing {label}")

    if (PAPER / "FINAL_TABLE_AUDIT.md").exists():
        table_text = (PAPER / "FINAL_TABLE_AUDIT.md").read_text(encoding="utf-8")
        for label in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Table 6"]:
            if label not in table_text:
                errors.append(f"Final table audit missing {label}")

    if (PAPER / "SUBMISSION_READINESS_CHECKLIST.md").exists():
        checklist_text = (PAPER / "SUBMISSION_READINESS_CHECKLIST.md").read_text(encoding="utf-8")
        if "`ready_for_author_review`" not in checklist_text:
            errors.append("Submission checklist missing ready_for_author_review readiness category.")
        for line in checklist_text.splitlines():
            separator = line.replace("|", "").replace(" ", "").strip()
            if not line.startswith("| ") or line.startswith("| Item ") or set(separator) <= {"-", ":"}:
                continue
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] not in VALID_CHECKLIST_STATUSES:
                errors.append(f"Submission checklist has invalid status: {cells[1]} in row {cells[0]}")

    if (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").exists():
        front_matter_text = (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").read_text(encoding="utf-8")
        for placeholder in REQUIRED_FRONT_MATTER_PLACEHOLDERS:
            if placeholder not in front_matter_text:
                errors.append(f"Front matter missing explicit placeholder: {placeholder}")

    if (PAPER / "AUTHOR_INPUT_FORM.md").exists():
        author_input_text = (PAPER / "AUTHOR_INPUT_FORM.md").read_text(encoding="utf-8")
        for field in REQUIRED_AUTHOR_INPUT_FIELDS:
            if field not in author_input_text:
                errors.append(f"Author input form missing field: {field}")
        for role in REQUIRED_CREDIT_ROLES:
            if role not in author_input_text:
                errors.append(f"Author input form missing CRediT role: {role}")
        if "Mohammad Ilyas" in author_input_text and "Unverified Author Context" not in author_input_text:
            errors.append("Known user name must remain inside an unverified author-context section.")
        if "Final affiliation wording | not_verified" not in author_input_text:
            errors.append("Author input form must mark final affiliation wording as not_verified.")

    if (PAPER / "AUTHOR_DECISION_LOG.md").exists():
        decision_text = (PAPER / "AUTHOR_DECISION_LOG.md").read_text(encoding="utf-8")
        for topic in REQUIRED_DECISION_TOPICS:
            if topic not in decision_text:
                errors.append(f"Author decision log missing topic: {topic}")
        if "Do not select final journal until author preferences are supplied" not in decision_text:
            errors.append("Author decision log must defer final journal selection.")

    if (PAPER / "MANUSCRIPT_AUTHOR_REVIEW_CHECKLIST.md").exists():
        checklist_text = (PAPER / "MANUSCRIPT_AUTHOR_REVIEW_CHECKLIST.md").read_text(encoding="utf-8")
        for status in AUTHOR_REVIEW_STATUSES:
            if status not in checklist_text:
                errors.append(f"Author review checklist missing status: {status}")
        for section in REQUIRED_AUTHOR_REVIEW_SECTIONS:
            if f"| {section} |" not in checklist_text:
                errors.append(f"Author review checklist missing section: {section}")

    if (PAPER / "MANUSCRIPT_AUTHOR_REVIEW_COPY.md").exists():
        review_copy_text = (PAPER / "MANUSCRIPT_AUTHOR_REVIEW_COPY.md").read_text(encoding="utf-8")
        for marker in REQUIRED_AUTHOR_REVIEW_MARKERS:
            if marker not in review_copy_text:
                errors.append(f"Author review copy missing marker: {marker}")
        if "ROC-AUC 0.6412" not in review_copy_text or "ROC-AUC 0.8675" not in review_copy_text:
            errors.append("Author review copy appears to have lost required result text.")

    citation_marker_count = len(re.findall(r"@([A-Za-z][A-Za-z0-9_:-]*)", manuscript_text))
    unique_citation_key_count = len(citations_in_text(manuscript_text))
    if citation_marker_count != EXPECTED_AUTHORITATIVE_MANUSCRIPT_METRICS["citation_markers"]:
        errors.append(f"Authoritative manuscript citation marker count changed: {citation_marker_count}")
    if unique_citation_key_count != EXPECTED_AUTHORITATIVE_MANUSCRIPT_METRICS["unique_citation_keys"]:
        errors.append(f"Authoritative manuscript unique citation key count changed: {unique_citation_key_count}")
    if len(manuscript_placeholders) != EXPECTED_AUTHORITATIVE_MANUSCRIPT_METRICS["reference_placeholders"]:
        errors.append(f"Authoritative manuscript citation placeholder count changed: {len(manuscript_placeholders)}")

    if (PAPER / "JOURNAL_CANDIDATE_MATRIX.md").exists():
        matrix_header, matrix_rows = first_markdown_table(PAPER / "JOURNAL_CANDIDATE_MATRIX.md")
        normalized_header = {item.strip() for item in matrix_header}
        missing_matrix_columns = REQUIRED_JOURNAL_MATRIX_COLUMNS.difference(normalized_header)
        if missing_matrix_columns:
            errors.append(f"Journal candidate matrix missing columns: {sorted(missing_matrix_columns)}")
        if not 6 <= len(matrix_rows) <= 10:
            errors.append(f"Journal candidate matrix must evaluate 6-10 journals; found {len(matrix_rows)}")
        matrix_text = (PAPER / "JOURNAL_CANDIDATE_MATRIX.md").read_text(encoding="utf-8")
        for journal in REQUIRED_JOURNALS_REVIEWED:
            if journal not in matrix_text:
                errors.append(f"Journal candidate matrix missing journal: {journal}")
        for row in matrix_rows:
            if len(row) != len(matrix_header):
                errors.append(f"Journal candidate matrix has malformed row: {row[:2]}")
                continue
            row_data = dict(zip(matrix_header, row))
            if row_data.get("verification date") != "2026-07-06":
                errors.append(f"Journal row lacks verification date: {row_data.get('journal', row[0])}")
            if "https://" not in row_data.get("official source", ""):
                errors.append(f"Journal row lacks official source URL: {row_data.get('journal', row[0])}")
            for fit_column in ["scope fit", "speech/affective-computing fit", "mental-health fit", "methodological fit"]:
                if row_data.get(fit_column) not in JOURNAL_FIT_VALUES:
                    errors.append(
                        f"Journal row has invalid {fit_column}: {row_data.get('journal', row[0])} -> {row_data.get(fit_column)}"
                    )
        if "not_verified" not in matrix_text:
            errors.append("Journal candidate matrix must mark unresolved information as not_verified.")

    if (PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md").exists():
        log_header, log_rows = first_markdown_table(PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md")
        normalized_log_header = {item.strip() for item in log_header}
        missing_log_columns = REQUIRED_VERIFICATION_LOG_COLUMNS.difference(normalized_log_header)
        if missing_log_columns:
            errors.append(f"Journal verification log missing columns: {sorted(missing_log_columns)}")
        if len(log_rows) < 6:
            errors.append(f"Journal verification log has too few rows: {len(log_rows)}")
        for row in log_rows:
            if len(row) != len(log_header):
                errors.append(f"Journal verification log has malformed row: {row[:2]}")
                continue
            row_data = dict(zip(log_header, row))
            if row_data.get("date checked") != "2026-07-06":
                errors.append(f"Verification log row lacks checked date: {row_data.get('journal', row[0])}")
            if "https://" not in row_data.get("official source", ""):
                errors.append(f"Verification log row lacks official source URL: {row_data.get('journal', row[0])}")
            if row_data.get("status") not in {"verified", "partially_verified", "not_verified"}:
                errors.append(f"Verification log row has invalid status: {row_data.get('status')}")
        log_text = (PAPER / "JOURNAL_INFORMATION_VERIFICATION_LOG.md").read_text(encoding="utf-8")
        if "not_verified" not in log_text:
            errors.append("Journal verification log must include not_verified for unresolved details.")

    stage3b_text = "\n".join(
        path.read_text(encoding="utf-8") for path in STAGE3B_REQUIRED_FILES if path.exists()
    )
    for term in PROHIBITED_UNVERIFIED_JOURNAL_METRIC_TERMS:
        if term in stage3b_text.lower():
            errors.append(f"Stage 3B journal package contains prohibited unverified metric term: {term}")
    if (
        "guaranteed acceptance" in stage3b_text.lower()
        and "no candidate should be described as guaranteed acceptance" not in stage3b_text.lower()
    ):
        errors.append("Journal package must not suggest guaranteed acceptance.")

    if (PAPER / "PAPER_WRITING_STATUS.md").exists():
        status_text = (PAPER / "PAPER_WRITING_STATUS.md").read_text(encoding="utf-8")
        for phrase in [
            "`awaiting_author_input_and_journal_selection`",
            "Author details missing",
            "Declaration details missing",
            "Journal preferences missing",
            "Final journal not selected",
            "Figures not generated",
            "Journal formatting not applied",
            "Manuscript not submitted",
            "Manuscript draft word count: 5309",
            "Experiment registry records: 42",
        ]:
            if phrase not in status_text:
                errors.append(f"Paper writing status missing Stage 3B phrase: {phrase}")

    if (PAPER / "TERMINOLOGY_STYLE_GUIDE.md").exists():
        terminology_text = (PAPER / "TERMINOLOGY_STYLE_GUIDE.md").read_text(encoding="utf-8")
        for term in ["within-corpus nested cross-validation", "diagnostic system", "research classification model"]:
            if term not in terminology_text:
                errors.append(f"Terminology guide missing term: {term}")

    for sentence in sentences(manuscript_text):
        normalized = sentence.lower()
        if "android-trained" in normalized and "external validation" in normalized:
            if "not external validation" not in normalized and "not cross-corpus external validation" not in normalized:
                errors.append(f"Android-trained result may be overstated as external validation: {sentence}")
        if "target-domain" in normalized and "zero-shot" in normalized:
            safe_markers = {
                "separation between",
                "different questions",
                "zero-shot result measures",
                "distinguishing",
                "not pure zero-shot",
                "rather than zero-shot",
            }
            if not any(marker in normalized for marker in safe_markers):
                errors.append(f"Target-domain protocol may be overstated as zero-shot: {sentence}")
        for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
            if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                errors.append(f"Potential clinical overclaim without negation or limitation: {sentence}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("manuscript validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
