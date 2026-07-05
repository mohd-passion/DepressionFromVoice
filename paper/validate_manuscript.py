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
