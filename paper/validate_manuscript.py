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
    PAPER / "DEPRESSION_SPEECH_EVIDENCE_TABLE.md",
    PAPER / "READING_INTERVIEW_LITERATURE_COMPARISON.md",
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
