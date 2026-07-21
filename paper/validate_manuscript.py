from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

BIB_PATH = PAPER / "references.bib"
MANUSCRIPT_PATH = PAPER / "MANUSCRIPT_DRAFT.md"
AUDIT_PATH = PAPER / "CITATION_PLACEHOLDER_AUDIT.csv"

DOCS_WITH_CITATIONS = [
    MANUSCRIPT_PATH,
    PAPER / "BSPC_MANUSCRIPT_DRAFT.md",
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

STAGE3C_REQUIRED_FILES = [
    PAPER / "AUTHOR_PREFERENCES_PRELIMINARY.md",
    PAPER / "PROVISIONAL_TARGET_JOURNAL_DECISION.md",
    PAPER / "JOURNAL_ADAPTATION_PLAN.md",
    PAPER / "AUTHOR_FINAL_QUESTIONS.md",
]

STAGE3D_REQUIRED_FILES = [
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

STAGE3E_REQUIRED_FILES = [
    PAPER / "BSPC_FIGURE_CAPTIONS.md",
    PAPER / "BSPC_FIGURE_DATA_AUDIT.csv",
    PAPER / "figures" / "README.md",
    PAPER / "figures" / "scripts" / "generate_bspc_figures.py",
    PAPER / "figures" / "source" / "bspc_figure_source_values.csv",
    PAPER / "figures" / "output" / "figure_generation_log.txt",
    *STAGE3E_FIGURE_OUTPUTS,
]

STAGE3F_REQUIRED_FILES = [
    PAPER / "BSPC_FIGURE_VISUAL_REVIEW.md",
    PAPER / "BSPC_DOCX_READY_MANUSCRIPT.md",
    PAPER / "BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md",
    PAPER / "BSPC_DOCX_CONVERSION_NOTES.md",
]

STAGE3G_REQUIRED_FILES = [
    PAPER / "scripts" / "create_bspc_review_docx.py",
    PAPER / "submission_review_package" / "README.md",
    PAPER / "submission_review_package" / "AUTHOR_REVIEW_PACKET_CHECKLIST.md",
    PAPER / "submission_review_package" / "logs" / "BSPC_DOCX_GENERATION_LOG.md",
    PAPER / "submission_review_package" / "docx" / "BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx",
    PAPER / "submission_review_package" / "docx" / "BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx",
    PAPER / "submission_review_package" / "docx" / "BSPC_MANUSCRIPT_FINAL_REVIEW.docx",
    PAPER / "submission_review_package" / "docx" / "BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx",
]

EXPERIMENT_RATIONALE_REQUIRED_FILES = [
    PAPER / "STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md",
    PAPER / "STUDY_EXPERIMENTS_RATIONALE_TABLE.csv",
    PAPER / "EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md",
    PAPER / "EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md",
    PAPER / "EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md",
    PAPER / "EXPERIMENT_DEPENDENCY_FLOW.md",
    PAPER / "EXPERIMENT_SCOPE_BOUNDARIES.md",
]

VOICE_PHYSIOLOGY_AUDIT_FILE = PAPER / "VOICE_PHYSIOLOGY_INSERT_AUDIT.md"

IEEE_TCSS_DIR = PAPER / "ieee_tcss"

STAGE4A_IEEE_TCSS_REQUIRED_FILES = [
    IEEE_TCSS_DIR / "README.md",
    IEEE_TCSS_DIR / "main.tex",
    IEEE_TCSS_DIR / "declarations.tex",
    IEEE_TCSS_DIR / "build_latex.ps1",
    IEEE_TCSS_DIR / "build_latex.py",
    IEEE_TCSS_DIR / "IEEE_TCSS_GUIDELINES_VERIFICATION.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_JOURNAL_FIT_AUDIT.md",
    IEEE_TCSS_DIR / "TITLE_REFRAMING_OPTIONS.md",
    IEEE_TCSS_DIR / "IEEE_REFERENCE_CONVERSION_AUDIT.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_COVER_LETTER_DRAFT.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_SUBMISSION_CHECKLIST.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_MIGRATION_AUDIT.md",
    IEEE_TCSS_DIR / "logs" / "LATEX_BUILD_LOG.md",
    IEEE_TCSS_DIR / "template" / "bare_jrnl_new_sample4.tex",
    IEEE_TCSS_DIR / "bib" / "references.bib",
    IEEE_TCSS_DIR / "supplementary" / "supplementary.tex",
    IEEE_TCSS_DIR / "IEEE_TEMPLATE_ALIGNMENT_AUDIT.md",
    IEEE_TCSS_DIR / "LATEX_SOURCE_SANITY_AUDIT.md",
    IEEE_TCSS_DIR / "LATEX_FIGURE_AUDIT.md",
    IEEE_TCSS_DIR / "LATEX_TABLE_AUDIT.md",
    IEEE_TCSS_DIR / "LOCAL_LATEX_SETUP_NOTES.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_RESEARCH_QUESTIONS_AUDIT.md",
    IEEE_TCSS_DIR / "IEEE_TCSS_RESEARCH_GAP_AUDIT.md",
    IEEE_TCSS_DIR / "RQ_TO_EVIDENCE_MAP.md",
    IEEE_TCSS_DIR / "IEEE_LATEX_COMPILE_QA_REPORT.md",
    IEEE_TCSS_DIR / "IEEE_PDF_VISUAL_REVIEW_CHECKLIST.md",
    IEEE_TCSS_DIR / "overleaf_upload" / "OVERLEAF_UPLOAD_README.md",
    IEEE_TCSS_DIR / "overleaf_upload" / "OVERLEAF_ZIP_MANIFEST.md",
    PAPER / "scripts" / "audit_ieee_tcss_latex_package.py",
]

STAGE4A_IEEE_TCSS_TABLE_FILES = [
    IEEE_TCSS_DIR / "tables" / "table1_dataset_task_summary.tex",
    IEEE_TCSS_DIR / "tables" / "table2_daic_development_results.tex",
    IEEE_TCSS_DIR / "tables" / "table3_cross_corpus_android_interview.tex",
    IEEE_TCSS_DIR / "tables" / "table4_android_reading_results.tex",
    IEEE_TCSS_DIR / "tables" / "table5_cross_task_results.tex",
    IEEE_TCSS_DIR / "tables" / "table6_result_scope_claim_matrix.tex",
]

STAGE4A_IEEE_TCSS_FIGURE_FILES = [
    IEEE_TCSS_DIR / "figures" / "figure1_study_design_protocol_overview.png",
    IEEE_TCSS_DIR / "figures" / "figure2_daic_phase1_key_results.png",
    IEEE_TCSS_DIR / "figures" / "figure3_daic_android_transfer_adaptation.png",
    IEEE_TCSS_DIR / "figures" / "figure4_android_cross_task_results.png",
    IEEE_TCSS_DIR / "figures" / "figure5_domain_shift_boundary_summary.png",
    IEEE_TCSS_DIR / "figures" / "figure1_study_design_protocol_overview.svg",
    IEEE_TCSS_DIR / "figures" / "figure2_daic_phase1_key_results.svg",
    IEEE_TCSS_DIR / "figures" / "figure3_daic_android_transfer_adaptation.svg",
    IEEE_TCSS_DIR / "figures" / "figure4_android_cross_task_results.svg",
    IEEE_TCSS_DIR / "figures" / "figure5_domain_shift_boundary_summary.svg",
]

STAGE4A_IEEE_TCSS_METRICS = {
    "0.6412",
    "0.8675",
    "0.8796",
    "0.8475",
    "0.8501",
    "0.9058",
    "0.9135",
    "550 shifted WavLM features",
    "0.4752",
    "0.6659",
    "0.1216",
}

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
    "figures_generated_awaiting_author_review",
    "pending_portal",
    "not_started",
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

REQUIRED_STAGE3C_PREFERENCE_PHRASES = {
    "SCI/SCIE mandatory | No",
    "Q1 mandatory | No",
    "Scopus acceptable if SCI/SCIE unavailable | Yes",
    "Maximum APC | not specified",
    "Must have no-APC subscription route | Yes",
    "Open access mandatory | No",
    "biomedical signal processing",
    "speech/language processing",
    "medical AI/digital health",
    "affective computing",
    "highest quartile where possible",
    "ORCID | not available/not provided",
    "Additional authors | no additional authors provided at this stage",
    "Treat `N/A` as not yet provided",
}

REQUIRED_UNRESOLVED_DECLARATION_MARKERS = {
    "Funding | [NOT PROVIDED]",
    "Competing interests | [NOT PROVIDED]",
    "Ethics approval or exemption | [NOT PROVIDED]",
    "Consent to participate | [NOT PROVIDED]",
    "Consent for publication | [NOT PROVIDED]",
    "Acknowledgements | [NOT PROVIDED]",
    "Use of AI-assisted writing tools | [NOT PROVIDED]",
    "Prior presentation or preprint status | [NOT PROVIDED]",
}

REQUIRED_STAGE3E_AUTHOR_PHRASES = {
    "Full name | Mohammad Ilyas",
    "Institutional email | mohd.passion.cse@mitmoradabad.edu.in",
    "ORCID | not available",
    "Corresponding-author status | Yes",
    "Full name | Dr. Vikram Singh",
    "Institutional email | viks@nitkkr.ac.in",
    "ORCID | not available",
    "Corresponding-author status | No",
}

REQUIRED_STAGE3E_DECLARATION_PHRASES = {
    "Funding | No funding was received for this study.",
    "Competing interests | The authors declare no competing interests.",
    "Consent to participate | This study used secondary datasets obtained under their original access terms.",
    "Consent for publication | This study used secondary datasets obtained under their original access terms.",
    "Acknowledgements | The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets.",
    "Use of AI-assisted writing tools | ChatGPT and Codex were used to assist with code organization",
    "Prior presentation or preprint status | A preprint is planned.",
    "BSPC submission route | The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
}

REQUIRED_STAGE3E_UNRESOLVED_TEXT: set[str] = set()

REQUIRED_STAGE3E_CONFIRMED_DECLARATIONS = {
    "No funding was received for this study.",
    "The authors declare no competing interests.",
    "This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.",
}

REQUIRED_STAGE3C_DECISION_PHRASES = {
    "Decision status: `provisional_target_selected`",
    "Final submission status: `not_selected`",
    "This is not a final author decision",
    "needs_reverification_before_submission",
    "No current impact factor",
    "No current",
    "Until those items are confirmed",
    "awaiting_author_confirmation_for_target_journal",
}

BSPC_DECLARATION_SECTIONS = {
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
}

REQUIRED_BSPC_PHRASES = {
    "Target journal: Biomedical Signal Processing and Control",
    "Submission status: journal-specific preparation only; not submitted.",
    "Article type: Full paper; final portal article-type mapping remains pending.",
    "## Highlights",
    "## Graphical Abstract",
    "## Declaration of Competing Interest",
    "## Funding",
    "## Ethics Approval",
    "## Consent to Participate",
    "## Consent for Publication",
    "## CRediT Authorship Contribution Statement",
    "## Acknowledgements",
    "## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process",
    "## Preprint Statement",
}

REQUIRED_BSPC_METRICS = {
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

REQUIRED_FIGURE_AUDIT_COLUMNS = {
    "figure",
    "panel",
    "metric",
    "plotted value",
    "source file",
    "official value",
    "match status",
    "notes",
}

REQUIRED_RATIONALE_TABLE_COLUMNS = {
    "phase",
    "experiment_name",
    "experiment_id_or_run_id",
    "dataset",
    "speech_task",
    "training_domain",
    "evaluation_domain",
    "model_or_feature_set",
    "classifier_or_protocol",
    "main_question",
    "rationale",
    "key_result",
    "significance",
    "main_limitation",
    "next_step_motivated",
    "manuscript_section_supported",
    "official_source_file",
}

REQUIRED_RATIONALE_EXPERIMENT_IDS = {
    "phase1a_metadata_baseline",
    "phase1a_wavlm_final_layer",
    "phase1b_wavlm_layer_09",
    "phase1c_aggregation_mean",
    "phase1d_timing_only",
    "phase1e_timing_only",
    "phase1f_frozen_wavlm",
    "phase1f_frozen_timing",
    "phase2a_android_audit",
    "phase2b_daic_to_android_zero_shot_wavlm",
    "phase2c_protocol_a_frozen_daic_reference",
    "phase2c_protocol_b_threshold_only",
    "phase2c_protocol_c_score_calibration",
    "phase2c_protocol_d_android_lr",
    "phase2c_domain_shift_summary",
    "phase2d_android_reading_zero_shot_daic",
    "phase2d_android_reading_logistic_regression",
    "phase2d_android_interview_vs_reading_comparison",
    "phase2e_matched_reading_within_task",
    "phase2e_matched_interview_within_task",
    "phase2e_reading_to_interview_transfer",
    "phase2e_interview_to_reading_transfer",
    "phase2e_pooled_to_reading",
    "phase2e_pooled_to_interview",
    "phase2e_cross_task_transfer_conclusion",
}

VOICE_PHYSIOLOGY_REQUIRED_CITATIONS = {
    "leal2025survey",
    "valstar2016avec",
    "tao2023featureCorrelation",
    "chen2022wavlm",
    "yang2021superb",
}

VOICE_PHYSIOLOGY_REQUIRED_PHRASES = {
    "respiratory, phonatory, resonatory and articulatory processes",
    "source-filter terms",
    "vocal-fold vibration",
    "subglottal pressure",
    "reduced pitch variability",
    "lower energy",
    "slower articulation",
    "longer pauses",
    "breathier voice quality",
    "should not be treated as diagnostic markers by themselves",
    "not as a standalone clinical model",
}

VOICE_PHYSIOLOGY_FORBIDDEN_PHRASES = {
    "DSM-5",
    "PHQ-8 severity",
    "DAIC-WOZ",
    "real-time monitoring",
    "privacy-preserving",
    "decision-support tool",
    "clinically validated",
    "deployment-ready",
    "diagnoses depression",
    "diagnostic system",
    "fairness was addressed",
}

REQUIRED_FIGURE_AUDIT_VALUES = {
    ("Figure 2", "ROC-AUC", "0.5065"),
    ("Figure 2", "ROC-AUC", "0.6775"),
    ("Figure 3", "ROC-AUC", "0.6412"),
    ("Figure 3", "Macro F1", "0.3095"),
    ("Figure 3", "Macro F1", "0.5688"),
    ("Figure 3", "ROC-AUC", "0.5694"),
    ("Figure 3", "ROC-AUC", "0.8675"),
    ("Figure 3", "Macro F1", "0.8263"),
    ("Figure 4", "ROC-AUC", "0.8796"),
    ("Figure 4", "ROC-AUC", "0.8675"),
    ("Figure 4", "ROC-AUC", "0.9048"),
    ("Figure 4", "ROC-AUC", "0.9065"),
    ("Figure 4", "ROC-AUC", "0.8475"),
    ("Figure 4", "ROC-AUC", "0.8501"),
    ("Figure 4", "ROC-AUC", "0.9058"),
    ("Figure 4", "ROC-AUC", "0.9135"),
    ("Figure 5", "DAIC-to-Android centroid cosine distance", "0.1767"),
    ("Figure 5", "Android reading/interview centroid cosine distance", "0.0442"),
    ("Figure 5", "DAIC-to-Android mean absolute standardized shift", "2.3116"),
    ("Figure 5", "Android reading/interview mean absolute standardized shift", "0.8831"),
    ("Figure 5", "DAIC-to-Android shifted features", "550/768"),
    ("Figure 5", "Android reading/interview shifted features", "292/768"),
    ("Figure 5", "DAIC-to-Android MMD^2", "0.6091"),
    ("Figure 5", "Coeff cosine", "0.4752"),
    ("Figure 5", "Sign agreement", "0.6659"),
    ("Figure 5", "Top-50 Jaccard", "0.1216"),
    ("Figure 5", "Matched score corr.", "0.7123"),
    ("Figure 5", "Cross-target corr.", "0.5555"),
}

REQUIRED_STAGE3F_DOCX_PLACEHOLDERS = {
    "[AUTHOR FINAL CONFIRMATION REQUIRED: the manuscript is not under consideration elsewhere.]",
    "[AUTHOR FINAL ACTION REQUIRED: final BSPC/Elsevier portal metadata entry and upload.]",
    "[AUTHOR FINAL ACTION REQUIRED: final BSPC submission-system technical checks.]",
    "[AUTHOR FINAL REVIEW REQUIRED: final cover letter review and journal-generated PDF proof check.]",
}

REQUIRED_STAGE3F_AUTHOR_QUESTIONS = {
    "Confirm Dr. Vikram Singh's email.",
    "Confirm Dr. Vikram Singh's ORCID, or state not available.",
    "Confirm Dr. Vikram Singh's CRediT roles.",
    "Confirm the consent statement for use of DAIC-WOZ and Androids Corpus data.",
    "Confirm acknowledgements, or state that no acknowledgements should be included.",
    "Confirm AI-assisted writing/tool-use disclosure wording according to BSPC/Elsevier policy.",
    "Confirm the preprint decision.",
    "Confirm the BSPC subscription/no-publication-fee route.",
    "Review and approve generated figures.",
    "Confirm whether to convert/submit after DOCX review.",
}

REQUIRED_STAGE3G_AUTHOR_QUESTIONS = {
    "Confirm the cover-letter statement: the manuscript is not under consideration elsewhere.",
    "Complete final BSPC/Elsevier portal metadata entry.",
    "Upload the manuscript, supplementary DOCX and required figure/source files in the BSPC portal.",
    "Complete final BSPC submission-system technical checks.",
    "Review the journal-generated PDF proof before approval.",
    "Confirm whether to submit after proof and portal checks are complete.",
}

FORBIDDEN_GENERATED_ARTIFACT_PATTERNS = {
    r"\bparticipant[_ -]?id\b",
    r"\bsubject[_ -]?id\b",
    r"\brecording[_ -]?id\b",
    r"\baudio[_ -]?path\b",
    r"\bfile[_ -]?path\b",
}


def bib_keys_and_dois() -> tuple[set[str], list[str], list[str]]:
    text = BIB_PATH.read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)
    dois = re.findall(r"doi\s*=\s*\{([^}]+)\}", text, flags=re.IGNORECASE)
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    duplicate_dois = sorted({doi.lower() for doi in dois if dois.count(doi) > 1})
    return set(keys), duplicate_keys, duplicate_dois


def citations_in_text(text: str) -> set[str]:
    # Avoid treating email domains as citation keys.
    return set(re.findall(r"(?<![A-Za-z0-9._%+-])@([A-Za-z][A-Za-z0-9_:-]*)", text))


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


def docx_text_and_media_count(path: Path) -> tuple[str, int, bool]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                return "", 0, False
            root = ET.fromstring(archive.read("word/document.xml"))
            text = "".join(
                node.text or ""
                for node in root.iter()
                if node.tag == f"{{{root.tag.split('}')[0].strip('{')}}}t" or node.tag.endswith("}t")
            )
            media_count = sum(1 for name in names if name.startswith("word/media/"))
    except (zipfile.BadZipFile, ET.ParseError, KeyError):
        return "", 0, False
    return text, media_count, True


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


def contains_prohibited_wording(text: str, phrase: str) -> bool:
    if " " in phrase or "-" in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def voice_physiology_section(text: str) -> str:
    match = re.search(
        r"### 2\.2 Physiological motivation for speech-based depression analysis\s+(.*?)\s+### 2\.3 Handcrafted acoustic and timing features",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


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

    for path in STAGE3C_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3C package file: {path.relative_to(ROOT)}")

    for path in STAGE3D_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3D package file: {path.relative_to(ROOT)}")

    for path in STAGE3E_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3E package file: {path.relative_to(ROOT)}")

    for path in STAGE3F_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3F package file: {path.relative_to(ROOT)}")

    for path in STAGE3G_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing Stage 3G package file: {path.relative_to(ROOT)}")

    for path in EXPERIMENT_RATIONALE_REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Missing experiment-rationale package file: {path.relative_to(ROOT)}")

    if not VOICE_PHYSIOLOGY_AUDIT_FILE.exists():
        errors.append(f"Missing voice physiology insert audit: {VOICE_PHYSIOLOGY_AUDIT_FILE.relative_to(ROOT)}")

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
        if "`bspc_final_review_package_ready_for_submission_portal_checks`" not in checklist_text:
            errors.append("Submission checklist missing Stage 3H readiness category.")
        for line in checklist_text.splitlines():
            separator = line.replace("|", "").replace(" ", "").strip()
            if not line.startswith("| ") or line.startswith("| Item ") or set(separator) <= {"-", ":"}:
                continue
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] not in VALID_CHECKLIST_STATUSES:
                errors.append(f"Submission checklist has invalid status: {cells[1]} in row {cells[0]}")

    if (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").exists():
        front_matter_text = (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").read_text(encoding="utf-8")
        if "Mohammad Ilyas" in front_matter_text and "No funding was received for this study." in front_matter_text:
            for phrase in [
                "1. Mohammad Ilyas",
                "2. Dr. Vikram Singh",
                "mohd.passion.cse@mitmoradabad.edu.in",
                "Dr. Vikram Singh is not the corresponding author.",
                "Mohammad Ilyas: not available.",
                "Dr. Vikram Singh: not available.",
                "This study used secondary datasets obtained under their original access terms.",
                "The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets.",
                "ChatGPT and Codex were used to assist with code organization",
                "A preprint is planned.",
                "The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
            ]:
                if phrase not in front_matter_text:
                    errors.append(f"Stage 3E front matter missing phrase: {phrase}")
            for phrase in [
                "No funding was received for this study.",
                "The authors declare no competing interests.",
                "This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.",
            ]:
                if phrase not in front_matter_text:
                    errors.append(f"Stage 3E front matter missing confirmed declaration: {phrase}")
        else:
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
        if "No funding was received for this study." in author_input_text:
            if "Final affiliation wording | Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India | confirmed for Stage 3H final-review package |" not in author_input_text:
                errors.append("Author input form must retain Stage 3H confirmed affiliation wording.")
            for phrase in REQUIRED_STAGE3E_AUTHOR_PHRASES:
                if phrase not in author_input_text:
                    errors.append(f"Author input form missing Stage 3E author phrase: {phrase}")
            for phrase in REQUIRED_STAGE3E_DECLARATION_PHRASES:
                if phrase not in author_input_text:
                    errors.append(f"Author input form missing Stage 3E declaration phrase: {phrase}")
            if "Use of AI-assisted writing tools | No" in author_input_text:
                errors.append("Author input form must not finalize AI-assisted writing disclosure as No.")
        else:
            if "Final affiliation wording | not_verified" not in author_input_text:
                errors.append("Author input form must mark final affiliation wording as not_verified.")
            for marker in REQUIRED_UNRESOLVED_DECLARATION_MARKERS:
                if marker not in author_input_text:
                    errors.append(f"Author input form must keep unresolved declaration marked: {marker}")
            if "| ORCID | [NOT PROVIDED] |" not in author_input_text:
                errors.append("Author input form must not invent ORCID values.")
        if "orcid.org/" in author_input_text.lower() or re.search(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", author_input_text):
            errors.append("Author input form must not invent ORCID identifiers.")
        if "No external funding" in author_input_text or "| Funding | N/A" in author_input_text:
            errors.append("Author input form must not invent or finalize funding wording.")

    if (PAPER / "AUTHOR_DECISION_LOG.md").exists():
        decision_text = (PAPER / "AUTHOR_DECISION_LOG.md").read_text(encoding="utf-8")
        for topic in REQUIRED_DECISION_TOPICS:
            if topic not in decision_text:
                errors.append(f"Author decision log missing topic: {topic}")
        if "Biomedical Signal Processing and Control selected for the BSPC final review package" not in decision_text:
            errors.append("Author decision log must record the Stage 3H BSPC target decision.")
        if "The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route." not in decision_text:
            errors.append("Author decision log must record the Stage 3H subscription-route decision.")

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
    if re.search(
        r"\b(?:impact factor|citescore|acceptance rate|acceptance-rate|quartile)\s*[:=]\s*\d",
        stage3b_text,
        flags=re.IGNORECASE,
    ):
        errors.append("Stage 3B journal package must not assert numeric journal metrics.")
    if (
        "guaranteed acceptance" in stage3b_text.lower()
        and "no candidate should be described as guaranteed acceptance" not in stage3b_text.lower()
    ):
        errors.append("Journal package must not suggest guaranteed acceptance.")

    if all(path.exists() for path in STAGE3C_REQUIRED_FILES):
        stage3c_text = "\n".join(path.read_text(encoding="utf-8") for path in STAGE3C_REQUIRED_FILES)
        preference_text = (PAPER / "AUTHOR_PREFERENCES_PRELIMINARY.md").read_text(encoding="utf-8")
        for phrase in REQUIRED_STAGE3C_PREFERENCE_PHRASES:
            if phrase not in preference_text:
                errors.append(f"Preliminary author preferences missing phrase: {phrase}")

        decision_text = (PAPER / "PROVISIONAL_TARGET_JOURNAL_DECISION.md").read_text(encoding="utf-8")
        for phrase in REQUIRED_STAGE3C_DECISION_PHRASES:
            if phrase not in decision_text:
                errors.append(f"Provisional target decision missing phrase: {phrase}")
        if "Biomedical Signal Processing and Control" not in decision_text:
            errors.append("Provisional target decision must name Biomedical Signal Processing and Control.")
        if "Final submission status: `selected`" in decision_text:
            errors.append("Stage 3C must not mark a final journal as selected.")

        adaptation_text = (PAPER / "JOURNAL_ADAPTATION_PLAN.md").read_text(encoding="utf-8")
        for phrase in [
            "does not authorize submission",
            "No final figures generated in Stage 3C",
            "No reference-style conversion in Stage 3C",
            "Do not add new experiments",
        ]:
            if phrase not in adaptation_text:
                errors.append(f"Journal adaptation plan missing safeguard: {phrase}")

        final_questions = re.findall(r"^\d+\. ", (PAPER / "AUTHOR_FINAL_QUESTIONS.md").read_text(encoding="utf-8"), flags=re.MULTILINE)
        if all(path.exists() for path in STAGE3G_REQUIRED_FILES):
            expected_question_count = 6
        elif all(path.exists() for path in STAGE3F_REQUIRED_FILES):
            expected_question_count = 10
        elif any(path.exists() for path in STAGE3D_REQUIRED_FILES):
            expected_question_count = 17
        else:
            expected_question_count = 15
        if len(final_questions) != expected_question_count:
            errors.append(
                f"Author final questions must contain exactly {expected_question_count} numbered questions; found {len(final_questions)}"
            )
        if all(path.exists() for path in STAGE3G_REQUIRED_FILES):
            questions_text = (PAPER / "AUTHOR_FINAL_QUESTIONS.md").read_text(encoding="utf-8")
            for question in REQUIRED_STAGE3G_AUTHOR_QUESTIONS:
                if question not in questions_text:
                    errors.append(f"Stage 3G author questions missing: {question}")
        elif expected_question_count == 10:
            questions_text = (PAPER / "AUTHOR_FINAL_QUESTIONS.md").read_text(encoding="utf-8")
            for question in REQUIRED_STAGE3F_AUTHOR_QUESTIONS:
                if question not in questions_text:
                    errors.append(f"Stage 3F author questions missing: {question}")

        if re.search(
            r"\b(?:impact factor|citescore|acceptance rate|acceptance-rate|quartile)\s*[:=]\s*\d",
            stage3c_text,
            flags=re.IGNORECASE,
        ):
            errors.append("Stage 3C package must not assert numeric journal metrics.")
        if re.search(r"\b(?:USD|CHF|EUR|INR)\s*[\d,]+", stage3c_text):
            errors.append("Stage 3C package must not add unreverified APC amounts.")
        if "orcid.org/" in stage3c_text.lower() or re.search(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", stage3c_text):
            errors.append("Stage 3C package must not invent ORCID identifiers.")

    if all(path.exists() for path in STAGE3D_REQUIRED_FILES):
        stage3d_text = "\n".join(path.read_text(encoding="utf-8") for path in STAGE3D_REQUIRED_FILES)
        bspc_text = (PAPER / "BSPC_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
        for phrase in REQUIRED_BSPC_PHRASES:
            if phrase not in bspc_text:
                errors.append(f"BSPC manuscript draft missing phrase: {phrase}")
        if "[REFERENCE NEEDED:" in bspc_text:
            errors.append("BSPC manuscript draft must not contain unresolved reference-needed markers.")
        for metric in REQUIRED_BSPC_METRICS:
            if metric not in bspc_text:
                errors.append(f"BSPC manuscript draft appears to have lost required metric/text: {metric}")

        abstract_match = re.search(r"## Abstract\s+(.*?)\s+## Keywords", bspc_text, flags=re.DOTALL)
        if not abstract_match:
            errors.append("BSPC manuscript draft missing abstract/keywords boundary.")
        else:
            abstract_words = re.findall(r"\b\S+\b", abstract_match.group(1))
            if len(abstract_words) > 250:
                errors.append(f"BSPC abstract exceeds 250 words: {len(abstract_words)}")

        keywords_match = re.search(r"## Keywords\s+(.*?)\s+## 1\. Introduction", bspc_text, flags=re.DOTALL)
        if not keywords_match:
            errors.append("BSPC manuscript draft missing keywords/introduction boundary.")
        else:
            keywords = [item.strip() for item in keywords_match.group(1).split(";") if item.strip()]
            if not 1 <= len(keywords) <= 7:
                errors.append(f"BSPC keywords must contain 1-7 entries; found {len(keywords)}")

        declarations_text = (PAPER / "BSPC_DECLARATIONS_DRAFT.md").read_text(encoding="utf-8")
        for section in BSPC_DECLARATION_SECTIONS:
            if section not in declarations_text:
                errors.append(f"BSPC declarations draft missing section: {section}")
        if "No funding was received for this study." in declarations_text:
            for phrase in REQUIRED_STAGE3E_CONFIRMED_DECLARATIONS:
                if phrase not in declarations_text:
                    errors.append(f"BSPC declarations draft missing Stage 3E declaration: {phrase}")
            for phrase in [
                "This study used secondary datasets obtained under their original access terms.",
                "The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets.",
                "ChatGPT and Codex were used to assist with code organization",
                "Final portal metadata checks for repository visibility, license and exact URL remain pending.",
                "Dr. Vikram Singh: Supervision.",
                "A preprint is planned.",
                "The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
            ]:
                if phrase not in declarations_text:
                    errors.append(f"BSPC declarations draft missing Stage 3H declaration phrase: {phrase}")
            if "No AI" in declarations_text or "Use of AI-assisted writing tools | No" in declarations_text:
                errors.append("BSPC declarations draft must not finalize AI-assisted writing disclosure as No.")
        else:
            if declarations_text.count("[AUTHOR CONFIRMATION REQUIRED]") < 10:
                errors.append("BSPC declarations draft must keep author-confirmation markers explicit.")
            for pattern, label in [
                (r"## Funding\s+\n\s*(?:No|None|N/A)\b", "funding"),
                (r"## Competing Interests\s+\n\s*(?:No|None|Nothing|N/A)\b", "competing interests"),
                (r"## Ethics Approval\s+\n\s*(?:No|None|Not required|N/A)\b", "ethics"),
            ]:
                if re.search(pattern, declarations_text, flags=re.IGNORECASE):
                    errors.append(f"BSPC declarations draft must not finalize {label} wording.")

        if "orcid.org/" in stage3d_text.lower() or re.search(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", stage3d_text):
            errors.append("Stage 3D package must not invent ORCID identifiers.")
        if "Submission status: submitted" in stage3d_text or "ready_for_submission" in stage3d_text:
            errors.append("Stage 3D package must not mark manuscript as submitted or submission-ready.")
        if re.search(
            r"\b(?:impact factor|citescore|acceptance rate|acceptance-rate|quartile)\s*[:=]\s*\d",
            stage3d_text,
            flags=re.IGNORECASE,
        ):
            errors.append("Stage 3D package must not assert journal metrics.")

        for sentence in sentences(bspc_text):
            normalized = sentence.lower()
            for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
                if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                    errors.append(f"BSPC draft potential clinical overclaim without limitation: {sentence}")

    if all(path.exists() for path in STAGE3E_REQUIRED_FILES):
        stage3e_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in STAGE3E_REQUIRED_FILES
            if path.suffix.lower() not in {".png"}
        )
        bspc_text = (PAPER / "BSPC_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
        captions_text = (PAPER / "BSPC_FIGURE_CAPTIONS.md").read_text(encoding="utf-8")

        if "figures_generated_awaiting_author_review" not in stage3e_text:
            errors.append("Stage 3E figure package must mark figures as awaiting author review.")
        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
            if label not in captions_text:
                errors.append(f"BSPC figure captions missing {label}")
            if f"[{label} about here.]" not in bspc_text:
                errors.append(f"BSPC manuscript missing callout for {label}")
        for label in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Table 6"]:
            if f"[{label} about here.]" not in bspc_text:
                errors.append(f"BSPC manuscript missing callout for {label}")
        if captions_text.count("Prohibited interpretation") < 5:
            errors.append("BSPC figure captions must include prohibited-interpretation cautions for all five figures.")
        if "not fabricated" not in captions_text:
            errors.append("BSPC figure captions must explicitly avoid fabricating unavailable MMD^2.")

        figure_rows = csv_rows(PAPER / "BSPC_FIGURE_DATA_AUDIT.csv", REQUIRED_FIGURE_AUDIT_COLUMNS)
        if len(figure_rows) < 30:
            errors.append(f"BSPC figure audit has too few rows: {len(figure_rows)}")
        figures_in_audit = {row["figure"] for row in figure_rows}
        if figures_in_audit != {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"}:
            errors.append(f"BSPC figure audit has unexpected figure set: {sorted(figures_in_audit)}")
        invalid_statuses = sorted(
            {row["match status"] for row in figure_rows}.difference({"match", "not_applicable"})
        )
        if invalid_statuses:
            errors.append(f"BSPC figure audit has invalid match statuses: {invalid_statuses}")
        mismatches = [
            row
            for row in figure_rows
            if row["match status"] == "match" and row["plotted value"] != row["official value"]
        ]
        if mismatches:
            errors.append(f"BSPC figure audit contains mismatched plotted values: {mismatches[:3]}")
        audit_values = {(row["figure"], row["metric"], row["plotted value"]) for row in figure_rows}
        missing_audit_values = sorted(REQUIRED_FIGURE_AUDIT_VALUES.difference(audit_values))
        if missing_audit_values:
            errors.append(f"BSPC figure audit missing required values: {missing_audit_values[:5]}")
        for row in figure_rows:
            source_path = ROOT / row["source file"]
            if not source_path.exists():
                errors.append(f"BSPC figure audit source file does not exist: {row['source file']}")

        for path in STAGE3E_FIGURE_OUTPUTS:
            if path.suffix.lower() == ".png":
                data = path.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 10_000:
                    errors.append(f"Generated PNG appears invalid or too small: {path.relative_to(ROOT)}")
            else:
                text = path.read_text(encoding="utf-8")
                if "<svg" not in text or "</svg>" not in text:
                    errors.append(f"Generated SVG appears invalid: {path.relative_to(ROOT)}")

        for pattern in FORBIDDEN_GENERATED_ARTIFACT_PATTERNS:
            if re.search(pattern, stage3e_text, flags=re.IGNORECASE):
                errors.append(f"Stage 3E generated text contains forbidden identifier pattern: {pattern}")

        for phrase in REQUIRED_STAGE3E_UNRESOLVED_TEXT:
            if phrase not in "\n".join(
                [
                    (PAPER / "BSPC_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8"),
                    (PAPER / "BSPC_DECLARATIONS_DRAFT.md").read_text(encoding="utf-8"),
                    (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").read_text(encoding="utf-8"),
                    (PAPER / "AUTHOR_INPUT_FORM.md").read_text(encoding="utf-8"),
                ]
            ):
                errors.append(f"Stage 3E unresolved author marker missing: {phrase}")
        combined_author_text = "\n".join(
            [
                (PAPER / "BSPC_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8"),
                (PAPER / "BSPC_DECLARATIONS_DRAFT.md").read_text(encoding="utf-8"),
                (PAPER / "JOURNAL_NEUTRAL_FRONT_MATTER.md").read_text(encoding="utf-8"),
                (PAPER / "AUTHOR_INPUT_FORM.md").read_text(encoding="utf-8"),
            ]
        )
        if "orcid.org/" in combined_author_text.lower() or re.search(
            r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", combined_author_text
        ):
            errors.append("Stage 3E author files must not invent ORCID identifiers.")
        if "No AI" in combined_author_text or "Use of AI-assisted writing tools | No" in combined_author_text:
            errors.append("Stage 3E author files must not finalize AI disclosure as No.")

    if all(path.exists() for path in STAGE3F_REQUIRED_FILES):
        docx_text = (PAPER / "BSPC_DOCX_READY_MANUSCRIPT.md").read_text(encoding="utf-8")
        supplement_text = (PAPER / "BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md").read_text(encoding="utf-8")
        conversion_text = (PAPER / "BSPC_DOCX_CONVERSION_NOTES.md").read_text(encoding="utf-8")
        review_text = (PAPER / "BSPC_FIGURE_VISUAL_REVIEW.md").read_text(encoding="utf-8")
        cover_text = (PAPER / "BSPC_COVER_LETTER_DRAFT.md").read_text(encoding="utf-8")
        table_plan_text = (PAPER / "BSPC_TABLE_FORMATTING_PLAN.md").read_text(encoding="utf-8")

        if "DOCX-ready package status: `bspc_final_review_docx_ready_portal_checks_pending`" not in docx_text:
            errors.append("DOCX-ready manuscript missing Stage 3H package status.")
        for placeholder in REQUIRED_STAGE3F_DOCX_PLACEHOLDERS:
            if placeholder not in docx_text:
                errors.append(f"DOCX-ready manuscript missing unresolved placeholder: {placeholder}")
        for phrase in [
            "## Table Captions And Placement Notes",
            "## Figure Captions",
            "## References",
            "[BIBLIOGRAPHY PLACEHOLDER: generate from `paper/references.bib`",
            "## Supplementary Material Note",
        ]:
            if phrase not in docx_text:
                errors.append(f"DOCX-ready manuscript missing section/phrase: {phrase}")
        for label in ["[Figure 1 about here.]", "[Figure 2 about here.]", "[Figure 3 about here.]", "[Figure 4 about here.]", "[Figure 5 about here.]"]:
            if label not in docx_text:
                errors.append(f"DOCX-ready manuscript missing figure placeholder: {label}")
        for label in ["[Table 1 about here.]", "[Table 2 about here.]", "[Table 3 about here.]", "[Table 4 about here.]", "[Table 5 about here.]", "[Table 6 about here.]"]:
            if label not in docx_text:
                errors.append(f"DOCX-ready manuscript missing table placeholder: {label}")
        for metric in REQUIRED_BSPC_METRICS:
            if metric not in docx_text:
                errors.append(f"DOCX-ready manuscript missing metric/text: {metric}")

        if review_text.count("ready_for_author_review") < 6:
            errors.append("Figure visual review must mark all figures ready for author review.")
        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
            if label not in review_text:
                errors.append(f"Figure visual review missing {label}")
        if "No figure revisions were required in Stage 3F" not in review_text:
            errors.append("Figure visual review must state whether revisions were required.")

        for label in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
            if f"| {label}" not in review_text:
                errors.append(f"Figure visual review summary missing row for {label}")
        if "main manuscript" not in table_plan_text or table_plan_text.count("main manuscript") < 6:
            errors.append("BSPC table plan must classify all six tables for the main manuscript.")

        for phrase in [
            "Status: `draft_prepared_awaiting_author_review`",
            "Supplementary Table S1",
            "Layer-wise and aggregation ablations",
            "Calibration and threshold details",
            "Domain-shift details",
            "Cross-task and classifier-boundary details",
            "No supplementary figures were generated in Stage 3F",
        ]:
            if phrase not in supplement_text:
                errors.append(f"Supplementary draft missing phrase: {phrase}")
        for prohibited in ["participant_id", "subject_id", "audio path:", "embeddings file", "fitted model file"]:
            if prohibited in supplement_text.lower():
                errors.append(f"Supplementary draft may contain prohibited artifact detail: {prohibited}")

        for phrase in [
            "Status: `docx_pending_tooling_or_manual_conversion`",
            "pandoc",
            "Result: unavailable on PATH",
            "DOCX generated: no",
            "PDF generated: no",
        ]:
            if phrase not in conversion_text:
                errors.append(f"DOCX conversion notes missing phrase: {phrase}")

        for phrase in [
            "Status: `draft_final_details_applied_author_final_review_required`",
            "Email: viks@nitkkr.ac.in. ORCID: not available.",
            "[AUTHOR FINAL CONFIRMATION REQUIRED: the manuscript is not under consideration elsewhere.]",
            "The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
            "A preprint is planned.",
            "The work is research-only.",
            "It does not claim clinical validation",
        ]:
            if phrase not in cover_text:
                errors.append(f"Cover letter draft missing phrase: {phrase}")
        if "No AI" in docx_text or "Use of AI-assisted writing tools | No" in docx_text:
            errors.append("DOCX-ready manuscript must not finalize AI disclosure as No.")
        if "Submission status: submitted" in docx_text or "ready_for_submission" in docx_text:
            errors.append("DOCX-ready manuscript must not mark the work as submitted or submission-ready.")

    if VOICE_PHYSIOLOGY_AUDIT_FILE.exists():
        for path in [PAPER / "BSPC_MANUSCRIPT_DRAFT.md", PAPER / "BSPC_DOCX_READY_MANUSCRIPT.md"]:
            text = path.read_text(encoding="utf-8")
            section = voice_physiology_section(text)
            if not section:
                errors.append(f"{path.relative_to(ROOT)} missing voice physiology subsection.")
                continue
            word_count = len(re.findall(r"\b\S+\b", section))
            if not 250 <= word_count <= 400:
                errors.append(f"{path.relative_to(ROOT)} voice physiology subsection must be 250-400 words; found {word_count}")
            for phrase in VOICE_PHYSIOLOGY_REQUIRED_PHRASES:
                if phrase not in section:
                    errors.append(f"{path.relative_to(ROOT)} voice physiology subsection missing phrase: {phrase}")
            section_citations = citations_in_text(section)
            missing_citations = sorted(VOICE_PHYSIOLOGY_REQUIRED_CITATIONS.difference(section_citations))
            if missing_citations:
                errors.append(f"{path.relative_to(ROOT)} voice physiology subsection missing citations: {missing_citations}")
            unknown_citations = sorted(section_citations.difference(bib_keys))
            if unknown_citations:
                errors.append(f"{path.relative_to(ROOT)} voice physiology subsection has unknown citations: {unknown_citations}")
            section_lower = section.lower()
            for phrase in VOICE_PHYSIOLOGY_FORBIDDEN_PHRASES:
                if phrase.lower() in section_lower:
                    errors.append(f"{path.relative_to(ROOT)} voice physiology subsection contains forbidden phrase: {phrase}")
            for sentence in sentences(section):
                normalized = sentence.lower()
                for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
                    if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                        errors.append(f"{path.relative_to(ROOT)} voice physiology potential clinical overclaim: {sentence}")

        audit_text = VOICE_PHYSIOLOGY_AUDIT_FILE.read_text(encoding="utf-8")
        for phrase in [
            "Source Material Used",
            "Manuscript Location",
            "Inserted subsection word count: 261 words.",
            "No new bibliography entries were added.",
            "Claims Deliberately Excluded",
            "No new experiment result, participant count, confidence interval, registry record or frozen metric was added.",
            "No claim that speech can diagnose depression was introduced.",
            "No claim that the system is clinically validated was introduced.",
        ]:
            if phrase not in audit_text:
                errors.append(f"Voice physiology insert audit missing phrase: {phrase}")

    if IEEE_TCSS_DIR.exists():
        for path in [*STAGE4A_IEEE_TCSS_REQUIRED_FILES, *STAGE4A_IEEE_TCSS_TABLE_FILES, *STAGE4A_IEEE_TCSS_FIGURE_FILES]:
            if not path.exists():
                errors.append(f"Missing IEEE TCSS Stage 4A file: {path.relative_to(ROOT)}")

        main_tex_path = IEEE_TCSS_DIR / "main.tex"
        if main_tex_path.exists():
            main_tex = main_tex_path.read_text(encoding="utf-8")
            for phrase in [
                r"\documentclass[lettersize,journal]{IEEEtran}",
                r"\usepackage{amsmath,amsfonts}",
                r"\usepackage{algorithmic}",
                r"\usepackage{algorithm}",
                r"\usepackage{array}",
                r"\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}",
                r"\usepackage{textcomp}",
                r"\usepackage{stfloats}",
                r"\usepackage{url}",
                r"\usepackage{verbatim}",
                r"\usepackage{graphicx}",
                r"\usepackage{cite}",
                r"\usepackage{booktabs}",
                r"\usepackage{multirow}",
                r"\markboth{IEEE Transactions on Computational Social Systems",
                r"\begin{abstract}",
                r"\begin{IEEEkeywords}",
                r"\section{Introduction}",
                r"\section{Related Work}",
                r"\section{Datasets and Experimental Design}",
                r"\section{Methods}",
                r"\section{Results}",
                r"\section{Discussion}",
                r"\section{Ethical and Clinical Considerations}",
                r"\section{Conclusion}",
                r"\bibliographystyle{IEEEtran}",
                r"\bibliography{bib/references}",
                r"\input{declarations}",
                "[AUTHOR CONFIRMATION REQUIRED: IEEE biography]",
                "No IEEE membership status is claimed.",
                "ORCID identifiers are not available for either author.",
            ]:
                if phrase not in main_tex:
                    errors.append(f"IEEE TCSS main.tex missing required phrase: {phrase}")
            for metric in STAGE4A_IEEE_TCSS_METRICS:
                if metric not in main_tex:
                    errors.append(f"IEEE TCSS main.tex missing preserved metric/text: {metric}")
            if "[@" in main_tex or "ZXQ" in main_tex:
                errors.append("IEEE TCSS main.tex contains unresolved markdown citation or conversion placeholder.")
            copied_bib = IEEE_TCSS_DIR / "bib" / "references.bib"
            copied_bib_keys = (
                set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", copied_bib.read_text(encoding="utf-8")))
                if copied_bib.exists()
                else set()
            )
            tex_citations = set()
            for match in re.finditer(r"\\cite\{([^}]+)\}", main_tex):
                tex_citations.update(key.strip() for key in match.group(1).split(",") if key.strip())
            unknown_tex_citations = sorted(tex_citations.difference(copied_bib_keys))
            if unknown_tex_citations:
                errors.append(f"IEEE TCSS main.tex contains unknown BibTeX keys: {unknown_tex_citations}")
            if len(tex_citations) < 28:
                errors.append(f"IEEE TCSS main.tex should preserve at least 28 citation keys; found {len(tex_citations)}")
            figure_includes = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", main_tex)
            svg_includes = [path for path in figure_includes if path.lower().endswith(".svg")]
            if svg_includes:
                errors.append(f"IEEE TCSS main.tex must not include SVG files in pdfLaTeX path: {svg_includes}")
            missing_figure_includes = [path for path in figure_includes if not (IEEE_TCSS_DIR / path).exists()]
            if missing_figure_includes:
                errors.append(f"IEEE TCSS main.tex has missing figure includes: {missing_figure_includes}")
            table_inputs = re.findall(r"\\input\{(tables/[^}]+)\}", main_tex)
            missing_table_inputs = [path for path in table_inputs if not (IEEE_TCSS_DIR / path).exists()]
            if missing_table_inputs:
                errors.append(f"IEEE TCSS main.tex has missing table inputs: {missing_table_inputs}")
            if r"\bibliography{bib/references}" in main_tex and not (IEEE_TCSS_DIR / "bib" / "references.bib").exists():
                errors.append("IEEE TCSS main.tex bibliography command points to missing bib/references.bib.")
            for phrase in [
                "speaker-specific voice characteristics",
                "voiceprint information",
                "single elicitation condition",
                "cross-corpus generalization remains limited",
                "representation choice, participant-level aggregation, timing descriptors",
                "should not be interpreted as having disentangled depression markers from biometric voice characteristics",
                "Dedicated speaker-disentanglement or adversarial-invariance experiments would be required",
            ]:
                if phrase not in main_tex:
                    errors.append(f"IEEE TCSS main.tex missing Stage 4F research-gap phrase: {phrase}")
            for prohibited_gap_overclaim in [
                "fully disentangles",
                "fully disentangled",
                "completely disentangles",
                "clinically validated",
                "deployment-ready",
                "proves",
            ]:
                if contains_prohibited_wording(main_tex.lower(), prohibited_gap_overclaim):
                    errors.append(f"IEEE TCSS main.tex contains prohibited research-gap overclaim: {prohibited_gap_overclaim}")
            for prohibited in [
                "orcid.org/",
                "Member, IEEE",
                "Senior Member, IEEE",
                "Submission status: submitted",
                "ready_for_submission",
                "clinically validated system",
                "deployment-ready system",
                "diagnoses depression",
                "PHQ-8 severity",
                "privacy-preserving assessment",
            ]:
                if prohibited.lower() in main_tex.lower():
                    errors.append(f"IEEE TCSS main.tex contains prohibited wording: {prohibited}")
            for sentence in sentences(main_tex):
                normalized = sentence.lower()
                for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
                    if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                        errors.append(f"IEEE TCSS potential clinical overclaim without limitation: {sentence}")

        declarations_path = IEEE_TCSS_DIR / "declarations.tex"
        if declarations_path.exists():
            declarations_text = declarations_path.read_text(encoding="utf-8")
            for phrase in [
                "No funding was received for this study.",
                "The authors declare no competing interests.",
                "This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.",
                "This study used secondary datasets obtained under their original access terms. Consent for the original data collection was managed by the respective dataset providers. No new human participants were recruited in this study.",
                "The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets.",
                "ChatGPT and Codex were used to assist with code organization, manuscript drafting support, editing, and workflow planning.",
                "A preprint is planned.",
            ]:
                if phrase not in declarations_text:
                    errors.append(f"IEEE TCSS declarations missing phrase: {phrase}")

        package_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in STAGE4A_IEEE_TCSS_REQUIRED_FILES
            if path.exists() and path.suffix in {".md", ".tex", ".ps1", ".py"}
        )
        for phrase in [
            "Official sources only were used.",
            "not_verified",
            "Fit status: `moderate_with_reframing_needed`",
            "Missing keys: none",
            "[AUTHOR FINAL CONFIRMATION REQUIRED: This manuscript is not under consideration elsewhere.]",
            "No IEEE submission or upload was performed.",
            "not asserted to be the original uploaded template",
            "No title change was applied automatically in `main.tex`.",
            "Real uploaded template found: no.",
            "Sanity status: `passed_with_external_tex_toolchain_unavailable`.",
            "Likely unescaped underscores | passed",
            "All `\\includegraphics` paths in `main.tex` were checked.",
            "All `\\input{tables/...}` paths in `main.tex` were checked.",
            "No SVG files are included in `main.tex`; PNG files are used for compilation.",
            "No table content, metrics or interpretation categories were changed in Stage 4B.",
            "No unused critical manuscript references were dropped",
            "Do not use online LaTeX compilers with restricted manuscript data unless the authors have confirmed that upload is allowed.",
            "Stage: 4E Overleaf package refresh after research-question alignment",
            "Fresh compile-check PDF path",
            "Unresolved citations | none in final pass",
            "ZIP path: `paper/ieee_tcss/overleaf_upload/ieee_tcss_overleaf_package.zip`",
            "Entry count: 15",
            "`main.tex` contains RQ1 | yes",
            "Do not treat the Overleaf PDF as submitted.",
            "All four RQs visible in Introduction",
            "Manuscript not under consideration elsewhere confirmed",
            "Stage: 4F research-gap framing",
            "Speaker-identity and voiceprint confounding were not explicitly framed as modeling risks in the Introduction.",
            "Research gap addressed",
            "Speaker/voiceprint confounding risk and representation validity",
            "no speaker-disentanglement experiment was performed",
        ]:
            if phrase not in package_text:
                errors.append(f"IEEE TCSS package missing audit/checklist phrase: {phrase}")
        build_log_text = (IEEE_TCSS_DIR / "logs" / "LATEX_BUILD_LOG.md").read_text(encoding="utf-8")
        missing_toolchain_phrases = [
            "Status: not_run_missing_tex_toolchain",
            "pdflatex available: False",
            "bibtex available: False",
            "PDF generated: False",
            "No PDF was generated because a complete TeX toolchain was not available on PATH.",
        ]
        successful_build_phrases = [
            "Status: succeeded",
            "PDF generated: True",
            "pdflatex:",
            "bibtex:",
            "LaTeX/package warnings present; inspect build/*.log.",
        ]
        if not (
            all(phrase in build_log_text for phrase in missing_toolchain_phrases)
            or all(phrase in build_log_text for phrase in successful_build_phrases)
        ):
            errors.append("IEEE TCSS build log must record either missing TeX tooling or a successful local PDF build.")
        status_text = (PAPER / "PAPER_WRITING_STATUS.md").read_text(encoding="utf-8")
        for phrase in [
            "Stage 4A status: IEEE Transactions on Computational Social Systems LaTeX preparation package created",
            "Stage 4A branch: `paper/ieee-tcss-latex-v1`.",
            "No IEEE upload, portal account creation, submission, final PDF generation, new experiments, registry modification, frozen-metric change, ORCID invention, IEEE membership invention, clinical validation claim or deployment-readiness claim was performed.",
            "Stage 4B branch: `paper/ieee-tcss-template-compile-v1`.",
            "Build result: no PDF generated because `pdflatex` and `bibtex` are not on PATH",
            "Stage 4D branch: `paper/ieee-tcss-research-questions-v1`.",
            "Four research questions were inserted near the end of the Introduction",
            "Build result: local LaTeX build succeeded and generated `paper/ieee_tcss/build/main.pdf` for review only; no IEEE upload or submission was performed.",
            "Stage 4E branch: `paper/ieee-tcss-research-questions-v1`.",
            "RQ-aligned Overleaf ZIP regenerated at `paper/ieee_tcss/overleaf_upload/ieee_tcss_overleaf_package.zip`.",
            "Manuscript-not-under-consideration confirmation remains pending.",
            "Stage 4F branch: `paper/ieee-tcss-research-gaps-v1`.",
            "Research-gap statements inserted in the Introduction",
            "speaker-disentanglement claims were changed or invented",
        ]:
            if phrase not in status_text:
                errors.append(f"Paper writing status missing Stage 4A phrase: {phrase}")
        checklist_text = (PAPER / "SUBMISSION_READINESS_CHECKLIST.md").read_text(encoding="utf-8")
        for phrase in [
            "IEEE TCSS LaTeX package | complete",
            "IEEE TCSS portal upload | not_started",
            "IEEE TCSS template upload | pending_portal",
            "IEEE TCSS template alignment audit | complete",
            "IEEE TCSS compile readiness | complete",
            "IEEE TCSS research questions | complete",
            "IEEE TCSS RQ evidence map | complete",
            "IEEE TCSS RQ-aligned Overleaf ZIP | complete",
            "IEEE TCSS PDF visual review | needs_author_input",
            "IEEE TCSS not-under-consideration confirmation | needs_author_input",
            "IEEE TCSS final portal submission | not_started",
            "IEEE TCSS research-gap framing | complete",
            "IEEE TCSS speaker-disentanglement claim guard | complete",
        ]:
            if phrase not in checklist_text:
                errors.append(f"Submission readiness checklist missing Stage 4A phrase: {phrase}")
        questions_text = (PAPER / "AUTHOR_FINAL_QUESTIONS.md").read_text(encoding="utf-8")
        for phrase in [
            "Stage 4A IEEE TCSS Additional Author/Portal Checks",
            "Confirm whether the current title should remain unchanged for IEEE TCSS",
            "Confirm the IEEE TCSS cover-letter statement",
            "no IEEE upload, portal account creation or submission should occur",
        ]:
            if phrase not in questions_text:
                errors.append(f"Author final questions missing Stage 4A phrase: {phrase}")

    if all(path.exists() for path in STAGE3G_REQUIRED_FILES):
        package_readme = (PAPER / "submission_review_package" / "README.md").read_text(encoding="utf-8")
        packet_checklist = (
            PAPER / "submission_review_package" / "AUTHOR_REVIEW_PACKET_CHECKLIST.md"
        ).read_text(encoding="utf-8")
        generation_log = (
            PAPER / "submission_review_package" / "logs" / "BSPC_DOCX_GENERATION_LOG.md"
        ).read_text(encoding="utf-8")
        generator_text = (PAPER / "scripts" / "create_bspc_review_docx.py").read_text(encoding="utf-8")
        manuscript_docx = PAPER / "submission_review_package" / "docx" / "BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx"
        supplement_docx = PAPER / "submission_review_package" / "docx" / "BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx"
        manuscript_final_docx = PAPER / "submission_review_package" / "docx" / "BSPC_MANUSCRIPT_FINAL_REVIEW.docx"
        supplement_final_docx = PAPER / "submission_review_package" / "docx" / "BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx"
        manuscript_docx_text, manuscript_media_count, manuscript_docx_valid = docx_text_and_media_count(manuscript_docx)
        supplement_docx_text, supplement_media_count, supplement_docx_valid = docx_text_and_media_count(supplement_docx)
        manuscript_final_text, manuscript_final_media_count, manuscript_final_valid = docx_text_and_media_count(manuscript_final_docx)
        supplement_final_text, supplement_final_media_count, supplement_final_valid = docx_text_and_media_count(supplement_final_docx)
        review_package_text = "\n".join([package_readme, packet_checklist, generation_log])
        docx_combined_text = "\n".join([manuscript_docx_text, supplement_docx_text, manuscript_final_text, supplement_final_text])
        combined_stage3g_text = "\n".join([review_package_text, docx_combined_text, generator_text])

        for phrase in [
            "Status: `bspc_final_review_package_ready_for_submission_portal_checks`",
            "for final author and portal checks only",
            "not submitted files",
            "no upload to Elsevier or BSPC has been performed",
            "final portal metadata entry, final portal upload, final BSPC submission-system checks",
            "Final CRediT statement is applied",
            "Funding, competing interests, ethics, consent, acknowledgements, AI disclosure, preprint status and BSPC subscription route wording are applied.",
            "The regenerated manuscript DOCX includes the physiology motivation subsection and five approved main figure PNG review copies.",
            "The experiment-rationale documentation package remains available in `paper/`.",
        ]:
            if phrase not in package_readme:
                errors.append(f"Stage 3G README missing phrase: {phrase}")

        allowed_review_statuses = {
            "included",
            "complete",
            "needs_author_input",
            "needs_visual_review",
            "pending_portal",
            "not_started",
            "not_applicable",
        }
        for line in packet_checklist.splitlines():
            separator = line.replace("|", "").replace(" ", "").strip()
            if not line.startswith("| ") or line.startswith("| Item ") or set(separator) <= {"-", ":"}:
                continue
            cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] not in allowed_review_statuses:
                errors.append(f"Author review packet checklist has invalid status: {cells[1]} in row {cells[0]}")
        for phrase in [
            "Generated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`",
            "Generated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`",
            "Generated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx`",
            "Generated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx`",
            "DOCX generation succeeded with `python-docx`",
            "Stage 3H final declaration application | complete",
            "Physiology subsection in manuscript DOCX | complete",
            "Experiment-rationale documentation | included",
            "Author emails | complete",
            "CRediT statement | complete",
            "Consent wording | complete",
            "BSPC route decision | complete",
            "Submission upload | pending_portal",
            "Submission confirmation | not_started",
        ]:
            if phrase not in packet_checklist:
                errors.append(f"Author review packet checklist missing phrase: {phrase}")

        for phrase in [
            "Date generated:",
            "Branch: `paper/bspc-final-declarations-v1`",
            "Stage 3H branch: `paper/bspc-final-declarations-v1`",
            "Source branch: `paper/bspc-final-review-package-v1`.",
            "Final author/declaration details applied: yes.",
            "Author approvals applied: figures approved, manuscript DOCX approved and supplementary DOCX approved.",
            "Dr. Vikram Singh email applied: `viks@nitkkr.ac.in`.",
            "ORCID status applied: not available for both authors; no ORCID identifier was invented.",
            "CRediT statement applied: Mohammad Ilyas full supplied role list; Dr. Vikram Singh: Supervision.",
            "Consent, acknowledgements and AI disclosure applied from Stage 3H author input.",
            "Preprint status applied: a preprint is planned.",
            "BSPC route applied: subscription route.",
            "Consolidation branch retained in history: `paper/bspc-final-review-package-v1`",
            "Source branch confirmed: `paper/bspc-review-docx-v1`",
            "Source branch confirmed: `paper/experiment-rationale-summary-v1`",
            "Source branch confirmed: `paper/voice-physiology-motivation-v1`",
            "Merge status: `paper/bspc-review-docx-v1` was already an ancestor of the consolidation branch; no merge conflicts occurred.",
            "Physiology subsection included: yes, `2.2 Physiological motivation for speech-based depression analysis`.",
            "Experiment-rationale documentation included: yes.",
            "Review DOCX files regenerated after Stage 3H declarations: yes.",
            "Final-review DOCX copies generated: yes.",
            "Design preset: `standard_business_brief`",
            "paper/BSPC_DOCX_READY_MANUSCRIPT.md",
            "paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md",
            "paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx",
            "paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx",
            "paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx",
            "paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx",
            "python-docx available: yes",
            "Figures inserted: yes",
            "Main figures inserted: 5 of 5",
            "LibreOffice/soffice available for render QA: no",
            "Visual render QA status: not_completed_soffice_unavailable",
            "page-image render QA via `render_docx.py` could not be completed",
            "Final PDF generated: no",
            "Submission/upload performed: no",
        ]:
            if phrase not in generation_log:
                errors.append(f"Stage 3G generation log missing phrase: {phrase}")
        for placeholder in REQUIRED_STAGE3F_DOCX_PLACEHOLDERS:
            if f"yes: `{placeholder}`" not in generation_log:
                errors.append(f"Stage 3H generation log missing retained portal marker: {placeholder}")

        if not manuscript_docx_valid:
            errors.append("Stage 3G manuscript review DOCX is not a readable DOCX package.")
        if not supplement_docx_valid:
            errors.append("Stage 3G supplementary review DOCX is not a readable DOCX package.")
        if not manuscript_final_valid:
            errors.append("Stage 3H manuscript final-review DOCX is not a readable DOCX package.")
        if not supplement_final_valid:
            errors.append("Stage 3H supplementary final-review DOCX is not a readable DOCX package.")
        if manuscript_docx.stat().st_size < 100_000:
            errors.append("Stage 3G manuscript review DOCX is unexpectedly small.")
        if supplement_docx.stat().st_size < 20_000:
            errors.append("Stage 3G supplementary review DOCX is unexpectedly small.")
        if manuscript_final_docx.stat().st_size < 100_000:
            errors.append("Stage 3H manuscript final-review DOCX is unexpectedly small.")
        if supplement_final_docx.stat().st_size < 20_000:
            errors.append("Stage 3H supplementary final-review DOCX is unexpectedly small.")
        if manuscript_media_count < 5:
            errors.append(f"Stage 3G manuscript DOCX should include five figure images; found {manuscript_media_count}.")
        if supplement_media_count != 0:
            errors.append(f"Stage 3G supplement DOCX should not include figure images; found {supplement_media_count}.")
        if manuscript_final_media_count < 5:
            errors.append(f"Stage 3H manuscript final-review DOCX should include five figure images; found {manuscript_final_media_count}.")
        if supplement_final_media_count != 0:
            errors.append(f"Stage 3H supplementary final-review DOCX should not include figure images; found {supplement_final_media_count}.")

        for placeholder in REQUIRED_STAGE3F_DOCX_PLACEHOLDERS:
            if placeholder not in manuscript_docx_text:
                errors.append(f"Stage 3G manuscript DOCX missing portal marker: {placeholder}")
            if placeholder not in manuscript_final_text:
                errors.append(f"Stage 3H manuscript final-review DOCX missing portal marker: {placeholder}")
        for phrase in [
            "FINAL REVIEW COPY - NOT SUBMITTED",
            "Authors: Mohammad Ilyas; Dr. Vikram Singh.",
            "Corresponding author: Mohammad Ilyas",
            "Author emails: Mohammad Ilyas, mohd.passion.cse@mitmoradabad.edu.in; Dr. Vikram Singh, viks@nitkkr.ac.in.",
            "Dr. Vikram Singh: Supervision.",
            "A preprint is planned.",
            "The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
            "Affiliations: Mohammad Ilyas",
            "Abstract",
            "Keywords",
            "Highlights",
            "[Figure 1 about here.]",
            "[Figure 5 about here.]",
            "[Table 1 about here.]",
            "[Table 6 about here.]",
            "Figure 1. Study design/protocol overview",
            "Figure 5. Domain-shift and classifier-boundary summary",
            "Declaration of Competing Interest",
            "Data and Code Availability",
            "Code Availability",
            "BIBLIOGRAPHY PLACEHOLDER",
            "Physiological motivation for speech-based depression analysis",
        ]:
            if phrase not in manuscript_docx_text:
                errors.append(f"Stage 3G manuscript DOCX missing phrase: {phrase}")
        for metric in REQUIRED_BSPC_METRICS:
            if metric not in manuscript_docx_text:
                errors.append(f"Stage 3G manuscript DOCX missing metric/text: {metric}")
        for phrase in [
            "BSPC Supplementary Material Draft",
            "Status: draft_prepared_awaiting_author_review",
            "Supplementary Table S1",
            "Supplementary Table S2",
            "Supplementary Table S3",
            "No supplementary figures were generated in Stage 3F",
            "raw participant-level predictions",
            "fitted models",
        ]:
            if phrase not in supplement_docx_text:
                errors.append(f"Stage 3G supplementary DOCX missing phrase: {phrase}")

        for path in [
            PAPER / "submission_review_package" / "docx" / "BSPC_MANUSCRIPT_AUTHOR_REVIEW_FALLBACK.txt",
            PAPER / "submission_review_package" / "docx" / "BSPC_SUPPLEMENTARY_AUTHOR_REVIEW_FALLBACK.txt",
        ]:
            if path.exists():
                errors.append(f"Stage 3G fallback file should not exist when DOCX generation succeeded: {path.relative_to(ROOT)}")

        if "Document(" not in generator_text or "create_fallback" not in generator_text:
            errors.append("Stage 3G DOCX generator must use python-docx with fallback support.")
        if "pandoc" in generator_text.lower():
            errors.append("Stage 3G DOCX generator must not use Pandoc.")
        if "pip install" in generator_text.lower():
            errors.append("Stage 3G DOCX generator must not install packages.")

        if "Submission status: submitted" in combined_stage3g_text:
            errors.append("Stage 3G review package must not mark the manuscript submitted or submission-ready.")
        if "No AI" in combined_stage3g_text or "Use of AI-assisted writing tools | No" in combined_stage3g_text:
            errors.append("Stage 3G review package must not finalize AI disclosure as No.")
        if "orcid.org/" in combined_stage3g_text.lower() or re.search(
            r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", combined_stage3g_text
        ):
            errors.append("Stage 3G review package must not invent ORCID identifiers.")
        allowed_emails = {"mohd.passion.cse@mitmoradabad.edu.in", "viks@nitkkr.ac.in"}
        email_scan_text = combined_stage3g_text
        for email in allowed_emails:
            email_scan_text = email_scan_text.replace(email, "")
        unexpected_emails = set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email_scan_text))
        if unexpected_emails:
            errors.append(f"Stage 3H review package contains unexpected email address(es): {sorted(unexpected_emails)}")
        for sentence in sentences(docx_combined_text):
            normalized = sentence.lower()
            for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
                if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                    errors.append(f"Stage 3G DOCX potential clinical overclaim without limitation: {sentence}")

    if all(path.exists() for path in EXPERIMENT_RATIONALE_REQUIRED_FILES):
        rationale_summary = (PAPER / "STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md").read_text(encoding="utf-8")
        reviewer_explanation = (PAPER / "EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md").read_text(encoding="utf-8")
        manuscript_insert = (PAPER / "EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md").read_text(encoding="utf-8")
        presentation_notes = (PAPER / "EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md").read_text(encoding="utf-8")
        dependency_flow = (PAPER / "EXPERIMENT_DEPENDENCY_FLOW.md").read_text(encoding="utf-8")
        scope_boundaries = (PAPER / "EXPERIMENT_SCOPE_BOUNDARIES.md").read_text(encoding="utf-8")
        rationale_package_text = "\n".join(
            [
                rationale_summary,
                reviewer_explanation,
                manuscript_insert,
                presentation_notes,
                dependency_flow,
                scope_boundaries,
            ]
        )
        rationale_rows = csv_rows(PAPER / "STUDY_EXPERIMENTS_RATIONALE_TABLE.csv", REQUIRED_RATIONALE_TABLE_COLUMNS)
        if len(rationale_rows) != 25:
            errors.append(f"Experiment-rationale table must contain 25 rows; found {len(rationale_rows)}")
        row_ids = {row["experiment_id_or_run_id"] for row in rationale_rows}
        missing_ids = sorted(REQUIRED_RATIONALE_EXPERIMENT_IDS.difference(row_ids))
        if missing_ids:
            errors.append(f"Experiment-rationale table missing required IDs: {missing_ids}")
        for row in rationale_rows:
            if not row["rationale"] or not row["key_result"] or not row["main_limitation"]:
                errors.append(f"Experiment-rationale table row lacks required narrative fields: {row['experiment_id_or_run_id']}")
            for source in [item.strip() for item in row["official_source_file"].split(";") if item.strip()]:
                if not source.startswith(("outputs/", "records/runs/", "paper/", "docs/")):
                    errors.append(f"Experiment-rationale table source not in approved project source set: {source}")

        for phrase in [
            "# Study Experiments and Rationale Summary",
            "## Purpose of This Document",
            "## Overall Experimental Logic",
            "## Phase 1: DAIC-WOZ Methodological Development",
            "### Phase 1A: Baseline Participant-Level Depression Classification",
            "### Phase 1B: WavLM Layer-Wise Analysis",
            "### Phase 1C: Aggregation Strategy Analysis",
            "### Phase 1D: Timing and Interaction Feature Analysis",
            "### Phase 1E: Late Fusion Analysis",
            "### Phase 1F: Final Frozen DAIC Evaluation",
            "## Phase 2: Androids Corpus Transfer and Task Analysis",
            "### Phase 2A: Androids Corpus Audit and Preparation",
            "### Phase 2B: DAIC-to-Android Zero-Shot Interview Transfer",
            "### Phase 2C: Domain Shift, Calibration, and Android-Trained Interview Models",
            "### Phase 2D: Android Reading-Speech Evaluation",
            "### Phase 2E: Reading-Interview Cross-Task Transfer",
            "## Final Experimental Interpretation",
            "## What the Experiments Demonstrate",
            "## What the Experiments Do Not Demonstrate",
            "## Clinical and Deployment Boundaries",
            "## How These Experiments Support the BSPC Manuscript",
        ]:
            if phrase not in rationale_summary:
                errors.append(f"Experiment-rationale summary missing required section: {phrase}")
        for metric in [
            "0.4437",
            "0.5195",
            "0.6412",
            "0.8675",
            "0.8796",
            "0.8475",
            "0.8501",
            "0.9058",
            "0.9135",
            "550 of 768",
            "0.1216",
        ]:
            if metric not in rationale_package_text:
                errors.append(f"Experiment-rationale package missing audited metric/text: {metric}")

        for phrase in [
            "## Why the Study Was Designed in Stages",
            "## Why DAIC-WOZ Was Used for Initial Method Development",
            "## Why Androids Corpus Was Used for External Task and Corpus Analysis",
            "## Why Zero-Shot Transfer Was Tested Before Adaptation",
            "## Why Calibration and Threshold Adaptation Were Separated",
            "## Why Android-Trained Models Were Reported Separately",
            "## Why Reading and Interview Speech Were Compared",
            "## Why Cross-Task Transfer Was Evaluated",
            "## Why Timing Features Were Treated Carefully",
            "## Why the Study Remains Research-Only",
        ]:
            if phrase not in reviewer_explanation:
                errors.append(f"Reviewer explanation missing section: {phrase}")

        insert_word_count = len(re.findall(r"\b\S+\b", manuscript_insert))
        if not 500 <= insert_word_count <= 900:
            errors.append(f"Manuscript rationale insert must be 500-900 words; found {insert_word_count}")
        for phrase in [
            "Phase 1 established the DAIC-WOZ methodology",
            "Phase 2A audited Androids Corpus before modeling",
            "Phase 2C separated these explanations",
            "Phase 2E then tested reading-interview transfer",
            "They do not establish clinical validation",
        ]:
            if phrase not in manuscript_insert:
                errors.append(f"Manuscript rationale insert missing phrase: {phrase}")

        for phrase in [
            "# Why Each Experiment Was Conducted",
            "## Step 1: Build a DAIC-WOZ Baseline",
            "## Step 7: Test Zero-Shot Transfer to Androids Corpus",
            "## Step 8: Diagnose Domain Shift",
            "## Step 11: Test Cross-Task Transfer",
            "## Final Takeaway",
        ]:
            if phrase not in presentation_notes:
                errors.append(f"Presentation notes missing phrase: {phrase}")
        if "```mermaid" not in dependency_flow or "Phase 2E pooled-task training and boundary summary" not in dependency_flow:
            errors.append("Experiment dependency flow must include Mermaid diagram through Phase 2E.")

        for column in [
            "experiment",
            "cross_corpus",
            "zero_shot",
            "target_labels_used",
            "classifier_refit",
            "threshold_adapted",
            "calibration_used",
            "within_corpus",
            "clinical_validation",
            "permitted_claim",
            "prohibited_claim",
        ]:
            if column not in scope_boundaries:
                errors.append(f"Experiment scope boundaries missing column: {column}")
        for phrase in [
            "Phase 2B DAIC-to-Android interview zero-shot",
            "Phase 2C Protocol B threshold-only adaptation",
            "Phase 2C Protocol C frozen-score calibration",
            "Phase 2E reading-to-interview",
            "None of these experiments is clinical validation.",
        ]:
            if phrase not in scope_boundaries:
                errors.append(f"Experiment scope boundaries missing phrase: {phrase}")

        rationale_package_lower = rationale_package_text.lower()
        for prohibited in ["proved", "diagnosed", "clinically validated", "deployment-ready", "generalizes universally", "state-of-the-art"]:
            if contains_prohibited_wording(rationale_package_lower, prohibited):
                errors.append(f"Experiment-rationale package contains prohibited wording: {prohibited}")
        for sentence in sentences(rationale_package_text):
            normalized = sentence.lower()
            for phrase in PROHIBITED_CLINICAL_OVERCLAIM_PHRASES:
                if phrase in normalized and not any(marker in normalized for marker in NEGATION_OR_LIMITING_MARKERS):
                    errors.append(f"Experiment-rationale potential clinical overclaim without limitation: {sentence}")
        if "complete experiment-rationale support package" not in (
            PAPER / "BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md"
        ).read_text(encoding="utf-8"):
            errors.append("BSPC supplementary draft missing experiment-rationale support package note.")
        if "Experiment rationale summary prepared" not in (PAPER / "SUBMISSION_READINESS_CHECKLIST.md").read_text(
            encoding="utf-8"
        ):
            errors.append("Submission checklist missing experiment-rationale item.")

    if (PAPER / "PAPER_WRITING_STATUS.md").exists():
        status_text = (PAPER / "PAPER_WRITING_STATUS.md").read_text(encoding="utf-8")
        for phrase in [
            "Author details missing",
            "Declaration details missing",
            "Journal preferences missing",
            "Provisional first target: Biomedical Signal Processing and Control",
            "Final journal not selected",
            "Figures not generated",
            "Journal formatting not applied",
            "Manuscript not submitted",
            "Manuscript draft word count: 6111",
            "Experiment registry records: 42",
        ]:
            if phrase not in status_text:
                errors.append(f"Paper writing status missing Stage 3B phrase: {phrase}")
        for phrase in [
            "Stage 3E status: confirmed author details and declarations applied; BSPC main figure package generated; awaiting author review; no DOCX/PDF or submission.",
            "Generated BSPC main figures: 5",
            "Figure captions were created in `paper/BSPC_FIGURE_CAPTIONS.md`.",
            "No raw audio, participant identifiers, embeddings, fitted models, prediction files or participant-level outputs were read or committed.",
        ]:
            if phrase not in status_text:
                errors.append(f"Paper writing status missing Stage 3E phrase: {phrase}")
        for phrase in [
            "Stage 3F status: BSPC figure visual review completed; DOCX-ready markdown package prepared; supplementary draft and conversion notes prepared; DOCX not generated because Pandoc was unavailable; no PDF or submission.",
            "DOCX-ready manuscript markdown word count: 6840",
            "Generated BSPC supplement figures: 0",
            "Tables retained in DOCX-ready manuscript: 6",
            "DOCX generated: no; Pandoc was unavailable on PATH during Stage 3F.",
            "No submission was performed.",
        ]:
            if phrase not in status_text:
                errors.append(f"Paper writing status missing Stage 3F phrase: {phrase}")
        if all(path.exists() for path in STAGE3G_REQUIRED_FILES):
            for phrase in [
                "Stage 3G status: BSPC author-review DOCX package generated without Pandoc; review copies need author review; no final PDF or submission.",
                "Stage 3H status: BSPC author declarations finalized, author approvals applied, final-review DOCX files regenerated without Pandoc; portal metadata/upload/system checks, cover letter final review, journal-generated PDF proof review and submission confirmation remain pending; no submission.",
                "BSPC review DOCX files generated: 4",
                "`bspc_final_review_package_ready_for_submission_portal_checks`",
                "Final BSPC review package status: `bspc_final_review_package_ready_for_submission_portal_checks`; final author/declaration inputs and DOCX approvals applied, with portal-only checks pending.",
                "Review DOCX manuscript generated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`.",
                "Review DOCX supplement generated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`.",
                "DOCX generation tool: `python-docx`; Pandoc was not used.",
                "Five main BSPC figure PNG review copies were inserted into the manuscript DOCX.",
                "Fallback text files generated: no.",
                "Final PDF generated: no.",
                "LibreOffice/soffice render QA: not completed because the renderer dependency was unavailable on PATH.",
                "DOCX structural QA confirmed the manuscript DOCX contains five embedded images, visible unresolved author placeholders and no submitted/submission-ready marker.",
                "Manuscript validation status after Stage 3G: passed.",
                "Unit-test status after Stage 3G: passed, 78 tests.",
                "Registry validation status after Stage 3G: passed, 42 records.",
                "## Completed Stage 3H Package Files",
                "Generated `paper/submission_review_package/docx/BSPC_MANUSCRIPT_FINAL_REVIEW.docx`",
                "Generated `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_FINAL_REVIEW.docx`",
                "## Remaining Portal-Only Work",
                "final BSPC/Elsevier portal metadata entry",
                "journal-generated PDF proof",
                "## Stage 3H Final Declaration Package Notes",
                "Stage 3H branch: `paper/bspc-final-declarations-v1`.",
                "Author details finalized: Dr. Vikram Singh, Department of Computer Engineering, National Institute of Technology, Kurukshetra, Haryana, India; email `viks@nitkkr.ac.in`; ORCID not available; corresponding author No.",
                "CRediT statement finalized: Dr. Vikram Singh: Supervision.",
                "Preprint status finalized: A preprint is planned.",
                "BSPC route finalized: The manuscript is being prepared for the Biomedical Signal Processing and Control subscription route.",
                "No Elsevier/BSPC upload, journal-generated PDF proof approval or submission confirmation was performed.",
                "Manuscript validation status after Stage 3H: passed.",
                "Unit-test status after Stage 3H: passed, 80 tests.",
                "Registry validation status after Stage 3H: passed, 42 records.",
                "No clinical claims, dataset counts, PHQ scoring details, frozen metrics or unverified references were introduced.",
            ]:
                if phrase not in status_text:
                    errors.append(f"Paper writing status missing Stage 3G phrase: {phrase}")
            expected_branch = "Current branch: `paper/bspc-final-declarations-v1`"
            if expected_branch not in status_text:
                errors.append(f"Paper writing status missing current branch phrase: {expected_branch}")
        if all(path.exists() for path in EXPERIMENT_RATIONALE_REQUIRED_FILES):
            for phrase in [
                "Experiment-rationale documentation status: prepared from official frozen project sources only; no experiments run and no metrics changed.",
                "Experiment-rationale table rows: 25",
                "## Completed Experiment-Rationale Package Files",
                "`paper/STUDY_EXPERIMENTS_RATIONALE_SUMMARY.md`",
                "`paper/STUDY_EXPERIMENTS_RATIONALE_TABLE.csv`",
                "`paper/EXPERIMENTAL_DESIGN_REVIEWER_EXPLANATION.md`",
                "`paper/EXPERIMENTAL_RATIONALE_MANUSCRIPT_INSERT.md`",
                "`paper/EXPERIMENTAL_RATIONALE_PRESENTATION_NOTES.md`",
                "`paper/EXPERIMENT_DEPENDENCY_FLOW.md`",
                "`paper/EXPERIMENT_SCOPE_BOUNDARIES.md`",
                "No experiments were run.",
                "No experiment registry records, raw outputs, participant counts, metrics, confidence intervals or interpretation categories were changed.",
                "No new scientific claims were introduced beyond the existing audited evidence.",
            ]:
                if phrase not in status_text:
                    errors.append(f"Paper writing status missing experiment-rationale phrase: {phrase}")
        if VOICE_PHYSIOLOGY_AUDIT_FILE.exists():
            for phrase in [
                "Voice physiology motivation status: concise subsection added to BSPC manuscript draft and DOCX-ready source; no experiments run and no metrics changed.",
                "Voice physiology insert word count: 261",
                "`paper/VOICE_PHYSIOLOGY_INSERT_AUDIT.md`",
                "No clinical diagnostic, deployment, privacy-preserving, old DAIC count or PHQ-8 severity-scoring claims were introduced.",
            ]:
                if phrase not in status_text:
                    errors.append(f"Paper writing status missing voice physiology phrase: {phrase}")
            for phrase in [
                "Final review package readiness category after Stage 3G consolidation: `bspc_final_review_package_generated_needs_author_review`.",
                "Readiness category: `bspc_final_review_package_ready_for_submission_portal_checks`.",
                "Review DOCX manuscript regenerated at `paper/submission_review_package/docx/BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx`.",
                "Review DOCX supplement regenerated at `paper/submission_review_package/docx/BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx`.",
                "The regenerated manuscript DOCX includes the physiology motivation subsection and five main figure PNG review copies.",
                "Final consolidation validation status: passed.",
                "Final consolidation unit-test status: passed, 80 tests.",
                "Final consolidation registry validation status: passed, 42 records.",
                "No metrics, references, author details, declarations or clinical claims were invented.",
            ]:
                if phrase not in status_text:
                    errors.append(f"Paper writing status missing final review package phrase: {phrase}")

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
