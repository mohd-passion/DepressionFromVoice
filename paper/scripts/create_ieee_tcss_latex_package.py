from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
IEEE = PAPER / "ieee_tcss"

TITLE = "Cross-Corpus and Cross-Task Depression Detection from Speech Using Frozen WavLM Representations"
SHORT_TITLE = "Cross-Corpus and Cross-Task Depression Detection from Speech"
DATE_CHECKED = "2026-07-13"
SOURCE_BRANCH = "paper/bspc-final-declarations-v1"
TARGET_BRANCH = "paper/ieee-tcss-latex-v1"

FIGURES = {
    1: ("figure1_study_design_protocol_overview.png", "Study design/protocol overview", "fig:study-design"),
    2: ("figure2_daic_phase1_key_results.png", "DAIC Phase 1 key results", "fig:daic-phase1"),
    3: ("figure3_daic_android_transfer_adaptation.png", "DAIC-to-Android transfer and adaptation", "fig:transfer-adaptation"),
    4: ("figure4_android_cross_task_results.png", "Android reading, interview and cross-task results", "fig:android-cross-task"),
    5: ("figure5_domain_shift_boundary_summary.png", "Domain-shift and classifier-boundary summary", "fig:domain-shift-boundary"),
}

TABLE_FILES = {
    1: ("table1_dataset_task_summary.tex", "Dataset and task summary", "tab:dataset-task-summary"),
    2: ("table2_daic_development_results.tex", "DAIC development results", "tab:daic-development"),
    3: ("table3_cross_corpus_android_interview.tex", "Cross-corpus Android interview results", "tab:android-interview-transfer"),
    4: ("table4_android_reading_results.tex", "Android reading results", "tab:android-reading"),
    5: ("table5_cross_task_results.tex", "Participant-coherent cross-task results", "tab:cross-task"),
    6: ("table6_result_scope_claim_matrix.tex", "Result-scope and claim matrix", "tab:result-scope"),
}

TABLE_NOTES = {
    1: "DAIC-WOZ denotes the Distress Analysis Interview Corpus--Wizard-of-Oz; HC denotes healthy control and PT denotes patient group labels in the Androids Corpus. Android labels are HC/PT group labels rather than PHQ-8 labels.",
    2: "Phase 1B/1C values are train-only nested cross-validation estimates. Phase 1F values are provisional local labeled DAIC test results, not official hidden AVEC test results.",
    3: "Protocols using Android labels for threshold adaptation or calibration are target-domain adaptation protocols, not pure zero-shot evaluation. Android-trained logistic regression is within-Android evaluation, not external validation.",
    4: "The Android-trained row is within-corpus validation and not independent external validation.",
    5: "All rows are within the Androids Corpus. Pooled gains are suggestive because uncertainty intervals include zero.",
    6: "This matrix preserves the distinction between methodological ablations, provisional local testing, zero-shot external evaluation, target-domain adaptation and within-corpus validation.",
}


def normalize_text(text: str) -> str:
    replacements = {
        "\u2013": "--",
        "\u2014": "---",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": r"$\times$",
        "â€“": "--",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_escape(text: str) -> str:
    text = normalize_text(text)
    cites: list[str] = []

    def repl_cite(match: re.Match[str]) -> str:
        raw = match.group(1)
        keys = [part.strip().lstrip("@") for part in re.split(r";|,", raw) if part.strip()]
        token = f"ZXQCITE{len(cites)}ZXQ"
        cites.append(r"\cite{" + ",".join(keys) + "}")
        return token

    text = re.sub(r"\[([^\]]*@[^\]]+)\]", repl_cite, text)
    code_spans: list[str] = []

    def repl_code(match: re.Match[str]) -> str:
        token = f"ZXQCODE{len(code_spans)}ZXQ"
        content = match.group(1).replace("\\", r"\textbackslash{}")
        content = content.replace("_", r"\_").replace("%", r"\%")
        code_spans.append(r"\texttt{" + content + "}")
        return token

    text = re.sub(r"`([^`]+)`", repl_code, text)
    escaped = []
    for ch in text:
        escaped.append(
            {
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
                "\\": r"\textbackslash{}",
            }.get(ch, ch)
        )
    out = "".join(escaped)
    for i, replacement in enumerate(cites):
        out = out.replace(f"ZXQCITE{i}ZXQ", replacement)
    for i, replacement in enumerate(code_spans):
        out = out.replace(f"ZXQCODE{i}ZXQ", replacement)
    return out.replace("MMD\\textasciicircum{}2", r"MMD\textsuperscript{2}")


def paragraph_to_latex(paragraph: str) -> str:
    if paragraph.startswith("- "):
        return r"\item " + latex_escape(paragraph[2:].strip())
    if re.match(r"^\d+\.\s+", paragraph):
        return r"\item " + latex_escape(re.sub(r"^\d+\.\s+", "", paragraph))
    return latex_escape(paragraph)


def parse_md_table_blocks(path: Path) -> dict[int, tuple[list[str], list[list[str]]]]:
    text = path.read_text(encoding="utf-8")
    blocks: dict[int, tuple[list[str], list[list[str]]]] = {}
    for match in re.finditer(r"## Table (\d+) - [^\n]+\n\n((?:\|.*\n)+)", text):
        table_no = int(match.group(1))
        header: list[str] = []
        rows: list[list[str]] = []
        for line in match.group(2).splitlines():
            stripped = line.strip()
            body = stripped.strip("|")
            if set(body.replace("|", "").replace(" ", "")) <= {"-", ":"}:
                continue
            cells = [cell.strip().strip("`") for cell in body.split("|")]
            if not header:
                header = cells
            else:
                rows.append(cells)
        blocks[table_no] = (header, rows)
    return blocks


def table_to_latex(table_no: int, header: list[str], rows: list[list[str]]) -> str:
    _filename, caption, label = TABLE_FILES[table_no]
    col_spec = "l" * len(header)

    def row(cells: list[str]) -> str:
        return " & ".join(latex_escape(cell) for cell in cells) + r" \\"

    lines = [
        r"\begin{table*}[!t]",
        r"\caption{" + latex_escape(caption) + "}",
        r"\label{" + label + "}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        row(header),
        r"\midrule",
    ]
    lines.extend(row(r) for r in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            r"\begin{minipage}{0.96\textwidth}",
            r"\vspace{2pt}",
            r"\footnotesize " + latex_escape(TABLE_NOTES[table_no]),
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_figure_captions(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    captions: dict[int, str] = {}
    for match in re.finditer(r"## Figure (\d+)\. [^\n]+\n\n(.*?)(?=\n## Figure |\Z)", text, flags=re.S):
        n = int(match.group(1))
        body = " ".join(line.strip() for line in match.group(2).splitlines() if line.strip())
        body = re.sub(r" Source basis:.*", "", body)
        body = re.sub(r" Sources are.*", "", body)
        body = re.sub(r" Prohibited interpretation:.*", "", body)
        captions[n] = body.strip()
    return captions


def figure_env(n: int, captions: dict[int, str]) -> str:
    filename, fallback_caption, label = FIGURES[n]
    caption = captions.get(n) or fallback_caption
    return "\n".join(
        [
            r"\begin{figure*}[!t]",
            r"\centering",
            r"\includegraphics[width=0.95\textwidth]{figures/" + filename + "}",
            r"\caption{" + latex_escape(caption) + "}",
            r"\label{" + label + "}",
            r"\end{figure*}",
        ]
    )


def convert_body(md: str, captions: dict[int, str]) -> str:
    in_body = False
    out: list[str] = []
    pending_items: list[str] = []

    def flush_items() -> None:
        nonlocal pending_items
        if pending_items:
            out.append(r"\begin{itemize}")
            out.extend(pending_items)
            out.append(r"\end{itemize}")
            out.append("")
            pending_items = []

    for raw in md.splitlines():
        stripped = raw.strip()
        if stripped == "## 1. Introduction":
            in_body = True
        if not in_body:
            continue
        if stripped in {
            "## Declaration of Competing Interest",
            "## Funding",
            "## Ethics Approval",
            "## Consent to Participate",
            "## Consent for Publication",
            "## CRediT Authorship Contribution Statement",
            "## Acknowledgements",
            "## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process",
            "## Preprint Statement",
            "## BSPC Submission Route",
            "## Table Captions And Placement Notes",
            "## Figure Captions",
            "## References",
            "## Supplementary Material Note",
        }:
            break
        if not stripped:
            flush_items()
            continue
        fig_match = re.match(r"\[Figure (\d+) about here\.\]", stripped)
        if fig_match:
            flush_items()
            out.append(figure_env(int(fig_match.group(1)), captions))
            out.append("")
            continue
        table_match = re.match(r"\[Table (\d+) about here\.\]", stripped)
        if table_match:
            flush_items()
            table_no = int(table_match.group(1))
            out.append(r"\input{tables/" + TABLE_FILES[table_no][0] + "}")
            out.append("")
            continue
        if stripped.startswith("## "):
            flush_items()
            title = re.sub(r"^##\s+\d+\.\s*", "", stripped)
            if title == "Materials and Methods":
                out.append(r"\section{Datasets and Experimental Design}")
            elif title == "Data and Code Availability":
                out.append(r"\section{Data and Code Availability}")
            else:
                out.append(r"\section{" + latex_escape(title) + "}")
            out.append("")
            continue
        if stripped.startswith("### "):
            flush_items()
            title = re.sub(r"^###\s+\d+\.\d+\s*", "", stripped)
            if title == "Ethical and Clinical Considerations":
                out.append(r"\section{Ethical and Clinical Considerations}")
            elif title == "Speech preprocessing":
                out.append(r"\section{Methods}")
                out.append("")
                out.append(r"\subsection{Speech Preprocessing}")
            elif title in {
                "WavLM representation",
                "Classifiers",
                "Evaluation protocols",
                "Leakage prevention",
                "Metrics and uncertainty",
                "Reproducibility and experiment governance",
            }:
                if r"\section{Methods}" not in out:
                    out.append(r"\section{Methods}")
                    out.append("")
                out.append(r"\subsection{" + latex_escape(title[:1].upper() + title[1:]) + "}")
            else:
                out.append(r"\subsection{" + latex_escape(title[:1].upper() + title[1:]) + "}")
            out.append("")
            continue
        if stripped.startswith("#### "):
            flush_items()
            out.append(r"\subsubsection{" + latex_escape(stripped[5:].strip()) + "}")
            out.append("")
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            pending_items.append(paragraph_to_latex(stripped))
            continue
        flush_items()
        out.append(paragraph_to_latex(stripped))
        out.append("")
    flush_items()
    body = "\n".join(out).strip()
    tcss_framing = (
        r"\section{Introduction}"
        "\n\n"
        "From an IEEE Transactions on Computational Social Systems perspective, this work treats speech as a human "
        "behavioral signal whose computational representation can shift across language, elicitation task and dataset "
        "context. The study therefore emphasizes robustness, transfer and responsible interpretation of behavioral "
        "computing systems rather than clinical deployment."
        "\n\n"
    )
    return body.replace(r"\section{Introduction}" + "\n\n", tcss_framing, 1)


def extract_abstract_keywords(md: str) -> tuple[str, str]:
    abstract = re.search(r"## Abstract\n\n(.*?)\n\n## Keywords", md, flags=re.S).group(1).strip()
    keywords = re.search(r"## Keywords\n\n(.*?)\n\n## 1\. Introduction", md, flags=re.S).group(1).strip()
    return abstract, keywords


def bib_keys(path: Path) -> list[str]:
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)", path.read_text(encoding="utf-8"))


def incomplete_bib_entries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\s*\{\s*([^,\s]+),(.*?)(?=\n@|\Z)", text, flags=re.S)
    return [key for key, body in entries if "title" not in body.lower() or "year" not in body.lower()]


def used_citation_keys(tex: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        keys.extend([key.strip() for key in match.group(1).split(",") if key.strip()])
    return sorted(set(keys))


def md_to_simple_tex(md: str) -> str:
    out: list[str] = []
    in_items = False
    for raw in md.splitlines():
        stripped = raw.strip()
        if not stripped:
            if in_items:
                out.append(r"\end{itemize}")
                in_items = False
            out.append("")
            continue
        if stripped.startswith("# "):
            if in_items:
                out.append(r"\end{itemize}")
                in_items = False
            out.append(r"\section*{" + latex_escape(stripped[2:].strip()) + "}")
        elif stripped.startswith("## "):
            if in_items:
                out.append(r"\end{itemize}")
                in_items = False
            out.append(r"\section{" + latex_escape(stripped[3:].strip()) + "}")
        elif stripped.startswith("### "):
            if in_items:
                out.append(r"\end{itemize}")
                in_items = False
            out.append(r"\subsection{" + latex_escape(stripped[4:].strip()) + "}")
        elif stripped.startswith("- "):
            if not in_items:
                out.append(r"\begin{itemize}")
                in_items = True
            out.append(r"\item " + latex_escape(stripped[2:].strip()))
        else:
            if in_items:
                out.append(r"\end{itemize}")
                in_items = False
            out.append(latex_escape(stripped))
    if in_items:
        out.append(r"\end{itemize}")
    return "\n".join(out)


def write_static_package_files() -> None:
    (IEEE / "README.md").write_text(
        f"""# IEEE TCSS LaTeX Preparation Package

Status: `draft_submission_preparation_not_submitted`

This directory contains a draft LaTeX preparation package for IEEE Transactions on Computational Social Systems. No submission, IEEE upload, portal account creation or final PDF submission has been performed.

Source branch: `{SOURCE_BRANCH}`

Working branch: `{TARGET_BRANCH}`

Primary manuscript: `paper/ieee_tcss/main.tex`

Bibliography: `paper/ieee_tcss/bib/references.bib`

Figures: `paper/ieee_tcss/figures/`

Tables: `paper/ieee_tcss/tables/`

Supplementary source: `paper/ieee_tcss/supplementary/supplementary.tex`

The package reframes the BSPC manuscript for an IEEE TCSS review-preparation context, but target-journal fit and all portal-specific requirements must be verified before upload.

The requested uploaded template `/mnt/data/bare_jrnl_new_sample4.tex` was not found in the local workspace or Codex attachment cache. The file at `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex` records this missing-template condition and is not asserted to be the original uploaded template.

Frozen metrics, participant counts, experiment registry records, PHQ scoring details and clinical limitations were not changed.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "IEEE_TCSS_GUIDELINES_VERIFICATION.md").write_text(
        f"""# IEEE TCSS Guidelines Verification

Date checked: {DATE_CHECKED}

Official sources only were used. Details that could not be verified from official IEEE pages are marked `not_verified`.

| Item | Status | Official source | Notes |
| --- | --- | --- | --- |
| Journal | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems | IEEE Transactions on Computational Social Systems. |
| Scope/aims | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems | The official TCSS page frames the journal around modeling, simulation, analysis and understanding of social systems from quantitative and computational perspectives. It lists social-system behavior, structure, dynamics, social intelligence/cognition and computational behavior modeling among topic areas. |
| Peer review | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems | The page states that articles are peer reviewed by at least two independent reviewers using single-anonymous peer review and plagiarism screening. |
| Article type | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | Regular papers and correspondences are referenced. This package is prepared as a regular paper draft. |
| Manuscript format | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | IEEE Transactions two-column format on 8.5 x 11 inch pages is specified. |
| LaTeX template guidance | verified | https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/ieee-article-templates/ | IEEE Author Center provides article templates for IEEE journal article preparation. The user-supplied local template file was not found in the workspace. |
| Abstract requirements | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | Abstract length is listed as 100-250 words. |
| Keywords requirements | not_verified | not_verified | A TCSS-specific keyword count was not verified from the official pages checked. |
| Graphical abstract requirement | not_verified | https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/ | IEEE Author Center provides graphical abstract guidance and notes that graphical abstracts, if used, must be peer reviewed. A TCSS requirement was not verified. |
| Biography requirement | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | The TCSS information page states that accepted papers require head-and-shoulders photos and technical biographies. This is treated as an accepted-paper-stage requirement. |
| ORCID | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | The page states that all IEEE journals require ORCID for all authors. No ORCID identifiers were invented in this package. |
| Reference style | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | References are required in numeric IEEE style. This package uses BibTeX with `IEEEtran`. |
| Figure requirements | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ ; https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/ | Official guidance requests sharp, high-contrast illustrations and IEEE Author Center graphics guidance. |
| Supplementary material policy | verified | https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/ | IEEE Author Center lists supplementary-material file types and says supplementary files should be labeled and uploaded separately. |
| Page limits | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | Regular papers are about 10 Transactions pages; correspondences are no more than 5 pages. Maximum overlength cap requires final confirmation during submission. |
| Overlength page charges | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | The page lists voluntary page charges and mandatory overlength page charges beyond regular-paper/correspondence limits. Final charge applicability must be confirmed in the portal. |
| Open access versus traditional route | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ ; https://journals.ieeeauthorcenter.ieee.org/choose-a-publishing-agreement/open-access-rights-management/ | TCSS is described as hybrid, with traditional and open-access routes. The TCSS page lists an OA fee, while IEEE Author Center says to check the current APC list for up-to-date pricing. |
| Submission portal | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | The TCSS page links submission to the manuscript submission system. No portal account or upload was performed. |
| Preprint policy | verified | https://www.ieeesmc.org/publications/transactions-on-computational-social-systems/information-for-authors/ | The page requires disclosure of preprints and specified preprint-site wording. |
| AI-tool disclosure requirements | not_verified | not_verified | A TCSS-specific AI-disclosure rule was not verified on the official pages checked. The author-approved AI disclosure is retained. |
| Data/code availability requirements | not_verified | not_verified | A TCSS-specific data/code availability requirement was not verified on the official pages checked. The existing data/code availability statement is retained. |
| IEEE Xplore journal page | not_verified | https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=6570650 | The IEEE Xplore page returned HTTP 418 in this environment and was not used for claims. |

No IEEE submission or upload was performed.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "IEEE_TCSS_JOURNAL_FIT_AUDIT.md").write_text(
        """# IEEE TCSS Journal Fit Audit

## Current Manuscript Framing

- Speech-based depression-related classification using frozen WavLM representations.
- Cross-corpus evaluation from DAIC-WOZ to Androids Corpus.
- Reading/interview cross-task transfer within Androids Corpus.
- Calibration, threshold and classifier-boundary analysis.
- Responsible AI and clinical-claim limitations.

## TCSS Fit Angle

The strongest TCSS angle is computational behavioral signal modeling: speech is treated as a human behavioral signal whose representation and decision boundary can change across corpus, language, elicitation task and label context. The paper also fits human-centered AI concerns by separating representation transfer from classifier transfer and by preserving research-only, non-deployment claims.

## Fit Assessment

Fit status: `moderate_with_reframing_needed`.

The manuscript is no longer being framed primarily as biomedical signal processing. It should emphasize computational social/behavioral systems, robustness under dataset and task shift, responsible behavioral computing, and the limits of automated mental-health signal interpretation.

## Risks

- The experiments are not a social-network, agent-based or large-scale social-system simulation study.
- The manuscript can look more biomedical than computational-social unless the introduction and discussion foreground behavioral-signal transfer, human-centered AI and responsible interpretation.
- TCSS fit should not be overclaimed; the experiments support robustness analysis for behavioral speech modeling, not broad social-system theory.

## Required Changes Before Submission

- Consider a title that explicitly uses behavioral signal analysis or behavioral computing language.
- Keep the abstract focused on cross-corpus/cross-task robustness and decision-boundary shift.
- Reframe the introduction around human behavioral signals and computational social/behavioral assessment.
- Strengthen related work around behavioral computing and responsible AI without inventing citations.
- Keep clinical limitations visible and avoid a purely diagnostic framing.
- Confirm TCSS article type, page length, supplementary upload rules and ORCID requirements in the final portal workflow.

## Overclaiming Check

No claim is made that this study establishes a clinical diagnostic system, deployment readiness, fairness across demographic groups or universal speech biomarkers.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "TITLE_REFRAMING_OPTIONS.md").write_text(
        f"""# Title Reframing Options for IEEE TCSS

Final title status: `unchanged_pending_author_decision`

Option 1, current author-approved title:

- {TITLE}

IEEE TCSS-suitable alternatives for author review:

1. Cross-Corpus and Cross-Task Evaluation of Self-Supervised Speech Representations for Depression-Related Behavioral Signal Analysis
2. Speech-Based Depression-Related Behavioral Signal Modeling Across Corpora and Tasks Using Frozen WavLM Representations
3. Representation Transfer and Classifier-Boundary Shift in Speech-Based Depression-Related Behavioral Computing
4. Robustness of Frozen Speech Representations for Depression-Related Behavioral Signals Across Corpora and Tasks
5. Cross-Dataset Behavioral Speech Modeling for Depression-Related Classification with Frozen WavLM Features

No title change was applied automatically in `main.tex`.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "IEEE_TCSS_COVER_LETTER_DRAFT.md").write_text(
        f"""# IEEE TCSS Cover Letter Draft

Dear Editor,

Please consider our manuscript, "{TITLE}," for review as a regular paper in IEEE Transactions on Computational Social Systems.

The manuscript examines speech as a depression-related behavioral signal and evaluates how frozen self-supervised WavLM representations behave across corpus and task shifts. The study separates zero-shot transfer, threshold adaptation, calibration and target-domain classifier learning, with explicit safeguards against clinical overclaiming.

[AUTHOR FINAL CONFIRMATION REQUIRED: This manuscript is not under consideration elsewhere.]

A preprint is planned. The study uses previously collected datasets under their original access terms and does not recruit new human participants. No clinical validation or deployment-readiness claim is made.

Sincerely,

Mohammad Ilyas
Department of Computer Science and Engineering
Moradabad Institute of Technology
Moradabad, Uttar Pradesh, India
mohd.passion.cse@mitmoradabad.edu.in
""",
        encoding="utf-8",
        newline="\n",
    )


def write_remaining_static_files() -> None:
    (IEEE / "IEEE_TCSS_SUBMISSION_CHECKLIST.md").write_text(
        """# IEEE TCSS Submission Checklist

Status values: `complete`, `pending_author`, `pending_portal`, `not_verified`, `not_applicable`.

| Item | Status | Notes |
| --- | --- | --- |
| IEEE TCSS LaTeX package created | complete | `paper/ieee_tcss/main.tex` created. |
| Source branch recorded | complete | Branched from `paper/bspc-final-declarations-v1`. |
| Uploaded template preserved | not_verified | `/mnt/data/bare_jrnl_new_sample4.tex` was not found locally; placeholder note created. |
| Official TCSS guideline verification | complete | Official IEEE sources checked on 2026-07-13. |
| Journal-fit audit | complete | TCSS fit assessed as moderate with reframing needed. |
| Title decision | pending_author | Current title retained; alternatives documented. |
| Author block | complete | Confirmed names, affiliations and emails used; no ORCID or membership invented. |
| ORCID | pending_author | IEEE requires ORCID; author ORCID values currently not available. |
| Bibliography | complete | `paper/references.bib` copied to `paper/ieee_tcss/bib/references.bib`. |
| Citation-key validity | complete | LaTeX citations use copied BibTeX keys only. |
| Figures | complete | Five approved BSPC figures copied as SVG and PNG; PNG used in LaTeX for review compilation. |
| Tables | complete | Six audited result tables converted to LaTeX snippets. |
| Supplementary material | complete | Supplementary draft converted to `supplementary/supplementary.tex`. |
| Frozen metrics | complete | Metrics preserved; no experiments run. |
| Experiment registry | complete | Registry validation remains read-only. |
| Clinical-claim safeguards | complete | No clinical validation or deployment-readiness claim introduced. |
| Cover letter | pending_author | Draft created; not-under-consideration statement requires author confirmation. |
| IEEE portal metadata | pending_portal | No IEEE portal entry completed. |
| IEEE upload/submission | not_applicable | No upload or submission performed. |
| LaTeX build | pending_portal | Build attempted locally; see `logs/LATEX_BUILD_LOG.md`. |
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "IEEE_TCSS_MIGRATION_AUDIT.md").write_text(
        """# IEEE TCSS Migration Audit

## Scope

Stage 4A migrated the Stage 3H BSPC final-review manuscript source into an IEEE TCSS LaTeX preparation package.

## Source Material

- Authoritative manuscript source: `paper/BSPC_DOCX_READY_MANUSCRIPT.md`.
- Supporting manuscript source: `paper/BSPC_MANUSCRIPT_DRAFT.md`.
- Result tables: `paper/PAPER_RESULTS_TABLES.md`.
- Figure captions: `paper/BSPC_FIGURE_CAPTIONS.md`.
- Bibliography: `paper/references.bib`.
- Supplement: `paper/BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md`.

## Changes Made

- Created `paper/ieee_tcss/main.tex` using IEEEtran journal structure.
- Copied verified BibTeX references into `paper/ieee_tcss/bib/references.bib`.
- Copied five approved figure files into `paper/ieee_tcss/figures/`.
- Converted six audited result tables into `paper/ieee_tcss/tables/`.
- Converted supplementary draft into `paper/ieee_tcss/supplementary/supplementary.tex`.
- Added IEEE TCSS guideline verification, journal-fit audit, title options, cover letter draft and submission checklist.

## Template Exception

The requested uploaded template `/mnt/data/bare_jrnl_new_sample4.tex` was not found in the Windows workspace or Codex attachment cache on 2026-07-13. A placeholder note was created at `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex` so the missing-template condition is explicit. The working manuscript uses the IEEEtran class and package structure requested in Stage 4A.

## Claims Preserved

- No frozen metrics, participant counts, confidence intervals or interpretation categories were changed.
- No experiment registry records were modified.
- No new experiments were run.
- No ORCID identifiers, IEEE memberships, results, clinical validation claims or deployment claims were invented.
- Clinical and ethical limitations remain in the manuscript.

## Remaining Author/Portal Work

- Confirm whether to keep the current title or select one of the TCSS reframing options.
- Provide ORCID identifiers if available; IEEE requires ORCID for all authors.
- Confirm the not-under-consideration cover-letter statement.
- Perform final TCSS portal checks, source upload and journal-generated PDF proof review.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "declarations.tex").write_text(
        r"""\section*{Acknowledgment}
The authors thank the creators and maintainers of the DAIC-WOZ and Androids Corpus datasets. Mohammad Ilyas acknowledges the academic guidance received during his Ph.D. research work.

\section*{Funding}
No funding was received for this study.

\section*{Competing Interests}
The authors declare no competing interests.

\section*{Ethics Approval}
This study used previously collected datasets obtained under their original access terms. No new human participants were recruited. Ethics approval was not sought because the study involved secondary analysis of existing datasets.

\section*{Consent to Participate}
This study used secondary datasets obtained under their original access terms. Consent for the original data collection was managed by the respective dataset providers. No new human participants were recruited in this study.

\section*{Consent for Publication}
This study used secondary datasets obtained under their original access terms. Consent for the original data collection was managed by the respective dataset providers. No new human participants were recruited in this study.

\section*{Author Contributions}
Mohammad Ilyas: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization, Project administration.

Dr. Vikram Singh: Supervision.

\section*{Declaration of Generative AI and AI-Assisted Technologies in the Writing Process}
ChatGPT and Codex were used to assist with code organization, manuscript drafting support, editing, and workflow planning. The authors reviewed and verified the manuscript content, results, references, and final scientific claims.

\section*{Preprint Statement}
A preprint is planned.
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "template" / "bare_jrnl_new_sample4.tex").write_text(
        r"""% Stage 4A template preservation note.
% The user-supplied path /mnt/data/bare_jrnl_new_sample4.tex was not available
% in this Windows Codex workspace or the local attachment cache on 2026-07-13.
% This file is a minimal IEEEtran-compatible placeholder kept at the requested
% reference path so the package records the missing upload honestly. It is not
% asserted to be the original uploaded IEEE template.
\documentclass[lettersize,journal]{IEEEtran}
\begin{document}
Template upload not found in the local workspace. Use paper/ieee_tcss/main.tex as the working manuscript.
\end{document}
""",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "build_latex.ps1").write_text(
        r"""param(
    [string]$MainFile = "main.tex"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build"
$LogDir = Join-Path $Root "logs"
$LogPath = Join-Path $LogDir "LATEX_BUILD_LOG.md"
New-Item -ItemType Directory -Force -Path $BuildDir, $LogDir | Out-Null

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"

if (-not $pdflatex -or -not $bibtex) {
    @"
# LaTeX Build Log

Last attempted: $timestamp

Status: not_run_missing_tex_toolchain

pdflatex available: $([bool]$pdflatex)

bibtex available: $([bool]$bibtex)

No PDF was generated because a complete TeX toolchain was not available on PATH.
"@ | Set-Content -Path $LogPath -Encoding UTF8
    Write-Host "LaTeX build skipped: pdflatex or bibtex not found. See $LogPath"
    exit 0
}

Push-Location $Root
try {
    Copy-Item -Path $MainFile -Destination $BuildDir -Force
    Copy-Item -Path "declarations.tex" -Destination $BuildDir -Force
    Copy-Item -Path "bib" -Destination $BuildDir -Recurse -Force
    Copy-Item -Path "figures" -Destination $BuildDir -Recurse -Force
    Copy-Item -Path "tables" -Destination $BuildDir -Recurse -Force
    Push-Location $BuildDir
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass1.log"
    & bibtex ([IO.Path]::GetFileNameWithoutExtension($MainFile)) | Tee-Object -FilePath "bibtex.log"
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass2.log"
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass3.log"
    Pop-Location
    @"
# LaTeX Build Log

Last attempted: $timestamp

Status: attempted

Output directory: paper/ieee_tcss/build

Review pdflatex_pass*.log and bibtex.log for warnings/errors.
"@ | Set-Content -Path $LogPath -Encoding UTF8
}
finally {
    while ((Get-Location).Path -ne $Root) { Pop-Location }
    Pop-Location
}
""",
        encoding="utf-8-sig",
        newline="\n",
    )

    (IEEE / "build_latex.py").write_text(
        "from pathlib import Path\n\n"
        "print('Use build_latex.ps1 for the primary IEEE TCSS LaTeX build attempt.')\n"
        "print(Path(__file__).resolve().parent / 'logs' / 'LATEX_BUILD_LOG.md')\n",
        encoding="utf-8",
        newline="\n",
    )

    (IEEE / "logs" / "LATEX_BUILD_LOG.md").write_text(
        """# LaTeX Build Log

Last attempted: not_run_yet

Status: `pending_build_attempt`

Run `powershell -ExecutionPolicy Bypass -File paper/ieee_tcss/build_latex.ps1` from the repository root.
""",
        encoding="utf-8",
        newline="\n",
    )


def write_main_and_audits() -> None:
    for sub in ["figures", "tables", "bib", "supplementary", "logs", "build", "template"]:
        (IEEE / sub).mkdir(parents=True, exist_ok=True)

    md = (PAPER / "BSPC_DOCX_READY_MANUSCRIPT.md").read_text(encoding="utf-8")
    captions = parse_figure_captions(PAPER / "BSPC_FIGURE_CAPTIONS.md")
    abstract, keywords = extract_abstract_keywords(md)
    body = convert_body(md, captions)
    for table_no, (header, rows) in sorted(parse_md_table_blocks(PAPER / "PAPER_RESULTS_TABLES.md").items()):
        (IEEE / "tables" / TABLE_FILES[table_no][0]).write_text(
            table_to_latex(table_no, header, rows),
            encoding="utf-8",
            newline="\n",
        )

    main_tex = f"""\\documentclass[lettersize,journal]{{IEEEtran}}

\\usepackage{{amsmath,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{algorithm}}
\\usepackage{{array}}
\\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{{subfig}}
\\usepackage{{textcomp}}
\\usepackage{{stfloats}}
\\usepackage{{url}}
\\usepackage{{verbatim}}
\\usepackage{{graphicx}}
\\usepackage{{cite}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}

\\begin{{document}}

\\title{{{latex_escape(TITLE)}}}

\\author{{Mohammad~Ilyas~and~Vikram~Singh%
\\thanks{{Mohammad Ilyas is with the Department of Computer Science and Engineering, Moradabad Institute of Technology, Moradabad, Uttar Pradesh, India (e-mail: mohd.passion.cse@mitmoradabad.edu.in). Corresponding author: Mohammad Ilyas.}}%
\\thanks{{Vikram Singh is with the Department of Computer Engineering, National Institute of Technology, Kurukshetra, Haryana, India (e-mail: viks@nitkkr.ac.in).}}%
\\thanks{{ORCID identifiers are not available for either author. No IEEE membership status is claimed.}}}}

\\markboth{{IEEE Transactions on Computational Social Systems,~Vol.~XX, No.~XX, 2026}}%
{{Ilyas \\MakeLowercase{{\\textit{{et al.}}}}: {latex_escape(SHORT_TITLE)}}}

\\maketitle

\\begin{{abstract}}
{latex_escape(abstract)}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{latex_escape(keywords)}
\\end{{IEEEkeywords}}

{body}

\\input{{declarations}}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{bib/references}}

\\begin{{IEEEbiography}}{{Mohammad Ilyas}}
[AUTHOR CONFIRMATION REQUIRED: IEEE biography]
\\end{{IEEEbiography}}

\\begin{{IEEEbiography}}{{Vikram Singh}}
[AUTHOR CONFIRMATION REQUIRED: IEEE biography]
\\end{{IEEEbiography}}

\\end{{document}}
"""
    (IEEE / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")

    supplement_md = (PAPER / "BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md").read_text(encoding="utf-8")
    supplement_tex = (
        "\\documentclass[lettersize,journal]{IEEEtran}\n"
        "\\usepackage{url}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        + md_to_simple_tex(supplement_md)
        + "\n\\end{document}\n"
    )
    (IEEE / "supplementary" / "supplementary.tex").write_text(
        supplement_tex,
        encoding="utf-8",
        newline="\n",
    )

    copied_keys = sorted(bib_keys(IEEE / "bib" / "references.bib"))
    used_keys = used_citation_keys(main_tex)
    missing = sorted(set(used_keys).difference(copied_keys))
    unused = sorted(set(copied_keys).difference(used_keys))
    duplicates = sorted({key for key in copied_keys if copied_keys.count(key) > 1})
    incomplete = incomplete_bib_entries(IEEE / "bib" / "references.bib")
    ref_audit = f"""# IEEE Reference Conversion Audit

Total BibTeX entries copied: {len(copied_keys)}

Citation keys used in `main.tex`: {len(used_keys)}

Missing keys: {', '.join(missing) if missing else 'none'}

Duplicate keys: {', '.join(duplicates) if duplicates else 'none'}

Unused BibTeX entries: {', '.join(unused) if unused else 'none'}

References with incomplete title/year metadata by lightweight scan: {', '.join(incomplete) if incomplete else 'none'}

Manual IEEE-format checking still required before submission: yes.

## Used Citation Keys

{chr(10).join('- `' + key + '`' for key in used_keys)}
"""
    (IEEE / "IEEE_REFERENCE_CONVERSION_AUDIT.md").write_text(
        ref_audit,
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    write_static_package_files()
    write_remaining_static_files()
    write_main_and_audits()


if __name__ == "__main__":
    main()
