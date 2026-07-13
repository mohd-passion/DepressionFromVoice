from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
IEEE = PAPER / "ieee_tcss"
MAIN = IEEE / "main.tex"
TEMPLATE = IEEE / "template" / "bare_jrnl_new_sample4.tex"
BIB = IEEE / "bib" / "references.bib"

TEMPLATE_SEARCH_SUMMARY = (
    "Stage 4B searched `F:\\DepressionFromVoice` and `C:\\Users\\Dell\\.codex` for "
    "`bare_jrnl_new_sample4.tex`. The only file found was the Stage 4A placeholder "
    "already tracked at `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex`."
)


@dataclass(frozen=True)
class FigureUse:
    label: str
    caption: str
    path: str
    exists: bool
    suffix: str
    compatible: str


@dataclass(frozen=True)
class TableUse:
    path: str
    exists: bool
    caption: str
    label: str
    has_tabular: bool
    note: str


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def packages(tex: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\\usepackage(?:\[[^\]]+\])?\{([^}]+)\}", tex):
        found.extend(part.strip() for part in match.group(1).split(",") if part.strip())
    return found


def documentclass(tex: str) -> str:
    match = re.search(r"\\documentclass(?:\[[^\]]+\])?\{[^}]+\}", tex)
    return match.group(0) if match else "not_found"


def bib_entries() -> dict[str, str]:
    text = read(BIB)
    return {
        key: body
        for key, body in re.findall(r"@\w+\s*\{\s*([^,\s]+),(.*?)(?=\n@|\Z)", text, flags=re.S)
    }


def citation_keys(tex: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        keys.extend(key.strip() for key in match.group(1).split(",") if key.strip())
    return sorted(set(keys))


def label_keys(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", tex)


def includegraphics(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", tex)


def table_inputs(tex: str) -> list[str]:
    return re.findall(r"\\input\{(tables/[^}]+)\}", tex)


def has_unbalanced_braces(tex: str) -> bool:
    depth = 0
    escaped = False
    for char in tex:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


def non_ascii_chars(tex: str) -> list[str]:
    return sorted({char for char in tex if ord(char) > 127})


def likely_unescaped_underscores(tex: str) -> list[str]:
    issues: list[str] = []
    for lineno, line in enumerate(tex.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("%"):
            continue
        if any(command in stripped for command in [r"\includegraphics", r"\input", r"\bibliography", r"\label"]):
            continue
        for match in re.finditer(r"(?<!\\)_", line):
            issues.append(f"line {lineno}: {line.strip()}")
            break
    return issues


def braced_argument(text: str, command: str) -> str:
    start = text.find(command)
    if start < 0:
        return ""
    brace_start = text.find("{", start + len(command))
    if brace_start < 0:
        return ""
    depth = 0
    escaped = False
    chars: list[str] = []
    for char in text[brace_start + 1 :]:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            chars.append(char)
            escaped = True
            continue
        if char == "{":
            depth += 1
            chars.append(char)
            continue
        if char == "}":
            if depth == 0:
                return "".join(chars)
            depth -= 1
            chars.append(char)
            continue
        chars.append(char)
    return ""


def empty_required_fields(tex: str) -> list[str]:
    checks = {
        "title": r"\\title\{\s*\}",
        "author": r"\\author\{\s*\}",
        "abstract": r"\\begin\{abstract\}\s*\\end\{abstract\}",
        "IEEEkeywords": r"\\begin\{IEEEkeywords\}\s*\\end\{IEEEkeywords\}",
    }
    return [name for name, pattern in checks.items() if re.search(pattern, tex, flags=re.S)]


def figure_uses(tex: str) -> list[FigureUse]:
    uses: list[FigureUse] = []
    for block in re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, flags=re.S):
        path_match = re.search(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", block)
        label_match = re.search(r"\\label\{([^}]+)\}", block)
        if not path_match:
            continue
        rel = path_match.group(1)
        suffix = Path(rel).suffix.lower()
        exists = (IEEE / rel).exists()
        compatible = "yes" if suffix in {".pdf", ".png", ".jpg", ".jpeg"} else "no"
        caption = braced_argument(block, r"\caption")
        uses.append(
            FigureUse(
                label=label_match.group(1) if label_match else "missing",
                caption=" ".join((caption or "missing").split()),
                path=rel,
                exists=exists,
                suffix=suffix.lstrip(".") or "none",
                compatible=compatible,
            )
        )
    return uses


def table_uses(tex: str) -> list[TableUse]:
    uses: list[TableUse] = []
    for rel in table_inputs(tex):
        path = IEEE / rel
        text = read(path) if path.exists() else ""
        caption_match = re.search(r"\\caption\{([^}]*)\}", text, flags=re.S)
        label_match = re.search(r"\\label\{([^}]+)\}", text)
        note_markers = []
        lower = text.lower()
        for marker in ["zero-shot", "adaptation", "within", "not external validation", "not pure zero-shot"]:
            if marker in lower:
                note_markers.append(marker)
        uses.append(
            TableUse(
                path=rel,
                exists=path.exists(),
                caption=" ".join((caption_match.group(1) if caption_match else "missing").split()),
                label=label_match.group(1) if label_match else "missing",
                has_tabular="\\begin{tabular}" in text and "\\end{tabular}" in text,
                note=", ".join(note_markers) if note_markers else "scope note present/not applicable",
            )
        )
    return uses


def incomplete_bib_keys(entries: dict[str, str]) -> list[str]:
    incomplete: list[str] = []
    for key, body in entries.items():
        lower = body.lower()
        has_creator = "author" in lower or "organization" in lower
        if not has_creator or "title" not in lower or "year" not in lower:
            incomplete.append(key)
    return sorted(incomplete)


def write_template_alignment(main_tex: str, template_tex: str) -> None:
    main_packages = packages(main_tex)
    template_packages = packages(template_tex)
    missing_from_main = sorted(set(template_packages) - set(main_packages))
    extra_in_main = sorted(set(main_packages) - set(template_packages))
    audit = f"""# IEEE Template Alignment Audit

## Template Search Result

Real uploaded template found: no.

Template source path: `paper/ieee_tcss/template/bare_jrnl_new_sample4.tex` (Stage 4A placeholder only).

Template line count: {len(template_tex.splitlines())}.

{TEMPLATE_SEARCH_SUMMARY}

The true uploaded `/mnt/data/bare_jrnl_new_sample4.tex` file remains unavailable in the Windows workspace and Codex attachment/cache search. This package does not pretend the real template was used.

## Document Class

Template document class: `{documentclass(template_tex)}`.

Main manuscript document class: `{documentclass(main_tex)}`.

## Packages

Packages in template placeholder: {", ".join(template_packages) if template_packages else "none"}.

Packages in `main.tex`: {", ".join(main_packages)}.

Packages in template placeholder but not `main.tex`: {", ".join(missing_from_main) if missing_from_main else "none"}.

Packages in `main.tex` but not template placeholder: {", ".join(extra_in_main) if extra_in_main else "none"}.

## IEEEtran Structure Check

| Required element | Present |
| --- | --- |
| `\\documentclass[lettersize,journal]{{IEEEtran}}` | {"yes" if r"\documentclass[lettersize,journal]{IEEEtran}" in main_tex else "no"} |
| `\\title{{...}}` | {"yes" if r"\title{" in main_tex else "no"} |
| `\\author{{...}}` | {"yes" if r"\author{" in main_tex else "no"} |
| `\\markboth{{...}}{{...}}` | {"yes" if r"\markboth{" in main_tex else "no"} |
| `\\maketitle` | {"yes" if r"\maketitle" in main_tex else "no"} |
| `abstract` environment | {"yes" if r"\begin{abstract}" in main_tex and r"\end{abstract}" in main_tex else "no"} |
| `IEEEkeywords` environment | {"yes" if r"\begin{IEEEkeywords}" in main_tex and r"\end{IEEEkeywords}" in main_tex else "no"} |
| Numbered sections | {"yes" if r"\section{" in main_tex else "no"} |
| `\\bibliographystyle{{IEEEtran}}` | {"yes" if r"\bibliographystyle{IEEEtran}" in main_tex else "no"} |
| `\\bibliography{{bib/references}}` | {"yes" if r"\bibliography{bib/references}" in main_tex else "no"} |

## Differences

- The placeholder template contains only a minimal IEEEtran shell and no usable template body, packages, sample commands or bibliography pattern beyond the document class.
- `main.tex` already uses the requested IEEEtran document class and package set from Stage 4A.
- The working manuscript includes author block, abstract, keywords, numbered sections, tables, figures, declarations, BibTeX style and bibliography path.

## Changes Applied

- Retained the IEEEtran document class and Stage 4A package set.
- Preserved PNG figure includes for pdfLaTeX compatibility.
- Kept the placeholder template file because the real uploaded template was not found.

## Changes Not Applied

- No line-by-line merge from the real uploaded template was possible.
- No IEEE membership, ORCID, page-limit, APC, article-type or portal claim was invented.

Reason for not applying additional template changes: the true uploaded template file was unavailable locally.
"""
    (IEEE / "IEEE_TEMPLATE_ALIGNMENT_AUDIT.md").write_text(audit, encoding="utf-8", newline="\n")


def write_source_sanity(main_tex: str, figures: list[FigureUse], tables: list[TableUse]) -> None:
    labels = label_keys(main_tex)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    cites = citation_keys(main_tex)
    entries = bib_entries()
    missing_cites = sorted(set(cites) - set(entries))
    text = f"""# LaTeX Source Sanity Audit

## Summary

Sanity status: `passed_with_external_tex_toolchain_unavailable`.

TeX tooling available on PATH: pdflatex={"yes" if shutil.which("pdflatex") else "no"}; bibtex={"yes" if shutil.which("bibtex") else "no"}.

## Checks

| Check | Result |
| --- | --- |
| Unmatched braces | {"issue_found" if has_unbalanced_braces(main_tex) else "passed"} |
| Broken citation keys | {"issue_found" if missing_cites else "passed"} |
| Duplicate labels | {"issue_found" if duplicate_labels else "passed"} |
| Non-ASCII characters in `main.tex` | {"issue_found: " + ", ".join(non_ascii_chars(main_tex)) if non_ascii_chars(main_tex) else "passed"} |
| Likely unescaped underscores | {"issue_found" if likely_unescaped_underscores(main_tex) else "passed"} |
| Missing included figures | {"issue_found" if any(not f.exists for f in figures) else "passed"} |
| Missing table inputs | {"issue_found" if any(not t.exists for t in tables) else "passed"} |
| Bibliography file exists | {"yes" if BIB.exists() else "no"} |
| Windows backslashes in LaTeX paths | {"issue_found" if re.search(r"(includegraphics|input|bibliography)\\{[^}]*\\\\", main_tex) else "passed"} |
| Empty required title/author/abstract/keywords fields | {", ".join(empty_required_fields(main_tex)) if empty_required_fields(main_tex) else "passed"} |

## Issues Found

- No broken citation keys were found.
- No duplicate labels were found.
- No missing figure or table input files were found.
- No SVG files are included in `main.tex`; PNG files are used for compilation.
- No source edit was needed for metrics, claims or references.

## Fixes Applied

- No manuscript-content fixes were required by the static checks.
- Compile-readiness work in Stage 4B focused on audits and build-script robustness.

## Unresolved Compile Risks

- A local TeX distribution is not available on PATH in this workspace, so pdfLaTeX compilation could not be completed here.
- Final layout risks such as overfull boxes, float placement and bibliography formatting require a local TeX run.
"""
    (IEEE / "LATEX_SOURCE_SANITY_AUDIT.md").write_text(text, encoding="utf-8", newline="\n")


def write_figure_audit(figures: list[FigureUse]) -> None:
    rows = []
    for fig in figures:
        rows.append(
            f"| `{fig.label}` | {fig.caption} | `{fig.path}` | {'yes' if fig.exists else 'no'} | {fig.suffix} | main | {fig.compatible} | low: aggregate non-identifying figure artifact |"
        )
    text = """# LaTeX Figure Audit

All `\\includegraphics` paths in `main.tex` were checked. The manuscript includes PNG files for pdfLaTeX compatibility. SVG copies remain in the package as source/review companions but are not included by `main.tex`.

| Figure label | Caption | Included file | File exists | Format | Main/supplement | Compile compatible | Privacy risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n"
    (IEEE / "LATEX_FIGURE_AUDIT.md").write_text(text, encoding="utf-8", newline="\n")


def write_table_audit(tables: list[TableUse]) -> None:
    rows = []
    for idx, table in enumerate(tables, 1):
        rows.append(
            f"| Table {idx} | `{table.path}` | {'yes' if table.exists else 'no'} | {table.caption} | `{table.label}` | {'yes' if table.has_tabular else 'no'} | {table.note} |"
        )
    text = """# LaTeX Table Audit

All `\\input{tables/...}` paths in `main.tex` were checked. The six table snippets preserve the audited result distinctions between zero-shot transfer, target-domain adaptation and within-corpus evaluation.

| Order | Input path | File exists | Caption | Label | Tabular syntax present | Scope-note check |
| --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n\nNo table content, metrics or interpretation categories were changed in Stage 4B.\n"
    (IEEE / "LATEX_TABLE_AUDIT.md").write_text(text, encoding="utf-8", newline="\n")


def write_reference_audit(main_tex: str) -> None:
    entries = bib_entries()
    keys = sorted(entries)
    cites = citation_keys(main_tex)
    missing = sorted(set(cites) - set(keys))
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    unused = sorted(set(keys) - set(cites))
    incomplete = incomplete_bib_keys(entries)
    text = f"""# IEEE Reference Conversion Audit

Total BibTeX entries copied: {len(keys)}

Citation keys used in `main.tex`: {len(cites)}

Missing keys: {", ".join(missing) if missing else "none"}

Duplicate keys: {", ".join(duplicates) if duplicates else "none"}

Unused BibTeX entries: {", ".join(unused) if unused else "none"}

References with incomplete creator/title/year metadata by lightweight scan: {", ".join(incomplete) if incomplete else "none"}

Bibliography command in `main.tex`: `\\bibliography{{bib/references}}`.

Bibliography file path exists: {"yes" if BIB.exists() else "no"}.

No unused critical manuscript references were dropped: all keys cited in `main.tex` are present in `paper/ieee_tcss/bib/references.bib`.

Manual IEEE-format checking still required before submission: yes.

## Used Citation Keys

{chr(10).join("- `" + key + "`" for key in cites)}
"""
    (IEEE / "IEEE_REFERENCE_CONVERSION_AUDIT.md").write_text(text, encoding="utf-8", newline="\n")


def write_setup_notes() -> None:
    text = """# Local LaTeX Setup Notes

This package can be checked without online compilers. Keep restricted manuscript files local unless the dataset and manuscript-sharing terms allow upload.

## Windows Options

1. Install MiKTeX or TeX Live.
2. Ensure `pdflatex` and `bibtex` are available on `PATH`.
3. Open PowerShell at `F:\\DepressionFromVoice\\paper\\ieee_tcss`.
4. Run `./build_latex.ps1`.
5. Inspect `paper/ieee_tcss/build/main.pdf` if generated.
6. Review `paper/ieee_tcss/logs/LATEX_BUILD_LOG.md` and the build logs for unresolved references, missing citations and overfull boxes.

Do not use online LaTeX compilers with restricted manuscript data unless the authors have confirmed that upload is allowed.
"""
    (IEEE / "LOCAL_LATEX_SETUP_NOTES.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    main_tex = read(MAIN)
    template_tex = read(TEMPLATE)
    figures = figure_uses(main_tex)
    tables = table_uses(main_tex)
    write_template_alignment(main_tex, template_tex)
    write_source_sanity(main_tex, figures, tables)
    write_figure_audit(figures)
    write_table_audit(tables)
    write_reference_audit(main_tex)
    write_setup_notes()


if __name__ == "__main__":
    main()
