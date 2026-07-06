from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from docx import Document
    from docx.enum.section import WD_SECTION_START
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    Document = None  # type: ignore[assignment]
    WD_SECTION_START = None  # type: ignore[assignment]
    WD_TABLE_ALIGNMENT = None  # type: ignore[assignment]
    WD_CELL_VERTICAL_ALIGNMENT = None  # type: ignore[assignment]
    WD_ALIGN_PARAGRAPH = None  # type: ignore[assignment]
    OxmlElement = None  # type: ignore[assignment]
    qn = None  # type: ignore[assignment]
    Inches = None  # type: ignore[assignment]
    Pt = None  # type: ignore[assignment]
    RGBColor = None  # type: ignore[assignment]
    PYTHON_DOCX_ERROR: str | None = str(exc)
else:
    PYTHON_DOCX_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
PACKAGE = PAPER / "submission_review_package"
DOCX_DIR = PACKAGE / "docx"
LOG_DIR = PACKAGE / "logs"

MANUSCRIPT_MD = PAPER / "BSPC_DOCX_READY_MANUSCRIPT.md"
SUPPLEMENT_MD = PAPER / "BSPC_SUPPLEMENTARY_MATERIAL_DRAFT.md"
CAPTIONS_MD = PAPER / "BSPC_FIGURE_CAPTIONS.md"

MANUSCRIPT_DOCX = DOCX_DIR / "BSPC_MANUSCRIPT_AUTHOR_REVIEW.docx"
SUPPLEMENT_DOCX = DOCX_DIR / "BSPC_SUPPLEMENTARY_AUTHOR_REVIEW.docx"
MANUSCRIPT_FALLBACK = DOCX_DIR / "BSPC_MANUSCRIPT_AUTHOR_REVIEW_FALLBACK.txt"
SUPPLEMENT_FALLBACK = DOCX_DIR / "BSPC_SUPPLEMENTARY_AUTHOR_REVIEW_FALLBACK.txt"
GENERATION_LOG = LOG_DIR / "BSPC_DOCX_GENERATION_LOG.md"

FIGURE_FILES = {
    "Figure 1": PAPER / "figures" / "output" / "figure1_study_design_protocol_overview.png",
    "Figure 2": PAPER / "figures" / "output" / "figure2_daic_phase1_key_results.png",
    "Figure 3": PAPER / "figures" / "output" / "figure3_daic_android_transfer_adaptation.png",
    "Figure 4": PAPER / "figures" / "output" / "figure4_android_cross_task_results.png",
    "Figure 5": PAPER / "figures" / "output" / "figure5_domain_shift_boundary_summary.png",
}

REQUIRED_PLACEHOLDERS = [
    "[AUTHOR CONFIRMATION REQUIRED: consent statement]",
    "[AUTHOR CONFIRMATION REQUIRED: acknowledgements]",
    "[AUTHOR CONFIRMATION REQUIRED: Dr. Vikram Singh email and ORCID]",
    "[AUTHOR CONFIRMATION REQUIRED: Dr. Vikram Singh CRediT roles]",
    "[AUTHOR CONFIRMATION REQUIRED: AI-assisted writing/tool-use disclosure]",
    "[AUTHOR CONFIRMATION REQUIRED: final BSPC route]",
]

PRESET_NAME = "standard_business_brief"
PAGE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
CELL_MARGIN_DXA = {"top": 80, "bottom": 80, "start": 120, "end": 120}
HEADER_FILL = "F2F4F7"
HEADING_BLUE = "2E74B5"
HEADING_DARK_BLUE = "1F4D78"
MUTED_TEXT = "666666"


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def ensure_dirs() -> None:
    DOCX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def rgb(hex_color: str):
    if RGBColor is None:
        raise RuntimeError("python-docx is unavailable")
    return RGBColor.from_string(hex_color)


def set_run_font(run, name: str = "Calibri", size_pt: float | None = None) -> None:
    run.font.name = name
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:ascii"), name)
    r_fonts.set(qn("w:hAnsi"), name)


def configure_document(document) -> None:
    section = document.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in [
        ("Heading 1", 16, HEADING_BLUE, 16, 8),
        ("Heading 2", 13, HEADING_BLUE, 12, 6),
        ("Heading 3", 12, HEADING_DARK_BLUE, 8, 4),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = rgb(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.10

    for name in ["List Bullet", "List Number"]:
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = 1.167

    caption = styles["Caption"]
    caption.font.name = "Calibri"
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = rgb(MUTED_TEXT)
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(9)

    footer = section.footer.paragraphs[0]
    footer.text = "Author review copy - not for submission"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in footer.runs:
        set_run_font(run, "Calibri", 9)
        run.font.color.rgb = rgb(MUTED_TEXT)


def add_title_block(document, title: str, subtitle: str) -> None:
    title_p = document.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(6)
    title_run = title_p.add_run(title)
    set_run_font(title_run, "Calibri", 18)
    title_run.bold = True
    title_run.font.color.rgb = rgb("0B2545")

    sub_p = document.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(10)
    sub_run = sub_p.add_run(subtitle)
    set_run_font(sub_run, "Calibri", 11)
    sub_run.italic = True
    sub_run.font.color.rgb = rgb(MUTED_TEXT)

    notice = document.add_paragraph()
    notice.paragraph_format.space_after = Pt(12)
    notice_run = notice.add_run(
        "AUTHOR REVIEW COPY - NOT FOR SUBMISSION. "
        "Unresolved author/declaration placeholders are intentionally preserved."
    )
    set_run_font(notice_run, "Calibri", 10)
    notice_run.bold = True
    notice_run.font.color.rgb = rgb("9B1C1C")


def clean_inline(text: str) -> str:
    return text.replace("`", "")


def add_body_paragraph(document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(clean_inline(text))
    set_run_font(run, "Calibri", 11)
    if "[AUTHOR CONFIRMATION REQUIRED:" in text or "[BIBLIOGRAPHY PLACEHOLDER:" in text:
        run.bold = True
        run.font.color.rgb = rgb("9B1C1C")


def add_placeholder_paragraph(document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run(text)
    set_run_font(run, "Calibri", 10)
    run.bold = True
    run.font.color.rgb = rgb("7A5A00")


def add_list_item(document, text: str, numbered: bool = False) -> None:
    paragraph = document.add_paragraph(style="List Number" if numbered else "List Bullet")
    run = paragraph.add_run(clean_inline(text))
    set_run_font(run, "Calibri", 11)
    if "[AUTHOR CONFIRMATION REQUIRED:" in text:
        run.bold = True
        run.font.color.rgb = rgb("9B1C1C")


def add_heading(document, text: str, level: int) -> None:
    style_level = min(max(level, 1), 3)
    paragraph = document.add_paragraph(style=f"Heading {style_level}")
    run = paragraph.add_run(clean_inline(text))
    set_run_font(run, "Calibri", {1: 16, 2: 13, 3: 12}[style_level])
    run.bold = True


def parse_caption_map() -> dict[str, str]:
    text = CAPTIONS_MD.read_text(encoding="utf-8")
    captions: dict[str, str] = {}
    matches = list(re.finditer(r"^## (Figure [1-5])\. (.+)$", text, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        figure = match.group(1)
        title = match.group(2).strip()
        body = " ".join(line.strip() for line in text[start:end].strip().splitlines() if line.strip())
        captions[figure] = f"{figure}. {title}. {body}"
    return captions


def insert_figure_review_copy(document, figure_label: str, captions: dict[str, str]) -> bool:
    image_path = FIGURE_FILES[figure_label]
    add_placeholder_paragraph(document, f"[INSERT {figure_label.upper()} HERE: see {image_path.relative_to(ROOT).as_posix()}]")
    if not image_path.exists():
        add_body_paragraph(document, f"{figure_label} PNG was not found at {image_path.relative_to(ROOT).as_posix()}.")
        return False

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.15))

    caption_text = captions.get(figure_label)
    if caption_text:
        caption_p = document.add_paragraph(style="Caption")
        caption_run = caption_p.add_run(clean_inline(caption_text))
        set_run_font(caption_run, "Calibri", 9)
        caption_run.font.color.rgb = rgb(MUTED_TEXT)
    return True


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    content = stripped.replace("|", "").replace(" ", "")
    return bool(content) and set(content) <= {"-", ":"}


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_blocks(lines: list[str]) -> Iterable[tuple[int, int, list[list[str]]]]:
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        start = index
        rows: list[list[str]] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            if not is_table_separator(lines[index]):
                rows.append(split_table_row(lines[index]))
            index += 1
        if rows:
            yield start, index, rows


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_cell_margins(table) -> None:
    tbl_pr = table._tbl.tblPr
    cell_mar = tbl_pr.first_child_found_in("w:tblCellMar")
    if cell_mar is None:
        cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(cell_mar)
    for side, value in CELL_MARGIN_DXA.items():
        node = cell_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            cell_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, column_count: int) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    widths = [PAGE_WIDTH_DXA // column_count for _ in range(column_count)]
    widths[-1] += PAGE_WIDTH_DXA - sum(widths)

    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(PAGE_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    if grid is not None:
        for grid_col in list(grid):
            grid.remove(grid_col)
        for width in widths:
            grid_col = OxmlElement("w:gridCol")
            grid_col.set(qn("w:w"), str(width))
            grid.append(grid_col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            set_cell_width(cell, widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_table_cell_margins(table)


def add_markdown_table(document, rows: list[list[str]]) -> None:
    if not rows:
        return
    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    set_table_geometry(table, column_count)

    for row_index, row in enumerate(rows):
        for column_index in range(column_count):
            cell = table.cell(row_index, column_index)
            text = clean_inline(row[column_index]) if column_index < len(row) else ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(text)
            set_run_font(run, "Calibri", 9)
            if row_index == 0:
                run.bold = True
                set_cell_shading(cell, HEADER_FILL)
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(6)


def add_markdown_content(document, markdown_text: str, captions: dict[str, str], include_figures: bool) -> int:
    lines = markdown_text.splitlines()
    table_ranges = {line_index: (end, rows) for line_index, end, rows in table_blocks(lines)}
    figures_inserted = 0
    index = 0
    title_consumed = False
    while index < len(lines):
        if index in table_ranges:
            end, rows = table_ranges[index]
            add_markdown_table(document, rows)
            index = end
            continue

        raw_line = lines[index]
        line = raw_line.strip()
        if not line:
            index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            hashes, heading = heading_match.groups()
            if len(hashes) == 1 and not title_consumed:
                title_consumed = True
                index += 1
                continue
            add_heading(document, heading, max(1, len(hashes) - 1))
            index += 1
            continue

        figure_match = re.match(r"^\[(Figure [1-5]) about here\.\]$", line)
        if figure_match:
            figure_label = figure_match.group(1)
            add_placeholder_paragraph(document, line)
            if include_figures and insert_figure_review_copy(document, figure_label, captions):
                figures_inserted += 1
            index += 1
            continue

        if re.match(r"^\[Table [1-6] about here\.\]$", line):
            add_placeholder_paragraph(document, line)
            index += 1
            continue

        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered:
            add_list_item(document, numbered.group(1), numbered=True)
            index += 1
            continue

        if line.startswith("- "):
            add_list_item(document, line[2:])
            index += 1
            continue

        add_body_paragraph(document, line)
        index += 1

    return figures_inserted


def create_docx(source: Path, output: Path, subtitle: str, include_figures: bool) -> int:
    if Document is None:
        raise RuntimeError("python-docx is unavailable")

    text = source.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1) if title_match else source.stem.replace("_", " ")

    document = Document()
    configure_document(document)
    add_title_block(document, title, subtitle)
    captions = parse_caption_map()
    figures_inserted = add_markdown_content(document, text, captions, include_figures=include_figures)
    document.save(output)
    return figures_inserted


def create_fallback(source: Path, output: Path, title: str) -> None:
    text = source.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(
            [
                title,
                "AUTHOR REVIEW COPY - NOT FOR SUBMISSION",
                "DOCX generation fallback: python-docx was unavailable.",
                "",
                text,
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def placeholder_status(paths: list[Path]) -> dict[str, bool]:
    combined = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.exists())
    return {placeholder: placeholder in combined for placeholder in REQUIRED_PLACEHOLDERS}


def write_generation_log(
    *,
    docx_available: bool,
    figures_inserted: int,
    outputs: list[Path],
    fallback_outputs: list[Path],
    known_limitations: list[str],
) -> None:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git(["rev-parse", "--short", "HEAD"])
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    retained = placeholder_status([MANUSCRIPT_MD, SUPPLEMENT_MD])
    soffice_available = shutil.which("soffice") is not None or shutil.which("libreoffice") is not None

    lines = [
        "# BSPC DOCX Generation Log",
        "",
        f"Date generated: {generated_at}",
        f"Branch: `{branch}`",
        f"Commit at generation: `{commit}`",
        f"Design preset: `{PRESET_NAME}`",
        "",
        "## Source Markdown Files",
        "",
        f"- `{MANUSCRIPT_MD.relative_to(ROOT).as_posix()}`",
        f"- `{SUPPLEMENT_MD.relative_to(ROOT).as_posix()}`",
        f"- `{CAPTIONS_MD.relative_to(ROOT).as_posix()}`",
        "",
        "## Output Files",
        "",
    ]
    if outputs:
        lines.extend(f"- `{path.relative_to(ROOT).as_posix()}`" for path in outputs)
    if fallback_outputs:
        lines.extend(f"- `{path.relative_to(ROOT).as_posix()}`" for path in fallback_outputs)

    lines.extend(
        [
            "",
            "## Tooling",
            "",
            f"- Tool used: {'python-docx 1.2.0-compatible generator' if docx_available else 'plain-text fallback generator'}",
            f"- python-docx available: {'yes' if docx_available else 'no'}",
            f"- python-docx import error: `{PYTHON_DOCX_ERROR}`" if PYTHON_DOCX_ERROR else "- python-docx import error: not_applicable",
            f"- Figures inserted: {'yes' if figures_inserted == 5 else 'partial' if figures_inserted else 'no'}",
            f"- Main figures inserted: {figures_inserted} of 5",
            f"- LibreOffice/soffice available for render QA: {'yes' if soffice_available else 'no'}",
            f"- Visual render QA status: {'pending_manual_render_review' if soffice_available else 'not_completed_soffice_unavailable'}",
            "- Final PDF generated: no",
            "- Submission/upload performed: no",
            "",
            "## Unresolved Placeholders Retained",
            "",
        ]
    )
    lines.extend(f"- {'yes' if present else 'no'}: `{placeholder}`" for placeholder, present in retained.items())
    lines.extend(["", "## Known Formatting Limitations", ""])
    lines.extend(f"- {item}" for item in known_limitations)
    lines.extend(
        [
            "",
            "## Next Manual Checks Needed",
            "",
            "- Author review of every unresolved placeholder.",
            "- Author visual approval of Figures 1-5.",
            "- Final BSPC route confirmation.",
            "- Final Word formatting check before any submission.",
            "- Final references formatting from `paper/references.bib`.",
            "- Final cover letter review; the existing cover letter remains draft-only.",
        ]
    )
    GENERATION_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ensure_dirs()
    outputs: list[Path] = []
    fallback_outputs: list[Path] = []
    figures_inserted = 0
    known_limitations = [
        "Review DOCX files are generated from Markdown and are not final BSPC submission files.",
        "References remain a bibliography placeholder and must be generated from `paper/references.bib` later.",
        "Unresolved author/declaration placeholders are intentionally visible.",
        "LibreOffice/soffice was not available on PATH during Stage 3G, so page-image render QA could not be completed in this environment.",
        "Final Word pagination and BSPC artwork compliance still require author/manual review.",
    ]

    if Document is None:
        create_fallback(MANUSCRIPT_MD, MANUSCRIPT_FALLBACK, "BSPC Manuscript Author Review")
        create_fallback(SUPPLEMENT_MD, SUPPLEMENT_FALLBACK, "BSPC Supplementary Material Author Review")
        fallback_outputs = [MANUSCRIPT_FALLBACK, SUPPLEMENT_FALLBACK]
        known_limitations.append("DOCX files were not generated because python-docx was unavailable.")
        write_generation_log(
            docx_available=False,
            figures_inserted=0,
            outputs=[],
            fallback_outputs=fallback_outputs,
            known_limitations=known_limitations,
        )
        return 0

    for fallback in [MANUSCRIPT_FALLBACK, SUPPLEMENT_FALLBACK]:
        if fallback.exists():
            fallback.unlink()

    figures_inserted = create_docx(
        MANUSCRIPT_MD,
        MANUSCRIPT_DOCX,
        "Biomedical Signal Processing and Control manuscript review copy",
        include_figures=True,
    )
    create_docx(
        SUPPLEMENT_MD,
        SUPPLEMENT_DOCX,
        "Biomedical Signal Processing and Control supplementary material review copy",
        include_figures=False,
    )
    outputs = [MANUSCRIPT_DOCX, SUPPLEMENT_DOCX]
    write_generation_log(
        docx_available=True,
        figures_inserted=figures_inserted,
        outputs=outputs,
        fallback_outputs=[],
        known_limitations=known_limitations,
    )
    print(f"generated={MANUSCRIPT_DOCX.relative_to(ROOT).as_posix()}")
    print(f"generated={SUPPLEMENT_DOCX.relative_to(ROOT).as_posix()}")
    print(f"figures_inserted={figures_inserted}")
    print(f"log={GENERATION_LOG.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
