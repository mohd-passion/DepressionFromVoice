from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
LOGS = ROOT / "logs"
LOG = LOGS / "LATEX_BUILD_LOG.md"


def write_log(status: str, lines: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    content = [
        "# LaTeX Build Log",
        "",
        f"Last attempted: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Status: {status}",
        "",
        *lines,
        "",
    ]
    LOG.write_text("\n".join(content), encoding="utf-8", newline="\n")


def copy_tree_or_file(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def run_step(name: str, command: list[str]) -> tuple[str, int]:
    result = subprocess.run(command, cwd=BUILD, text=True, capture_output=True, check=False)
    log_path = BUILD / f"{name}.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8", newline="\n")
    return log_path.name, result.returncode


def main() -> int:
    main_file = "main.tex"
    main_path = ROOT / main_file
    BUILD.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if not main_path.exists():
        write_log("failed_missing_main_tex", [f"Main file not found: {main_path}", "No source files were modified."])
        print(f"LaTeX build failed: main file not found. See {LOG}")
        return 1

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        write_log(
            "not_run_missing_tex_toolchain",
            [
                f"pdflatex available: {bool(pdflatex)}",
                "",
                f"bibtex available: {bool(bibtex)}",
                "",
                "PDF generated: False",
                "",
                "No PDF was generated because a complete TeX toolchain was not available on PATH.",
                "",
                "Next steps: install MiKTeX or TeX Live, ensure pdflatex and bibtex are on PATH, then rerun this helper from paper/ieee_tcss.",
                "",
                "No source files were modified.",
            ],
        )
        print(f"LaTeX build skipped: pdflatex or bibtex not found. See {LOG}")
        return 0

    for name in [main_file, "declarations.tex"]:
        copy_tree_or_file(ROOT / name, BUILD / name)
    for name in ["bib", "figures", "tables"]:
        copy_tree_or_file(ROOT / name, BUILD / name)

    stem = Path(main_file).stem
    steps = [
        ("pdflatex_pass1", [pdflatex, "-interaction=nonstopmode", "-halt-on-error", main_file]),
        ("bibtex", [bibtex, stem]),
        ("pdflatex_pass2", [pdflatex, "-interaction=nonstopmode", "-halt-on-error", main_file]),
        ("pdflatex_pass3", [pdflatex, "-interaction=nonstopmode", "-halt-on-error", main_file]),
    ]
    results = [run_step(name, command) for name, command in steps]
    combined = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in BUILD.glob("*.log"))
    warnings = []
    if "LaTeX Warning" in combined or "Package " in combined and "Warning" in combined:
        warnings.append("LaTeX/package warnings present; inspect build/*.log.")
    if "undefined references" in combined or "undefined citations" in combined or "Citation " in combined:
        warnings.append("Unresolved references or citations may be present.")
    if "Overfull \\hbox" in combined or "Overfull \\vbox" in combined:
        warnings.append("Overfull boxes detected.")
    if not warnings:
        warnings.append("No warning summary detected by the lightweight log scan.")

    pdf_path = BUILD / f"{stem}.pdf"
    status = "succeeded" if pdf_path.exists() and all(code == 0 for _, code in results) else "failed_see_build_logs"
    write_log(
        status,
        [
            "Output directory: paper/ieee_tcss/build",
            "",
            f"PDF path: {pdf_path}",
            "",
            f"PDF generated: {pdf_path.exists()}",
            "",
            "Tool paths:",
            f"- pdflatex: {pdflatex}",
            f"- bibtex: {bibtex}",
            "",
            "Build steps:",
            *[f"- {name}: exit {code}, log build/{log_name}" for (log_name, code), (name, _) in zip(results, steps)],
            "",
            "Warning summary:",
            *[f"- {warning}" for warning in warnings],
            "",
            "No source files were modified.",
        ],
    )
    print(f"LaTeX build status: {status}. See {LOG}")
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
