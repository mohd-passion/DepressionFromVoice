# Local LaTeX Setup Notes

This package can be checked without online compilers. Keep restricted manuscript files local unless the dataset and manuscript-sharing terms allow upload.

## Windows Options

1. Install MiKTeX or TeX Live.
2. Ensure `pdflatex` and `bibtex` are available on `PATH`.
3. Open PowerShell at `F:\DepressionFromVoice\paper\ieee_tcss`.
4. Run `./build_latex.ps1`.
5. Inspect `paper/ieee_tcss/build/main.pdf` if generated.
6. Review `paper/ieee_tcss/logs/LATEX_BUILD_LOG.md` and the build logs for unresolved references, missing citations and overfull boxes.

Do not use online LaTeX compilers with restricted manuscript data unless the authors have confirmed that upload is allowed.
