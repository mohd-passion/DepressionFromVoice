"""Generate BSPC manuscript figures from compact aggregate records.

The script intentionally avoids raw audio, participant identifiers, embeddings,
fitted models and prediction files. It reads only planning documents, registry
summaries and compact aggregate run records already committed for paper writing.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit(
        "Pillow is required for PNG output. Install Pillow or use the bundled "
        "Codex Python runtime that already includes it."
    ) from exc


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "source"
OUTPUT_DIR = FIGURE_DIR / "output"
AUDIT_PATH = PAPER_DIR / "BSPC_FIGURE_DATA_AUDIT.csv"
CAPTIONS_PATH = PAPER_DIR / "BSPC_FIGURE_CAPTIONS.md"
SOURCE_VALUES_PATH = SOURCE_DIR / "bspc_figure_source_values.csv"
LOG_PATH = OUTPUT_DIR / "figure_generation_log.txt"

PHASE2C_DIR = ROOT / "records" / "runs" / "phase2c_20260703_125614"
PHASE2D_DIR = ROOT / "records" / "runs" / "phase2d_20260704_125931"
PHASE2E_DIR = ROOT / "records" / "runs" / "phase2e_20260704_151323"
ANDROID_TASK_MMD2_PATH = ROOT / "paper" / "ieee_tcss" / "mmd2" / "android_reading_interview_mmd2.json"

INPUT_PATHS = [
    PAPER_DIR / "FIGURE_SPECIFICATIONS.md",
    PAPER_DIR / "BSPC_FIGURE_PRODUCTION_PLAN.md",
    PAPER_DIR / "PAPER_RESULTS_TABLES.md",
    PAPER_DIR / "PAPER_EVIDENCE_MAP.md",
    ROOT / "outputs" / "experiment_registry" / "experiment_summary.csv",
    PHASE2C_DIR / "phase2c_protocol_comparison.csv",
    PHASE2C_DIR / "PHASE2C_WITHIN_ANDROID_REPORT.json",
    PHASE2D_DIR / "phase2d_task_comparison.csv",
    PHASE2D_DIR / "PHASE2D_ANDROID_READING_REPORT.json",
    PHASE2E_DIR / "phase2e_protocol_comparison.csv",
    PHASE2E_DIR / "phase2e_transfer_gap_analysis.csv",
    PHASE2E_DIR / "phase2e_classifier_boundary_analysis.csv",
    PHASE2E_DIR / "PHASE2E_CROSS_TASK_REPORT.json",
    ANDROID_TASK_MMD2_PATH,
]

AUDIT_FIELDS = [
    "figure",
    "panel",
    "metric",
    "plotted value",
    "source file",
    "official value",
    "match status",
    "notes",
]

SOURCE_VALUE_FIELDS = [
    "figure",
    "panel",
    "label",
    "metric",
    "value",
    "source file",
    "source record",
]

FORBIDDEN_GENERATED_TEXT_PATTERNS = [
    re.compile(r"\bparticipant[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\bsubject[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\brecording[_ -]?id\b", re.IGNORECASE),
    re.compile(r"\baudio[_ -]?path\b", re.IGNORECASE),
    re.compile(r"\bfile[_ -]?path\b", re.IGNORECASE),
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_inputs(paths: Iterable[Path]) -> None:
    missing = [rel(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required figure source file(s):\n- " + "\n- ".join(missing))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"No row found where {key}={value!r}")


def find_row_multi(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    joined = ", ".join(f"{key}={value!r}" for key, value in conditions.items())
    raise KeyError(f"No row found where {joined}")


def as_float(value: str | float | int) -> float:
    return float(value)


def f4(value: str | float | int) -> str:
    return f"{as_float(value):.4f}"


def pct(value: str | float | int) -> str:
    return f"{as_float(value) * 100:.1f}%"


def wrap_text(text: str, max_width: float | None, font_size: int) -> list[str]:
    if "\n" in text:
        lines: list[str] = []
        for part in text.splitlines():
            lines.extend(wrap_text(part, max_width, font_size))
        return lines
    if not max_width:
        return [text]

    max_chars = max(8, int(max_width / (font_size * 0.53)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if len(proposed) <= max_chars:
            current = proposed
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def font_path(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


@dataclass(frozen=True)
class Op:
    name: str
    args: dict[str, Any]


class Artwork:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.ops: list[Op] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "#263238",
        stroke_width: float = 1.5,
        radius: float = 8,
    ) -> None:
        self.ops.append(
            Op(
                "rect",
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "fill": fill,
                    "stroke": stroke,
                    "stroke_width": stroke_width,
                    "radius": radius,
                },
            )
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#263238",
        stroke_width: float = 2,
        dash: str | None = None,
    ) -> None:
        self.ops.append(
            Op(
                "line",
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "stroke": stroke,
                    "stroke_width": stroke_width,
                    "dash": dash,
                },
            )
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#263238",
        stroke_width: float = 2.5,
    ) -> None:
        self.ops.append(
            Op(
                "arrow",
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": stroke, "stroke_width": stroke_width},
            )
        )

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str,
        stroke: str = "#263238",
        stroke_width: float = 1.5,
    ) -> None:
        self.ops.append(
            Op("circle", {"x": x, "y": y, "r": r, "fill": fill, "stroke": stroke, "stroke_width": stroke_width})
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 24,
        fill: str = "#1f2a33",
        bold: bool = False,
        anchor: str = "start",
        max_width: float | None = None,
        line_spacing: float = 1.22,
    ) -> None:
        self.ops.append(
            Op(
                "text",
                {
                    "x": x,
                    "y": y,
                    "lines": wrap_text(text, max_width, size),
                    "size": size,
                    "fill": fill,
                    "bold": bold,
                    "anchor": anchor,
                    "line_spacing": line_spacing,
                },
            )
        )

    def save_svg(self, path: Path) -> None:
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{escape(self.title)}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]
        for op in self.ops:
            args = op.args
            if op.name == "rect":
                parts.append(
                    '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                    'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'.format(**args)
                )
            elif op.name == "line":
                dash = f' stroke-dasharray="{args["dash"]}"' if args.get("dash") else ""
                parts.append(
                    f'<line x1="{args["x1"]}" y1="{args["y1"]}" x2="{args["x2"]}" y2="{args["y2"]}" '
                    f'stroke="{args["stroke"]}" stroke-width="{args["stroke_width"]}" '
                    f'stroke-linecap="round"{dash}/>'
                )
            elif op.name == "arrow":
                parts.append(self._svg_arrow(**args))
            elif op.name == "circle":
                parts.append(
                    '<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                    'stroke-width="{stroke_width}"/>'.format(**args)
                )
            elif op.name == "text":
                weight = "700" if args["bold"] else "400"
                parts.append(
                    f'<text x="{args["x"]}" y="{args["y"]}" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="{args["size"]}" font-weight="{weight}" fill="{args["fill"]}" '
                    f'text-anchor="{args["anchor"]}">'
                )
                for index, line in enumerate(args["lines"]):
                    dy = 0 if index == 0 else args["line_spacing"]
                    dy_attr = f'{dy}em' if index > 0 else "0"
                    parts.append(f'<tspan x="{args["x"]}" dy="{dy_attr}">{escape(line)}</tspan>')
                parts.append("</text>")
        parts.append("</svg>")
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")

    def save_png(self, path: Path) -> None:
        image = Image.new("RGB", (self.width, self.height), "#ffffff")
        draw = ImageDraw.Draw(image)
        for op in self.ops:
            args = op.args
            if op.name == "rect":
                bbox = [args["x"], args["y"], args["x"] + args["w"], args["y"] + args["h"]]
                draw.rounded_rectangle(
                    bbox,
                    radius=int(args["radius"]),
                    fill=args["fill"],
                    outline=args["stroke"],
                    width=max(1, int(args["stroke_width"])),
                )
            elif op.name == "line":
                draw.line(
                    [args["x1"], args["y1"], args["x2"], args["y2"]],
                    fill=args["stroke"],
                    width=max(1, int(args["stroke_width"])),
                )
            elif op.name == "arrow":
                self._png_arrow(draw, **args)
            elif op.name == "circle":
                bbox = [args["x"] - args["r"], args["y"] - args["r"], args["x"] + args["r"], args["y"] + args["r"]]
                draw.ellipse(bbox, fill=args["fill"], outline=args["stroke"], width=max(1, int(args["stroke_width"])))
            elif op.name == "text":
                font = font_path(args["size"], args["bold"])
                line_height = int(args["size"] * args["line_spacing"])
                for index, line in enumerate(args["lines"]):
                    bbox = draw.textbbox((0, 0), line, font=font)
                    width = bbox[2] - bbox[0]
                    if args["anchor"] == "middle":
                        x = args["x"] - width / 2
                    elif args["anchor"] == "end":
                        x = args["x"] - width
                    else:
                        x = args["x"]
                    draw.text((x, args["y"] + index * line_height), line, fill=args["fill"], font=font)
        image.save(path)

    @staticmethod
    def _arrow_head_points(x1: float, y1: float, x2: float, y2: float, size: float = 14) -> list[tuple[float, float]]:
        angle = math.atan2(y2 - y1, x2 - x1)
        left = angle + math.pi * 0.82
        right = angle - math.pi * 0.82
        return [(x2, y2), (x2 + size * math.cos(left), y2 + size * math.sin(left)), (x2 + size * math.cos(right), y2 + size * math.sin(right))]

    def _svg_arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float) -> str:
        points = self._arrow_head_points(x1, y1, x2, y2)
        point_text = " ".join(f"{x},{y}" for x, y in points)
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" stroke-linecap="round"/>'
            f'<polygon points="{point_text}" fill="{stroke}"/>'
        )

    def _png_arrow(
        self, draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float
    ) -> None:
        draw.line([x1, y1, x2, y2], fill=stroke, width=max(1, int(stroke_width)))
        draw.polygon(self._arrow_head_points(x1, y1, x2, y2), fill=stroke)


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.source_rows: list[dict[str, str]] = []

    def add(
        self,
        figure: str,
        panel: str,
        metric: str,
        value: str | int | float,
        source_file: Path,
        official_value: str | int | float | None = None,
        notes: str = "",
        source_record: str = "",
        label: str = "",
    ) -> None:
        official = value if official_value is None else official_value
        value_text = f4(value) if isinstance(value, (int, float)) else str(value)
        official_text = f4(official) if isinstance(official, (int, float)) else str(official)
        if value_text == "not_applicable":
            status = "not_applicable"
        elif value_text == official_text:
            status = "match"
        else:
            status = "mismatch"
        self.rows.append(
            {
                "figure": figure,
                "panel": panel,
                "metric": metric,
                "plotted value": value_text,
                "source file": rel(source_file),
                "official value": official_text,
                "match status": status,
                "notes": notes,
            }
        )
        self.source_rows.append(
            {
                "figure": figure,
                "panel": panel,
                "label": label or metric,
                "metric": metric,
                "value": value_text,
                "source file": rel(source_file),
                "source record": source_record,
            }
        )

    def write(self) -> None:
        with AUDIT_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        with SOURCE_VALUES_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_VALUE_FIELDS)
            writer.writeheader()
            writer.writerows(self.source_rows)


COLORS = {
    "blue": "#315f9a",
    "teal": "#2b8a83",
    "green": "#4f8a3d",
    "gold": "#c99b2e",
    "rust": "#bd5b37",
    "plum": "#70558a",
    "gray": "#6d7882",
    "light_blue": "#dbe8f6",
    "light_teal": "#dcefeb",
    "light_gold": "#f3e6bf",
    "light_rust": "#f1d9cf",
    "light_gray": "#edf1f3",
}


def title(canvas: Artwork, main: str, subtitle: str) -> None:
    canvas.text(60, 38, main, size=34, bold=True)
    canvas.text(60, 82, subtitle, size=20, fill="#46525c")


def label_box(canvas: Artwork, x: float, y: float, w: float, h: float, heading: str, body: str, fill: str) -> None:
    canvas.rect(x, y, w, h, fill=fill, stroke="#51606c", stroke_width=1.3, radius=8)
    canvas.text(x + 18, y + 25, heading, size=20, bold=True, max_width=w - 36)
    canvas.text(x + 18, y + 58, body, size=16, fill="#34434d", max_width=w - 36)


def axis(canvas: Artwork, x: float, y: float, w: float, h: float, ymax: float, ticks: list[float]) -> None:
    canvas.line(x, y, x, y + h, stroke="#52616b", stroke_width=1.5)
    canvas.line(x, y + h, x + w, y + h, stroke="#52616b", stroke_width=1.5)
    for tick in ticks:
        ty = y + h - (tick / ymax) * h
        canvas.line(x - 6, ty, x + w, ty, stroke="#d6dde2", stroke_width=1)
        canvas.text(x - 14, ty - 9, f"{tick:.2f}", size=14, fill="#52616b", anchor="end")


def simple_bar_chart(
    canvas: Artwork,
    x: float,
    y: float,
    w: float,
    h: float,
    items: list[dict[str, Any]],
    ymax: float = 1.0,
    title_text: str = "",
    ylabel: str = "",
) -> None:
    if title_text:
        canvas.text(x, y - 26, title_text, size=22, bold=True)
    if ylabel:
        canvas.text(x - 50, y + 10, ylabel, size=15, fill="#52616b")
    axis(canvas, x, y, w, h, ymax, [0.0, ymax * 0.25, ymax * 0.5, ymax * 0.75, ymax])
    slot = w / len(items)
    bar_w = min(82, slot * 0.52)
    for index, item in enumerate(items):
        cx = x + slot * index + slot / 2
        bh = (item["value"] / ymax) * h
        canvas.rect(cx - bar_w / 2, y + h - bh, bar_w, bh, fill=item["color"], stroke="#34434d", radius=4)
        canvas.text(cx, y + h - bh - 30, f4(item["value"]), size=17, bold=True, anchor="middle")
        canvas.text(cx, y + h + 24, item["label"], size=16, anchor="middle", max_width=slot * 0.9)
        if item.get("note"):
            canvas.text(cx, y + h + 84, item["note"], size=13, fill="#5f6f78", anchor="middle", max_width=slot * 0.9)


def grouped_bar_chart(
    canvas: Artwork,
    x: float,
    y: float,
    w: float,
    h: float,
    groups: list[dict[str, Any]],
    metrics: list[tuple[str, str]],
    ymax: float = 1.0,
    title_text: str = "",
) -> None:
    if title_text:
        canvas.text(x, y - 28, title_text, size=22, bold=True)
    axis(canvas, x, y, w, h, ymax, [0.0, 0.25, 0.5, 0.75, 1.0])
    slot = w / len(groups)
    bar_w = min(42, slot / (len(metrics) + 1.8))
    for g_index, group in enumerate(groups):
        center = x + slot * g_index + slot / 2
        start = center - (bar_w * len(metrics) + 8 * (len(metrics) - 1)) / 2
        for m_index, (metric, color) in enumerate(metrics):
            value = group[metric]
            bx = start + m_index * (bar_w + 8)
            bh = (value / ymax) * h
            canvas.rect(bx, y + h - bh, bar_w, bh, fill=color, stroke="#34434d", radius=4)
            canvas.text(bx + bar_w / 2, y + h - bh - 24, f4(value), size=13, bold=True, anchor="middle")
        canvas.text(center, y + h + 24, group["label"], size=15, anchor="middle", max_width=slot * 0.92)
        canvas.text(center, y + h + 74, group["protocol_note"], size=13, fill="#5f6f78", anchor="middle", max_width=slot * 0.92)
    lx = x + w - 220
    for idx, (metric, color) in enumerate(metrics):
        canvas.rect(lx, y - 22 + idx * 30, 22, 16, fill=color, stroke="#34434d", radius=3)
        canvas.text(lx + 32, y - 24 + idx * 30, metric, size=15)


def draw_figure1(audit: Audit) -> tuple[str, Artwork]:
    canvas = Artwork(1500, 980, "Figure 1 study design and protocol overview")
    title(
        canvas,
        "Figure 1. Study design and protocol overview",
        "Shared frozen WavLM layer-9 pipeline with protocol-specific evaluation branches",
    )

    label_box(
        canvas,
        70,
        150,
        240,
        155,
        "Input corpora",
        "DAIC-WOZ interview speech; Android interview speech; Android reading speech",
        COLORS["light_blue"],
    )
    label_box(
        canvas,
        370,
        150,
        240,
        155,
        "Audio windows",
        "16 kHz audio; 10-second windows; 5-second hop; compact aggregate records only",
        COLORS["light_teal"],
    )
    label_box(
        canvas,
        670,
        150,
        240,
        155,
        "Representation",
        "Frozen WavLM layer 9; temporal mean pooling; participant mean aggregation",
        COLORS["light_gold"],
    )
    label_box(
        canvas,
        970,
        150,
        240,
        155,
        "Classifier",
        "Logistic regression; thresholds and calibration kept protocol-specific",
        COLORS["light_rust"],
    )
    label_box(
        canvas,
        1220,
        150,
        210,
        155,
        "Scope guard",
        "Research evaluation only; no clinical deployment claim",
        COLORS["light_gray"],
    )

    for x1, x2 in [(310, 370), (610, 670), (910, 970), (1210, 1220)]:
        canvas.arrow(x1, 228, x2, 228, stroke="#52616b")

    branch_y = 440
    branches = [
        ("DAIC development", "Layer/aggregation selection; provisional local labeled DAIC test", COLORS["blue"]),
        ("DAIC-to-Android", "Frozen zero-shot, threshold adaptation, calibration and Android interview LR", COLORS["teal"]),
        ("Android reading", "Frozen reading reference and Android-trained reading LR", COLORS["green"]),
        ("Cross-task Android", "Participant-coherent reading/interview and pooled-task transfer", COLORS["rust"]),
    ]
    start_x = 110
    for idx, (heading, body, color) in enumerate(branches):
        x = start_x + idx * 345
        label_box(canvas, x, branch_y, 280, 150, heading, body, "#ffffff")
        canvas.rect(x, branch_y, 280, 12, fill=color, stroke=color, radius=4)
        canvas.arrow(1090, 305, x + 140, branch_y, stroke=color, stroke_width=2.2)

    label_box(
        canvas,
        260,
        700,
        420,
        130,
        "Interpretation discipline",
        "Separate representation transfer, threshold shift, calibration shift, classifier-boundary shift and task transfer.",
        COLORS["light_blue"],
    )
    label_box(
        canvas,
        820,
        700,
        420,
        130,
        "Privacy discipline",
        "Figures use aggregate metrics and schematic placeholders only; no raw participant-level artifacts.",
        COLORS["light_teal"],
    )
    canvas.arrow(710, branch_y + 150, 470, 700, stroke="#52616b")
    canvas.arrow(980, branch_y + 150, 1030, 700, stroke="#52616b")

    audit.add(
        "Figure 1",
        "schematic",
        "study design schematic",
        "not_applicable",
        PAPER_DIR / "FIGURE_SPECIFICATIONS.md",
        notes="Schematic uses protocol labels only; no numeric participant-level record is plotted.",
        source_record="Figure 1 specification",
    )
    return "figure1_study_design_protocol_overview", canvas


def draw_figure2(registry_rows: list[dict[str, str]], audit: Audit) -> tuple[str, Artwork]:
    source = ROOT / "outputs" / "experiment_registry" / "experiment_summary.csv"
    wavlm = find_row(registry_rows, "experiment_id", "phase1f_frozen_wavlm")
    timing = find_row(registry_rows, "experiment_id", "phase1f_frozen_timing")
    items = [
        {
            "label": "WavLM free-speech candidate",
            "value": as_float(wavlm["test_auc"]),
            "color": COLORS["blue"],
            "note": "Selected for Android transfer",
            "record": "phase1f_frozen_wavlm",
        },
        {
            "label": "Guided-interview timing candidate",
            "value": as_float(timing["test_auc"]),
            "color": COLORS["gold"],
            "note": "Not Android-compatible",
            "record": "phase1f_frozen_timing",
        },
    ]
    canvas = Artwork(1300, 840, "Figure 2 DAIC Phase 1 key results")
    title(
        canvas,
        "Figure 2. DAIC Phase 1 key results",
        "Free-speech WavLM candidate versus guided-interview timing on the provisional local labeled DAIC test",
    )
    simple_bar_chart(canvas, 180, 190, 780, 440, items, title_text="ROC-AUC", ylabel="Metric value")
    label_box(
        canvas,
        1000,
        205,
        230,
        230,
        "Caveat",
        "These are provisional local labeled DAIC test results, not hidden AVEC test results.",
        COLORS["light_rust"],
    )
    label_box(
        canvas,
        1000,
        470,
        230,
        160,
        "Decision",
        "The WavLM candidate was retained because it is compatible with unconstrained speech transfer.",
        COLORS["light_teal"],
    )
    for item in items:
        audit.add(
            "Figure 2",
            "A",
            "ROC-AUC",
            item["value"],
            source,
            notes="Phase 1F provisional local labeled DAIC test; not an official hidden test estimate.",
            source_record=item["record"],
            label=item["label"],
        )
    return "figure2_daic_phase1_key_results", canvas


def draw_figure3(phase2c_rows: list[dict[str, str]], audit: Audit) -> tuple[str, Artwork]:
    source = PHASE2C_DIR / "phase2c_protocol_comparison.csv"
    specs = [
        ("A_frozen_daic_at_0_5", "Frozen DAIC\n0.5 threshold", "Zero-shot"),
        ("B_threshold_only", "Threshold-only\nadaptation", "Target labels for threshold"),
        ("C_frozen_score_calibration", "Frozen-score\ncalibration", "Target labels for calibration"),
        ("D_android_lr_threshold_adapted", "Android-trained\ninterview LR", "Target labels and refit"),
    ]
    groups: list[dict[str, Any]] = []
    for protocol, label, note in specs:
        row = find_row(phase2c_rows, "protocol", protocol)
        group = {
            "label": label,
            "protocol_note": note,
            "ROC-AUC": as_float(row["roc_auc"]),
            "Macro F1": as_float(row["macro_f1"]),
            "record": protocol,
        }
        groups.append(group)
        for metric, column in [("ROC-AUC", "roc_auc"), ("Macro F1", "macro_f1")]:
            audit.add(
                "Figure 3",
                "A",
                metric,
                as_float(row[column]),
                source,
                notes="DAIC-to-Android interview protocol comparison; aggregate participant-level metric.",
                source_record=protocol,
                label=label.replace("\n", " "),
            )

    canvas = Artwork(1500, 960, "Figure 3 DAIC to Android transfer and adaptation")
    title(
        canvas,
        "Figure 3. DAIC-to-Android transfer and adaptation",
        "Zero-shot frozen scores versus target-label-used threshold, calibration and classifier-boundary protocols",
    )
    grouped_bar_chart(
        canvas,
        140,
        185,
        1160,
        500,
        groups,
        [("ROC-AUC", COLORS["blue"]), ("Macro F1", COLORS["teal"])],
        title_text="Android interview performance by protocol",
    )
    label_box(
        canvas,
        150,
        760,
        540,
        140,
        "Protocol boundary",
        "Protocols B and C use target-domain labels for adaptation. Protocol D refits a target-domain classifier.",
        COLORS["light_gold"],
    )
    label_box(
        canvas,
        770,
        760,
        540,
        140,
        "Allowed interpretation",
        "The frozen DAIC model retained limited ranking transfer; Android-trained boundaries improved within-corpus discrimination.",
        COLORS["light_blue"],
    )
    return "figure3_daic_android_transfer_adaptation", canvas


def draw_figure4(
    phase2d_task_rows: list[dict[str, str]],
    phase2e_rows: list[dict[str, str]],
    gap_rows: list[dict[str, str]],
    audit: Audit,
) -> tuple[str, Artwork]:
    source2d = PHASE2D_DIR / "phase2d_task_comparison.csv"
    source2e = PHASE2E_DIR / "phase2e_protocol_comparison.csv"
    source_gap = PHASE2E_DIR / "phase2e_transfer_gap_analysis.csv"

    task_items: list[dict[str, Any]] = []
    for task, color in [("Android reading", COLORS["green"]), ("Android interview", COLORS["teal"])]:
        row = find_row(phase2d_task_rows, "task", task)
        task_items.append({"label": task.replace("Android ", ""), "value": as_float(row["roc_auc"]), "color": color})
        audit.add(
            "Figure 4",
            "A",
            "ROC-AUC",
            as_float(row["roc_auc"]),
            source2d,
            notes="Phase 2D descriptive task comparison.",
            source_record=task,
            label=task,
        )

    protocols = [
        ("matched_reading_within_task", "Matched\nreading", COLORS["green"]),
        ("matched_interview_within_task", "Matched\ninterview", COLORS["teal"]),
        ("reading_to_interview", "Reading to\ninterview", COLORS["rust"]),
        ("interview_to_reading", "Interview to\nreading", COLORS["gold"]),
        ("pooled_to_reading", "Pooled to\nreading", COLORS["plum"]),
        ("pooled_to_interview", "Pooled to\ninterview", COLORS["blue"]),
    ]
    cross_items: list[dict[str, Any]] = []
    for protocol, label, color in protocols:
        row = find_row_multi(phase2e_rows, protocol=protocol, threshold_variant="threshold_0_5")
        cross_items.append({"label": label, "value": as_float(row["roc_auc"]), "color": color})
        audit.add(
            "Figure 4",
            "B",
            "ROC-AUC",
            as_float(row["roc_auc"]),
            source2e,
            notes="Phase 2E participant-coherent cross-task protocol comparison.",
            source_record=f"{protocol}; threshold_0_5",
            label=label.replace("\n", " "),
        )

    for comparison in ["reading_to_interview_gap", "interview_to_reading_gap", "pooled_to_interview_gain", "pooled_to_reading_gain"]:
        row = find_row_multi(gap_rows, comparison=comparison, threshold_variant="threshold_0_5", metric="roc_auc")
        audit.add(
            "Figure 4",
            "C",
            "ROC-AUC difference",
            as_float(row["difference_a_minus_b"]),
            source_gap,
            notes="Displayed as interpretation text only; intervals are not drawn from this compact record.",
            source_record=comparison,
            label=comparison,
        )

    canvas = Artwork(1550, 980, "Figure 4 Android reading interview and cross-task results")
    title(
        canvas,
        "Figure 4. Android reading, interview and cross-task results",
        "Within-task, cross-task and pooled-task ROC-AUC from compact aggregate records",
    )
    simple_bar_chart(canvas, 130, 190, 430, 420, task_items, title_text="Phase 2D task comparison", ylabel="ROC-AUC")
    simple_bar_chart(canvas, 670, 190, 720, 420, cross_items, title_text="Phase 2E participant-coherent protocols")
    label_box(
        canvas,
        130,
        735,
        520,
        145,
        "Cross-task finding",
        "Cross-task transfer remained strong but below matched-task point estimates.",
        COLORS["light_blue"],
    )
    label_box(
        canvas,
        760,
        735,
        520,
        145,
        "Pooled-task caution",
        "Pooled point estimates were highest, but superiority should not be stated as definitive from these aggregate records.",
        COLORS["light_gold"],
    )
    return "figure4_android_cross_task_results", canvas


def draw_figure5(
    phase2c_report: dict[str, Any],
    phase2d_report: dict[str, Any],
    phase2e_report: dict[str, Any],
    android_task_mmd2_report: dict[str, Any],
    audit: Audit,
) -> tuple[str, Artwork]:
    source2c = PHASE2C_DIR / "PHASE2C_WITHIN_ANDROID_REPORT.json"
    source2d = PHASE2D_DIR / "PHASE2D_ANDROID_READING_REPORT.json"
    source2e = PHASE2E_DIR / "PHASE2E_CROSS_TASK_REPORT.json"

    daic_shift = phase2c_report["domain_shift_summary"]
    task_shift = phase2d_report["task_shift"]
    task_std = task_shift["standardized_feature_shift_using_interview_std"]
    boundary = phase2e_report["classifier_boundary_summary"]
    task_mmd2 = android_task_mmd2_report["results"]["primary_paired_common_participants"]["mmd2"]

    daic_feature_share = daic_shift["features_shifted_more_than_one_daic_std"] / 768
    task_feature_share = task_std["features_shifted_more_than_one_interview_std"] / 768
    domain_metrics = [
        ("Centroid distance", daic_shift["centroid_cosine_distance"], task_shift["task_centroid_cosine_distance"], 0.25),
        ("Mean abs shift", daic_shift["mean_absolute_standardized_shift"], task_std["mean_absolute_shift"], 2.5),
        ("Shifted feature share", daic_feature_share, task_feature_share, 1.0),
    ]
    boundary_items = [
        ("Coeff cosine", boundary["coefficient_cosine_similarity_standardized_space"]["mean"], COLORS["blue"]),
        ("Sign agreement", boundary["coefficient_sign_agreement"]["mean"], COLORS["teal"]),
        ("Top-50 Jaccard", boundary["top_abs_weight_feature_jaccard_at_50"]["mean"], COLORS["rust"]),
        ("Matched score corr.", boundary["score_correlation_matched_source_tasks"]["mean"], COLORS["green"]),
        ("Cross-target corr.", boundary["score_correlation_cross_targets"]["mean"], COLORS["gold"]),
    ]

    audit.add(
        "Figure 5",
        "A",
        "DAIC-to-Android centroid cosine distance",
        daic_shift["centroid_cosine_distance"],
        source2c,
        notes="Domain-shift summary statistic.",
        source_record="domain_shift_summary.centroid_cosine_distance",
        label="DAIC-to-Android",
    )
    audit.add(
        "Figure 5",
        "A",
        "Android reading/interview centroid cosine distance",
        task_shift["task_centroid_cosine_distance"],
        source2d,
        notes="Task-shift summary statistic.",
        source_record="task_shift.task_centroid_cosine_distance",
        label="Android reading versus interview",
    )
    audit.add(
        "Figure 5",
        "A",
        "DAIC-to-Android mean absolute standardized shift",
        daic_shift["mean_absolute_standardized_shift"],
        source2c,
        notes="Domain-shift summary statistic.",
        source_record="domain_shift_summary.mean_absolute_standardized_shift",
        label="DAIC-to-Android",
    )
    audit.add(
        "Figure 5",
        "A",
        "Android reading/interview mean absolute standardized shift",
        task_std["mean_absolute_shift"],
        source2d,
        notes="Task-shift summary statistic.",
        source_record="task_shift.standardized_feature_shift_using_interview_std.mean_absolute_shift",
        label="Android reading versus interview",
    )
    audit.add(
        "Figure 5",
        "A",
        "DAIC-to-Android shifted features",
        f'{daic_shift["features_shifted_more_than_one_daic_std"]}/768',
        source2c,
        official_value=f'{daic_shift["features_shifted_more_than_one_daic_std"]}/768',
        notes="Converted to share for plotting; count label retained.",
        source_record="domain_shift_summary.features_shifted_more_than_one_daic_std",
        label="DAIC-to-Android",
    )
    audit.add(
        "Figure 5",
        "A",
        "Android reading/interview shifted features",
        f'{task_std["features_shifted_more_than_one_interview_std"]}/768',
        source2d,
        official_value=f'{task_std["features_shifted_more_than_one_interview_std"]}/768',
        notes="Converted to share for plotting; count label retained.",
        source_record="task_shift.standardized_feature_shift_using_interview_std.features_shifted_more_than_one_interview_std",
        label="Android reading versus interview",
    )
    audit.add(
        "Figure 5",
        "A",
        "DAIC-to-Android shifted feature share",
        daic_feature_share,
        source2c,
        notes="Computed as shifted features divided by 768 WavLM layer-9 dimensions.",
        source_record="550/768",
        label="DAIC-to-Android",
    )
    audit.add(
        "Figure 5",
        "A",
        "Android reading/interview shifted feature share",
        task_feature_share,
        source2d,
        notes="Computed as shifted features divided by 768 WavLM layer-9 dimensions.",
        source_record="292/768",
        label="Android reading versus interview",
    )
    audit.add(
        "Figure 5",
        "A",
        "DAIC-to-Android MMD^2",
        daic_shift["mmd2"],
        source2c,
        notes="MMD^2 from Phase 2C compact official record.",
        source_record="domain_shift_summary.mmd2",
        label="DAIC-to-Android",
    )
    audit.add(
        "Figure 5",
        "A",
        "Android reading/interview MMD^2",
        task_mmd2,
        ANDROID_TASK_MMD2_PATH,
        notes="Computed in Stage 4J follow-up from existing participant-level reading and interview WavLM layer-9 embeddings using the Phase 2C biased RBF MMD^2 method.",
        source_record="results.primary_paired_common_participants.mmd2",
        label="Android reading versus interview",
    )
    for metric, value, _color in boundary_items:
        key = {
            "Coeff cosine": "coefficient_cosine_similarity_standardized_space",
            "Sign agreement": "coefficient_sign_agreement",
            "Top-50 Jaccard": "top_abs_weight_feature_jaccard_at_50",
            "Matched score corr.": "score_correlation_matched_source_tasks",
            "Cross-target corr.": "score_correlation_cross_targets",
        }[metric]
        audit.add(
            "Figure 5",
            "B",
            metric,
            value,
            source2e,
            notes="Mean across Phase 2E participant-coherent folds.",
            source_record=f"classifier_boundary_summary.{key}.mean",
            label=metric,
        )

    canvas = Artwork(1550, 940, "Figure 5 domain shift and classifier boundary summary")
    title(
        canvas,
        "Figure 5. Domain-shift and classifier-boundary summary",
        "Aggregate feature-shift statistics and Phase 2E classifier-boundary similarity metrics",
    )
    panel_x, panel_y = 90, 170
    canvas.text(panel_x, panel_y - 34, "A. Feature-distribution shift", size=23, bold=True)
    canvas.text(panel_x + 140, panel_y + 18, "DAIC to Android", size=18, bold=True, anchor="middle")
    canvas.text(panel_x + 430, panel_y + 18, "Reading vs interview", size=18, bold=True, anchor="middle")
    for idx, (metric, daic_value, task_value, maximum) in enumerate(domain_metrics):
        y = panel_y + 70 + idx * 125
        canvas.text(panel_x, y + 8, metric, size=17, bold=True, max_width=140)
        for col, value, color, label in [
            (
                panel_x + 160,
                daic_value,
                COLORS["blue"],
                f4(daic_value) if metric != "Shifted feature share" else f"{pct(daic_value)} (550/768)",
            ),
            (
                panel_x + 450,
                task_value,
                COLORS["teal"],
                f4(task_value) if metric != "Shifted feature share" else f"{pct(task_value)} (292/768)",
            ),
        ]:
            bar_w = 210 * min(1.0, value / maximum)
            canvas.rect(col, y, 210, 28, fill="#edf1f3", stroke="#c4ccd2", radius=4)
            canvas.rect(col, y, bar_w, 28, fill=color, stroke=color, radius=4)
            canvas.text(col + 105, y + 42, label, size=14, anchor="middle")
    label_box(
        canvas,
        panel_x,
        660,
        590,
        110,
        "Available MMD^2",
        f'DAIC-to-Android MMD^2 = {f4(daic_shift["mmd2"])}. Android reading/interview MMD^2 = {f4(task_mmd2)}.',
        COLORS["light_gold"],
    )

    boundary_items_chart = [
        {"label": label, "value": value, "color": color}
        for label, value, color in boundary_items
    ]
    simple_bar_chart(
        canvas,
        840,
        220,
        570,
        420,
        boundary_items_chart,
        ymax=1.0,
        title_text="B. Classifier-boundary and score similarity",
        ylabel="Mean value",
    )
    label_box(
        canvas,
        850,
        710,
        520,
        130,
        "Interpretation boundary",
        "Moderate coefficient similarity and lower feature-overlap support partly task-specific classifier boundaries.",
        COLORS["light_blue"],
    )
    return "figure5_domain_shift_boundary_summary", canvas


def write_captions() -> None:
    CAPTIONS_PATH.write_text(
        """# BSPC Figure Captions

Status: figures_generated_awaiting_author_review

## Figure 1. Study design/protocol overview

Study design and protocol overview for the frozen WavLM layer-9 participant-level research pipeline. The schematic distinguishes DAIC-WOZ development, DAIC-to-Android interview transfer, Android reading evaluation and participant-coherent Android cross-task transfer. Source basis: `paper/FIGURE_SPECIFICATIONS.md`, `paper/BSPC_FIGURE_PRODUCTION_PLAN.md`, `paper/PAPER_RESULTS_TABLES.md` and `paper/PAPER_EVIDENCE_MAP.md`. Prohibited interpretation: do not imply that DAIC timing features were used in Android WavLM transfer or that the workflow is clinically deployable.

## Figure 2. DAIC Phase 1 key results

DAIC Phase 1F provisional local labeled test ROC-AUC for the frozen WavLM free-speech candidate and the guided-interview timing candidate. The WavLM candidate is lower on this DAIC-only local split but was retained for Android transfer because it is compatible with unconstrained speech; the timing candidate depends on guided-interview timing definitions. Source basis: `outputs/experiment_registry/experiment_summary.csv`. Prohibited interpretation: do not treat these as official hidden AVEC test results or as clinical validation.

## Figure 3. DAIC-to-Android transfer and adaptation

Android interview protocol comparison for frozen DAIC zero-shot scores, threshold-only adaptation, frozen-score calibration and Android-trained logistic regression. ROC-AUC and macro F1 are plotted from `records/runs/phase2c_20260703_125614/phase2c_protocol_comparison.csv`. Prohibited interpretation: do not label threshold-only adaptation or calibration as pure zero-shot, and do not describe Android-trained logistic regression as external validation.

## Figure 4. Android reading, interview and cross-task results

Android reading/interview task results and participant-coherent Phase 2E cross-task/pooled-task ROC-AUC values. Sources are `records/runs/phase2d_20260704_125931/phase2d_task_comparison.csv`, `records/runs/phase2e_20260704_151323/phase2e_protocol_comparison.csv` and `records/runs/phase2e_20260704_151323/phase2e_transfer_gap_analysis.csv`. Prohibited interpretation: do not claim pooled training is definitively superior from point estimates alone.

## Figure 5. Domain-shift and classifier-boundary summary

Aggregate domain-shift and classifier-boundary summaries. Panel A compares DAIC-to-Android feature shift with Android reading/interview task shift using centroid cosine distance, mean absolute standardized shift, shifted-feature share and MMD^2 values (0.6091 for DAIC-to-Android and 0.2864 for Android reading/interview). Panel B shows mean Phase 2E classifier-boundary and score-similarity metrics. Sources are `records/runs/phase2c_20260703_125614/PHASE2C_WITHIN_ANDROID_REPORT.json`, `records/runs/phase2d_20260704_125931/PHASE2D_ANDROID_READING_REPORT.json`, `records/runs/phase2e_20260704_151323/PHASE2E_CROSS_TASK_REPORT.json` and `paper/ieee_tcss/mmd2/android_reading_interview_mmd2.json`. Android reading/interview MMD^2 was computed from existing participant-level WavLM layer-9 embeddings using the same biased RBF MMD^2 estimator used for the DAIC-to-Android MMD^2 audit. Prohibited interpretation: do not treat coefficients as causal feature importance or claim a single mechanism explains domain shift.
""",
        encoding="utf-8",
    )


def validate_generated_text(paths: Iterable[Path]) -> None:
    violations: list[str] = []
    for path in paths:
        if path.suffix.lower() == ".png":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_GENERATED_TEXT_PATTERNS:
            if pattern.search(text):
                violations.append(f"{rel(path)} matched forbidden generated-text pattern {pattern.pattern}")
    if violations:
        raise SystemExit("Generated output privacy check failed:\n- " + "\n- ".join(violations))


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_inputs(INPUT_PATHS)

    registry_rows = read_csv(ROOT / "outputs" / "experiment_registry" / "experiment_summary.csv")
    phase2c_rows = read_csv(PHASE2C_DIR / "phase2c_protocol_comparison.csv")
    phase2c_report = read_json(PHASE2C_DIR / "PHASE2C_WITHIN_ANDROID_REPORT.json")
    phase2d_task_rows = read_csv(PHASE2D_DIR / "phase2d_task_comparison.csv")
    phase2d_report = read_json(PHASE2D_DIR / "PHASE2D_ANDROID_READING_REPORT.json")
    phase2e_rows = read_csv(PHASE2E_DIR / "phase2e_protocol_comparison.csv")
    phase2e_gap_rows = read_csv(PHASE2E_DIR / "phase2e_transfer_gap_analysis.csv")
    phase2e_report = read_json(PHASE2E_DIR / "PHASE2E_CROSS_TASK_REPORT.json")
    android_task_mmd2_report = read_json(ANDROID_TASK_MMD2_PATH)

    audit = Audit()
    figures = [
        draw_figure1(audit),
        draw_figure2(registry_rows, audit),
        draw_figure3(phase2c_rows, audit),
        draw_figure4(phase2d_task_rows, phase2e_rows, phase2e_gap_rows, audit),
        draw_figure5(phase2c_report, phase2d_report, phase2e_report, android_task_mmd2_report, audit),
    ]

    generated_paths: list[Path] = []
    for slug, artwork in figures:
        svg_path = OUTPUT_DIR / f"{slug}.svg"
        png_path = OUTPUT_DIR / f"{slug}.png"
        artwork.save_svg(svg_path)
        artwork.save_png(png_path)
        generated_paths.extend([svg_path, png_path])

    write_captions()
    audit.write()
    generated_paths.extend([CAPTIONS_PATH, AUDIT_PATH, SOURCE_VALUES_PATH])
    validate_generated_text(generated_paths)

    LOG_PATH.write_text(
        "\n".join(
            [
                "BSPC figure package generated successfully.",
                "Status: figures_generated_awaiting_author_review",
                "",
                "Input files:",
                *[f"- {rel(path)}" for path in INPUT_PATHS],
                "",
                "Output files:",
                *[f"- {rel(path)}" for path in generated_paths],
                f"- {rel(LOG_PATH)}",
                "",
                "Privacy scope: aggregate compact records only; no raw audio, participant identifiers, embeddings, fitted models or prediction files read.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(figures)} figures, {len(audit.rows)} audit rows and source manifest at {rel(SOURCE_VALUES_PATH)}.")


if __name__ == "__main__":
    main()
