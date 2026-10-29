"""JSON and standalone HTML reports (one HTML file, no external assets, works offline)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .llm import MODEL
from .pipeline import RunResult
from .qa_checks import CHECK_LABELS

TEMPLATE = Path(__file__).parent / "templates" / "report.html"


def build_report(runs: list[RunResult], feedback: dict | None = None, demo: bool = False,
                 settings: dict | None = None) -> dict:
    return {
        "tool": "AI Localization QA",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "demo": demo,
        "model": "demo (canned responses, no API calls)" if demo else MODEL,
        "settings": settings or {},
        "check_labels": CHECK_LABELS,
        "runs": [r.to_dict() for r in runs],
        "feedback": feedback,
    }


def write_json(report: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html(report: dict, path: Path) -> None:
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__REPORT_DATA__", data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
