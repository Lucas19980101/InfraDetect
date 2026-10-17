"""Small, conservative summary of a completed integrated scene run.

This module intentionally avoids invented accuracy or engineering diagnoses:
model candidate counts and run completion are not ground-truth metrics.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    """Read a JSON object, rejecting malformed or non-object files."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summarize_run(output: Path) -> dict:
    """Report available stage evidence without interpreting it as a diagnosis."""

    output = output.expanduser().resolve()
    status = _read_json(output / "status.json")
    report_path = output / "report.json"
    report = _read_json(report_path) if report_path.is_file() else {}
    scene = report.get("scene") or {}
    return {
        "scene_id": scene.get("scene_id"),
        "source_dataset": scene.get("source_dataset"),
        "complete": status.get("complete") is True,
        "stage": status.get("stage"),
        "error": status.get("error"),
        "registration_accepted": scene.get("registration_accepted"),
        "two_dimensional_evidence": (output / "fusion").is_dir(),
        "cross_modal_evidence": (output / "cross_modal").is_dir(),
        "three_dimensional_projection": report.get("three_dimensional_projection", "not_run"),
        "report": str(report_path) if report_path.is_file() else None,
        "status_file": str(output / "status.json"),
        "interpretation": "Model evidence for human review; not verified defect ground truth",
    }
