"""Checks for safe CLI inputs and run-result interpretation."""

import json
from pathlib import Path

import pytest

from infradetect_prototype.contract import SceneRequest, default_project_root
from infradetect_prototype.report import summarize_run


def test_default_root_contains_pipeline():
    """The installed-in-repository package resolves the formal pipeline."""

    assert (default_project_root() / "infradetect" / "pipeline.py").is_file()


def test_run_refuses_existing_output(tmp_path: Path):
    """A shareable CLI must never overwrite an experiment by accident."""

    manifest = tmp_path / "scene.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "result"
    output.mkdir()
    (output / "keep.txt").write_text("original", encoding="utf-8")
    request = SceneRequest(default_project_root(), manifest, output)
    with pytest.raises(FileExistsError):
        request.checked(will_write=True)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "original"
    assert request.checked(will_write=False).output == output


def test_summary_does_not_invent_3d_or_accuracy(tmp_path: Path):
    """A complete 2D run is not automatically a 3D or accuracy result."""

    (tmp_path / "status.json").write_text(
        json.dumps({"complete": True, "stage": "complete"}), encoding="utf-8")
    (tmp_path / "report.json").write_text(
        json.dumps({"scene": {"scene_id": "demo", "source_dataset": "synthetic",
                              "registration_accepted": False}}), encoding="utf-8")
    result = summarize_run(tmp_path)
    assert result["complete"] is True
    assert result["three_dimensional_projection"] == "not_run"
    assert "accuracy" not in result


def test_summary_reports_failure_without_report(tmp_path: Path):
    """An interrupted run can still be inspected through status.json."""

    (tmp_path / "status.json").write_text(
        json.dumps({"complete": False, "stage": "fusion", "error": "failed"}),
        encoding="utf-8")
    result = summarize_run(tmp_path)
    assert result["stage"] == "fusion"
    assert result["error"] == "failed"
