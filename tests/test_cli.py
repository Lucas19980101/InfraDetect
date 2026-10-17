"""Regression checks for command dispatch without loading model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infradetect_prototype import cli
from infradetect_prototype import engine
from infradetect_prototype.contract import SceneRequest
from infradetect_prototype.geometry import ColmapGeometry, aggregate_points


def test_check_uses_formal_root_and_does_not_write(tmp_path: Path, monkeypatch):
    """Preflight runs the formal package in a read-only subprocess."""

    root = tmp_path / "project"
    (root / "infradetect").mkdir(parents=True)
    (root / "infradetect" / "pipeline.py").write_text("", encoding="utf-8")
    manifest = tmp_path / "scene.json"
    manifest.write_text("{}", encoding="utf-8")
    observed = {}

    def fake_run(command, **kwargs):
        """Capture the child-process contract without invoking models."""

        observed.update({"command": command, **kwargs})
        return SimpleNamespace(stdout=json.dumps({"scene_id": "sample"}))

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    result = engine.run_scene(SceneRequest(root, manifest, tmp_path / "unused"), check_only=True)
    assert result["scene_id"] == "sample"
    assert observed["cwd"] == root
    assert observed["command"][1:3] == ["-m", "infradetect"]
    assert "--check" in observed["command"]
    assert not (tmp_path / "unused").exists()


def test_run_requires_explicit_output():
    """A model run cannot accidentally write to a guessed directory."""

    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--manifest", "scene.json"])
    assert error.value.code == 2


def test_geometry_refuses_overwrite(tmp_path: Path):
    """Grouping retains prior candidate instances when output is nonempty."""

    ply = tmp_path / "cloud.ply"
    summary = tmp_path / "summary.json"
    ply.write_bytes(b"ply")
    summary.write_text("{}", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("safe", encoding="utf-8")
    with pytest.raises(FileExistsError):
        aggregate_points(root=tmp_path, projected_ply=ply,
                         projection_summary=summary, output=output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "safe"


def test_colmap_geometry_interface_reports_readiness(tmp_path: Path):
    """A lone PLY is insufficient: projection also needs cameras and depth."""

    model = tmp_path / "sparse"
    workspace = tmp_path / "dense"
    ply = workspace / "fused.ply"
    assert not ColmapGeometry(model, workspace, ply).inspect()["ready_for_projection"]
    for directory in (model, workspace / "sparse"):
        directory.mkdir(parents=True)
        for name in ("cameras", "images", "points3D"):
            (directory / f"{name}.bin").write_bytes(b"test")
    depth = workspace / "stereo" / "depth_maps"
    depth.mkdir(parents=True)
    (depth / "frame.geometric.bin").write_bytes(b"test")
    ply.write_bytes(b"ply\n")
    report = ColmapGeometry(model, workspace, ply).inspect()
    assert report["ready_for_projection"] is True
    assert report["geometric_depth_map_count"] == 1
