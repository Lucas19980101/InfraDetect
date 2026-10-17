"""Check the four-channel scene contract and explicit optional-channel metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from infradetect.pipeline import inspect
from infradetect.evidence.fusion import _preview


def _manifest(tmp_path: Path, channels: int, extras: list[str] | None = None) -> Path:
    """Create a scene using bundled model files without loading their weights."""

    source = tmp_path / "scene.npy"
    np.save(source, np.zeros((6, 8, channels), dtype=np.uint8))
    definition = json.loads((ROOT / "infradetect/examples/scene.example.json").read_text())
    definition["scene"]["id"] = "four_channel_test"
    definition["scene"]["image"] = str(source)
    if extras is None:
        definition["scene"].pop("extra_channels", None)
    else:
        definition["scene"]["extra_channels"] = extras
    manifest = tmp_path / "scene.json"
    manifest.write_text(json.dumps(definition), encoding="utf-8")
    return manifest


def test_four_channel_scene_is_standard(tmp_path: Path):
    """A BGR+thermal stack runs preflight without a dummy Height plane."""

    definition, report = inspect(_manifest(tmp_path, 4), ROOT)
    assert definition["scene"]["image"]
    assert report["image_channels"] == 4
    assert report["extra_channels"] == []
    assert _preview(tmp_path / "scene.npy", (6, 8)).shape == (6, 8, 3)


def test_legacy_five_channel_scene_still_reads(tmp_path: Path):
    """Existing TBBR stacks remain usable while their fifth plane is ignored."""

    _, report = inspect(_manifest(tmp_path, 5), ROOT)
    assert report["extra_channels"] == ["height_legacy_unused"]
    # A user who copies the new four-channel example may leave [] in place.
    _, copied_template_report = inspect(_manifest(tmp_path, 5, []), ROOT)
    assert copied_template_report["extra_channels"] == ["height_legacy_unused"]


def test_new_extra_channels_need_names(tmp_path: Path):
    """Future per-pixel planes must be declared; current models still ignore them."""

    with pytest.raises(ValueError, match="extra_channels"):
        inspect(_manifest(tmp_path, 6), ROOT)
    _, report = inspect(_manifest(tmp_path, 6, ["height", "reflectance"]), ROOT)
    assert report["extra_channels"] == ["height", "reflectance"]


def test_extra_channel_count_must_match(tmp_path: Path):
    """Prevent a manifest from silently mislabeling the input plane order."""

    with pytest.raises(ValueError, match="extra_channels"):
        inspect(_manifest(tmp_path, 4, ["height"]), ROOT)


def test_capture_metadata_is_recorded_separately(tmp_path: Path):
    """Photo-level GPS/time/heading stays in provenance, outside model pixels."""

    manifest = _manifest(tmp_path, 4)
    definition = json.loads(manifest.read_text())
    definition["scene"]["capture_metadata"] = {
        "captured_at": "2026-10-16T10:00:00+08:00",
        "gps": {"latitude": 22.3, "longitude": 114.2},
        "heading_deg": 45.0,
    }
    manifest.write_text(json.dumps(definition), encoding="utf-8")
    _, report = inspect(manifest, ROOT)
    assert report["capture_metadata"] == definition["scene"]["capture_metadata"]
    assert report["image_channels"] == 4
