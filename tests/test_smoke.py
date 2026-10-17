"""Check that the self-test needs only bundled files and protects prior output."""

from __future__ import annotations

import json

import numpy as np
import pytest

from infradetect_prototype.contract import default_project_root
from infradetect_prototype import smoke


def test_smoke_generates_four_channel_scene(tmp_path, monkeypatch):
    """A small synthetic scene must use bundled model paths and normal run API."""

    observed = {}

    def fake_run(request, *, quiet):
        """Inspect the request without loading the five real checkpoints."""

        observed["request"] = request
        observed["quiet"] = quiet
        return {"complete": True, "stage": "complete"}

    monkeypatch.setattr(smoke, "run_scene", fake_run)
    target = tmp_path / "new_smoke"
    result = smoke.run_smoke(default_project_root(), target)
    scene = np.load(target / "synthetic_scene.npy", allow_pickle=False)
    manifest = json.loads((target / "scene.json").read_text(encoding="utf-8"))
    assert scene.shape == (96, 128, 4)
    assert manifest["scene"]["source_dataset"].startswith("synthetic_smoke_only")
    assert manifest["models"]["rgb_defect"]["checkpoint"].startswith("models/")
    assert observed["request"].output == target / "result"
    assert observed["quiet"] is True
    assert result["complete"] and result["accuracy_evaluation"] is False


def test_smoke_refuses_nonempty_directory(tmp_path):
    """A repeated self-test may not overwrite a teammate's previous results."""

    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        smoke.run_smoke(default_project_root(), target)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
