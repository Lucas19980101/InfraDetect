"""Run a self-contained, synthetic end-to-end check of the delivered 2D system.

The generated image is deliberately not a building dataset. Its purpose is to
exercise bundled weights, manifest resolution, nine pipeline stages and outputs
without requiring files from the original InfraDetect development checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import SceneRequest
from .engine import run_scene


def run_smoke(root: Path, output: Path) -> dict:
    """Create a tiny paired scene in a new directory and run all 2D stages.

    Inputs are only the delivered folder, its Python environment and an empty
    output location. The scene's four channels share the same pixel grid, so
    the bundled identity transform is valid for this synthetic check only.
    A completed run demonstrates wiring, not defect-detection accuracy.
    """

    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    template = root / "infradetect" / "examples" / "scene.example.json"
    if not template.is_file():
        raise FileNotFoundError(f"Bundled scene template missing: {template}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new empty smoke output directory: {output}")

    # A small constant image keeps the check fast and reproducible on CPU.
    # Its numbers carry no thermal or defect-ground-truth interpretation.
    scene = np.zeros((96, 128, 4), dtype=np.uint8)
    scene[..., 0] = 90   # B
    scene[..., 1] = 110  # G
    scene[..., 2] = 130  # R
    scene[..., 3] = 125  # synthetic Thermal
    definition = json.loads(template.read_text(encoding="utf-8"))
    definition["scene"].update(
        id="synthetic_smoke",
        source_dataset="synthetic_smoke_only_not_accuracy_evaluation",
        image=str(output / "synthetic_scene.npy"),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "synthetic_scene.npy", scene, allow_pickle=False)
    manifest = output / "scene.json"
    manifest.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")

    # The normal run path must load all five weights and complete its stages.
    result = run_scene(SceneRequest(root, manifest, output / "result"), quiet=True)
    return {"synthetic_only": True, "accuracy_evaluation": False,
            "manifest": str(manifest), "output": str(output / "result"),
            "complete": result.get("complete") is True, "stage": result.get("stage")}
