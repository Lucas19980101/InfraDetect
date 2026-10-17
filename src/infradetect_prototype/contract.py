"""Typed input/output contracts and side-effect-free path checks.

The authoritative scene-manifest validation remains in ``infradetect.pipeline``;
these checks catch confusing command-line mistakes before a model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneRequest:
    """Inputs needed to run one already-paired RGB/thermal scene."""

    project_root: Path
    manifest: Path
    output: Path

    def checked(self, *, will_write: bool) -> "SceneRequest":
        """Resolve paths and reject missing inputs or accidental run overwrite."""

        root = self.project_root.expanduser().resolve()
        manifest = self.manifest.expanduser().resolve()
        output = self.output.expanduser().resolve()
        if not (root / "infradetect" / "pipeline.py").is_file():
            raise FileNotFoundError(f"InfraDetect project root not found: {root}")
        if not manifest.is_file():
            raise FileNotFoundError(f"Scene manifest not found: {manifest}")
        if will_write and output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Choose a new empty output directory: {output}")
        return SceneRequest(root, manifest, output)


def default_project_root() -> Path:
    """Locate the containing InfraDetect checkout from this package's source."""

    # Layout: <delivery>/src/infradetect_prototype/contract.py.
    return Path(__file__).resolve().parents[2]
