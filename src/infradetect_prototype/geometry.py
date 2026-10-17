"""Validated gateways to existing COLMAP projection and multi-view modules.

Geometry is optional: a single RGB/thermal pair has no defensible 3D position.
All three stages use their original implementations so pixel-coordinate and
occlusion logic stays in one maintained place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class ColmapGeometry:
    """One COLMAP reconstruction: original cameras and dense outputs together.

    These three paths must come from the same reconstruction. The checker below
    verifies file presence only; it cannot prove coordinate-frame consistency.
    """

    model: Path
    workspace: Path
    ply: Path

    def inspect(self) -> dict[str, object]:
        """Check geometry inputs without loading a cloud or running a model."""

        def has_model(directory: Path) -> bool:
            """Accept either a complete binary or text COLMAP sparse model."""

            names = ("cameras", "images", "points3D")
            return any(all((directory / f"{name}.{suffix}").is_file()
                           for name in names) for suffix in ("bin", "txt"))

        depth_dir = self.workspace / "stereo" / "depth_maps"
        depth_count = sum(1 for _ in depth_dir.glob("*.geometric.bin")) if depth_dir.is_dir() else 0
        checks = {
            "original_sparse_model": has_model(self.model),
            "dense_sparse_model": has_model(self.workspace / "sparse"),
            "geometric_depth_maps": depth_count > 0,
            "fused_ply": self.ply.is_file() and self.ply.stat().st_size > 0,
        }
        return {
            "backend": "colmap",
            "ready_for_projection": all(checks.values()),
            "checks": checks,
            "geometric_depth_map_count": depth_count,
            "paths": {"model": str(self.model), "workspace": str(self.workspace),
                      "ply": str(self.ply)},
            "note": "只检查文件结构；相机、深度图和点云是否同源仍须核对。",
        }


def _required(paths: dict[str, Path]) -> None:
    """Reject missing geometry inputs before starting an expensive stage."""

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")


def _empty_output(output: Path) -> None:
    """Preserve existing experiments by requiring a new/empty output directory."""

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new empty output directory: {output}")


def _call(module: str, args: list[str], root: Path) -> None:
    """Launch one packaged geometry module using the caller's Python environment."""

    if not (root / "infradetect" / "geometry").is_dir():
        raise FileNotFoundError(f"InfraDetect project root not found: {root}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root), environment.get("PYTHONPATH", "")) if part
    )
    subprocess.run([sys.executable, "-m", module, *args], cwd=root,
                   env=environment, check=True)


def project_points(*, root: Path, model: Path, workspace: Path, ply: Path,
                   fusion: Path, homographies: Path, output: Path,
                   ir_config: Path | None = None) -> None:
    """Project 2D evidence onto an existing dense cloud with depth checks."""

    root = root.resolve()
    inputs = {"COLMAP model": model, "dense workspace": workspace,
              "fused cloud": ply, "per-image fusion": fusion,
              "RGB-to-IR transforms": homographies}
    if ir_config is not None:
        inputs["IR config"] = ir_config
    _required(inputs)
    _empty_output(output)
    args = ["--model", str(model), "--workspace", str(workspace),
            "--ply", str(ply), "--fusion", str(fusion),
            "--homographies", str(homographies), "--output", str(output)]
    if ir_config is not None:
        args.extend(["--ir-config", str(ir_config)])
    _call("infradetect.geometry.project", args, root)


def aggregate_points(*, root: Path, projected_ply: Path,
                     projection_summary: Path, output: Path) -> None:
    """Group projected candidate points into reviewable 3D instances."""

    _required({"projected cloud": projected_ply,
               "projection summary": projection_summary})
    _empty_output(output)
    _call("infradetect.geometry.aggregate",
          ["--ply", str(projected_ply), "--projection-summary",
           str(projection_summary), "--output", str(output)],
          root.resolve())


def compare_views(*, root: Path, projected_ply: Path,
                  projection_summary: Path, original_model: Path,
                  dense_workspace: Path, fusion_root: Path,
                  correspondence_root: Path, output: Path,
                  membership_file: Path | None = None) -> None:
    """Check independent views, occlusion, and RGB/IR correspondence in 3D."""

    inputs = {"projected cloud": projected_ply,
              "projection summary": projection_summary,
              "original COLMAP model": original_model,
              "dense workspace": dense_workspace,
              "per-image fusion": fusion_root,
              "per-image correspondence": correspondence_root}
    if membership_file is not None:
        inputs["candidate memberships"] = membership_file
    _required(inputs)
    _empty_output(output)
    args = ["--projected-ply", str(projected_ply),
            "--projection-summary", str(projection_summary),
            "--original-model", str(original_model),
            "--dense-workspace", str(dense_workspace),
            "--fusion-root", str(fusion_root),
            "--correspondence-root", str(correspondence_root),
            "--output", str(output)]
    if membership_file is not None:
        args.extend(["--membership-file", str(membership_file)])
    _call("infradetect.geometry.multiview", args, root.resolve())
