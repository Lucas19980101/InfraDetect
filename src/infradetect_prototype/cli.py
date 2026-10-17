"""One documented command-line entry for the shareable prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import SceneRequest, default_project_root
from .engine import run_scene
from .geometry import ColmapGeometry, aggregate_points, compare_views, project_points
from .report import summarize_run


def _path(value: str) -> Path:
    """Resolve user paths before changing working directory for a stage."""

    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Declare only stable, team-facing commands and their required inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=_path, default=default_project_root(),
                        help="Root of this standalone prototype delivery folder")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Run all 2D stages on a synthetic scene")
    smoke.add_argument("--output", type=_path, required=True,
                       help="New empty directory for synthetic input and results")
    for name in ("check", "run"):
        scene = commands.add_parser(name, help=f"{name} one paired RGB/thermal scene")
        scene.add_argument("--manifest", type=_path, required=True)
        scene.add_argument("--output", type=_path,
                           help="New result directory; required for run, unused by check")
    summary = commands.add_parser("summary", help="Summarize an existing scene run")
    summary.add_argument("--output", type=_path, required=True)
    projection = commands.add_parser("project-3d", help="Project evidence to a dense COLMAP cloud")
    geometry = commands.add_parser("geometry-check", help="Inspect COLMAP geometry without inference")
    for flag in ("model", "workspace", "ply"):
        geometry.add_argument(f"--{flag}", type=_path, required=True)
    baseline = commands.add_parser("geometry-baseline", help="Score deviations from a dominant wall plane")
    baseline.add_argument("--ply", type=_path, required=True, help="A cropped single-wall PLY cloud")
    baseline.add_argument("--output", type=_path, required=True, help="New empty result directory")
    baseline.add_argument("--fit-fraction", type=float, default=0.002)
    baseline.add_argument("--candidate-fraction", type=float, default=0.006)
    for flag in ("model", "workspace", "ply", "fusion", "homographies", "output"):
        projection.add_argument(f"--{flag}", type=_path, required=True)
    projection.add_argument("--ir-config", type=_path)
    grouping = commands.add_parser("aggregate-3d", help="Group candidate cloud points")
    grouping.add_argument("--projected-ply", type=_path, required=True)
    grouping.add_argument("--projection-summary", type=_path, required=True)
    grouping.add_argument("--output", type=_path, required=True)
    views = commands.add_parser("compare-views", help="Check evidence across visible 3D views")
    for flag in ("projected-ply", "projection-summary", "original-model",
                 "dense-workspace", "fusion-root", "correspondence-root", "output"):
        views.add_argument(f"--{flag}", type=_path, required=True)
    views.add_argument("--membership-file", type=_path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch a command and print a concise, machine-readable result."""

    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root
    if args.command == "smoke":
        from .smoke import run_smoke

        result = run_smoke(root, args.output)
    elif args.command in ("check", "run"):
        if args.command == "run" and args.output is None:
            parser.error("run requires --output")
        # The underlying preflight never writes to its output path.
        output = args.output or root / "outputs" / "unused_preflight_output"
        result = run_scene(SceneRequest(root, args.manifest, output),
                           check_only=args.command == "check")
    elif args.command == "summary":
        result = summarize_run(args.output)
    elif args.command == "geometry-check":
        result = ColmapGeometry(args.model, args.workspace, args.ply).inspect()
    elif args.command == "geometry-baseline":
        # The lightweight CLI is installable on its own; load the sibling
        # geometry package only when this optional operation is requested.
        sys.path.insert(0, str(root))
        from infradetect.geometry.baseline import run_wall_baseline
        result = run_wall_baseline(ply=args.ply, output=args.output,
                                   fit_fraction=args.fit_fraction,
                                   candidate_fraction=args.candidate_fraction)
    elif args.command == "project-3d":
        project_points(root=root, model=args.model, workspace=args.workspace,
                       ply=args.ply, fusion=args.fusion,
                       homographies=args.homographies, output=args.output,
                       ir_config=args.ir_config)
        result = {"stage": "project-3d", "output": str(args.output)}
    elif args.command == "aggregate-3d":
        aggregate_points(root=root, projected_ply=args.projected_ply,
                         projection_summary=args.projection_summary, output=args.output)
        result = {"stage": "aggregate-3d", "output": str(args.output)}
    else:
        compare_views(root=root, projected_ply=args.projected_ply,
                      projection_summary=args.projection_summary,
                      original_model=args.original_model,
                      dense_workspace=args.dense_workspace,
                      fusion_root=args.fusion_root,
                      correspondence_root=args.correspondence_root,
                      membership_file=args.membership_file, output=args.output)
        result = {"stage": "compare-views", "output": str(args.output)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
