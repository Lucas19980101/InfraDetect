#!/usr/bin/env python3
"""Query the CSV RGB-to-3D correspondence export without extra dependencies."""

import argparse
import csv
from pathlib import Path


def latest_run(root: Path) -> Path:
    marker = root / "Log/correspondence/latest_run.txt"
    if not marker.is_file():
        raise FileNotFoundError("No correspondence run found; replay a bag with correspondence/save_en=true")
    return Path(marker.read_text().strip())


def load_by_id(path: Path, key: str):
    with path.open(newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="correspondence run directory; defaults to latest_run.txt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    point = subparsers.add_parser("point", help="list image observations for one 3D point ID")
    point.add_argument("point_id")
    region = subparsers.add_parser("region", help="list 3D points projected into an image rectangle")
    region.add_argument("frame_id")
    region.add_argument("x0", type=float)
    region.add_argument("y0", type=float)
    region.add_argument("x1", type=float)
    region.add_argument("y1", type=float)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run = args.run or latest_run(repo)
    frames = load_by_id(run / "frames.csv", "frame_id")
    points = load_by_id(run / "points.csv", "point_id")

    with (run / "observations.csv").open(newline="") as handle:
        observations = csv.DictReader(handle)
        if args.command == "point":
            for obs in observations:
                if obs["point_id"] == args.point_id:
                    print({**points[args.point_id], **frames[obs["frame_id"]], **obs})
        else:
            x0, x1 = sorted((args.x0, args.x1))
            y0, y1 = sorted((args.y0, args.y1))
            for obs in observations:
                if (obs["frame_id"] == args.frame_id and x0 <= float(obs["u"]) <= x1
                        and y0 <= float(obs["v"]) <= y1):
                    print({**points[obs["point_id"]], **obs})


if __name__ == "__main__":
    main()
