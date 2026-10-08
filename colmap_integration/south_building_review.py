#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Serve a local-only review UI for south-building RGB/COLMAP candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pycolmap


DEFAULT_DATASET = Path("../collected_datasets/south-building")
DEFAULT_RUN = Path("benchmark/reconstruction/runs/south-building-rgb-tiles")
VALID_STATUSES = {"confirmed", "rejected", "uncertain", "unreviewed"}


def build_payload(dataset: Path, run_dir: Path) -> dict:
    reconstruction = pycolmap.Reconstruction(dataset / "sparse")
    with (run_dir / "points.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = []
    for row in rows:
        if row["crack"] != "1" and row["spalling"] != "1":
            continue
        point_id = int(row["point3D_id"])
        point = reconstruction.points3D[point_id]
        observations = []
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            xy = image.points2D[element.point2D_idx].xy
            cache_path = run_dir / "observations" / f"{image.name}.npz"
            with np.load(cache_path) as cache:
                indices = np.flatnonzero(cache["point_ids"] == point_id)
                if not len(indices):
                    continue
                probability = cache["values"][int(indices[0])].tolist()
            camera = reconstruction.cameras[image.camera_id]
            observations.append({
                "image": image.name,
                "x": round(float(xy[0]), 2),
                "y": round(float(xy[1]), 2),
                "width": camera.width,
                "height": camera.height,
                "crack": round(float(probability[0]), 4),
                "spalling": round(float(probability[1]), 4),
            })
        observations.sort(key=lambda item: item["image"])
        candidates.append({
            "id": point_id,
            "xyz": [float(value) for value in point.xyz],
            "class": "spalling" if row["spalling"] == "1" else "crack",
            "crack": round(float(row["crack_median"]), 4),
            "spalling": round(float(row["spalling_median"]), 4),
            "views": int(row["valid_views"]),
            "observations": observations,
        })
    candidates.sort(key=lambda point: (point["class"], -point[point["class"]], point["id"]))
    # The full 61k-point model is available in the PLY. A deterministic sample
    # keeps the browser review canvas responsive while retaining its 3D shape.
    background = [
        [*map(float, point.xyz), *map(int, point.color)]
        for index, (_, point) in enumerate(sorted(reconstruction.points3D.items()))
        if index % 5 == 0
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"background": background, "candidates": candidates, "summary": summary}


def save_reviews(path: Path, reviews: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".reviews-", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(reviews, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    run_dir = args.run_dir.resolve()
    payload = build_payload(dataset, run_dir)
    valid_ids = {point["id"] for point in payload["candidates"]}
    valid_images = {observation["image"] for point in payload["candidates"] for observation in point["observations"]}
    review_path = run_dir / "reviews.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    token = secrets.token_urlsafe(24)
    html_path = Path(__file__).with_name("south_building_review.html")

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                html = html_path.read_bytes().replace(b"__REVIEW_TOKEN__", token.encode())
                self.send_bytes(html, "text/html; charset=utf-8")
            elif path == "/api/data":
                self.send_bytes(json.dumps({**payload, "reviews": reviews}).encode(), "application/json")
            elif path.startswith("/images/") or path.startswith("/previews/"):
                name = unquote(path.split("/", 2)[2])
                if path.startswith("/previews/"):
                    name = name.removesuffix(".png")
                if name not in valid_images or Path(name).name != name:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                source = dataset / "images" / name if path.startswith("/images/") else run_dir / "previews" / f"{name}.png"
                if not source.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_bytes(source.read_bytes(), "image/jpeg" if source.suffix.lower() in {".jpg", ".jpeg"} else "image/png")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/review" or self.headers.get("X-Review-Token") != token:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                point_id = int(data["point_id"])
                status = data["status"]
                if point_id not in valid_ids or status not in VALID_STATUSES:
                    raise ValueError("Invalid point or status")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            key = str(point_id)
            if status == "unreviewed":
                reviews.pop(key, None)
            else:
                reviews[key] = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
            save_reviews(review_path, reviews)
            self.send_bytes(json.dumps({"point_id": point_id, "status": status}).encode(), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review UI: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Review labels: {review_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
