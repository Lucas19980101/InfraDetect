"""Render strict 3D defect points back into their RGB and IR source views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.io import loadmat

from .project_colmap_fragment import (DEFECT_COLORS, apply_homography,
                                      project_original, visible_indices)


def read_enriched_ply(path: Path) -> np.memmap:
    type_map = {"float": "<f4", "uchar": "u1", "ushort": "<u2"}
    with path.open("rb") as handle:
        lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.decode("ascii").strip()
            lines.append(text)
            if text == "end_header":
                break
        offset = handle.tell()
    count = int(next(line.rsplit(" ", 1)[1] for line in lines
                     if line.startswith("element vertex ")))
    fields = []
    for line in lines:
        if line.startswith("property "):
            _, kind, name = line.split()
            if kind not in type_map:
                raise ValueError(f"Unsupported PLY property: {line}")
            fields.append((name, type_map[kind]))
    required = {"x", "y", "z", "status", "defect_class"}
    if not required.issubset(name for name, _ in fields):
        raise ValueError(f"PLY lacks required fields: {required}")
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset,
                     shape=(count,))


def draw_points(image: np.ndarray, x: np.ndarray, y: np.ndarray,
                classes: np.ndarray, radius: int) -> np.ndarray:
    result = image.copy()
    for px, py, class_id in zip(x, y, classes):
        color_rgb = DEFECT_COLORS[class_id - 1] if 1 <= class_id <= 5 else np.asarray((255, 40, 40))
        color_bgr = tuple(map(int, color_rgb[::-1]))
        cv2.circle(result, (int(round(px)), int(round(py))), radius, color_bgr, -1,
                   lineType=cv2.LINE_AA)
    return result


def title(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(result, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def run(args: argparse.Namespace) -> dict:
    original = pycolmap.Reconstruction(args.model)
    dense = pycolmap.Reconstruction(args.workspace / "sparse")
    cloud = read_enriched_ply(args.ply)
    points = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float64)
    strict = cloud["status"] == 1
    records = []

    for image in sorted(dense.images.values(), key=lambda item: item.name):
        stem = Path(image.name).stem
        rgb = cv2.imread(str(args.rgb / image.name), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(args.ir / f"{stem}.png"), cv2.IMREAD_COLOR)
        status = cv2.imread(str(args.fusion / stem / "defect_status.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or status is None:
            raise FileNotFoundError(f"Missing RGB/IR/status input for {image.name}")
        depth_map = pycolmap.DepthMap()
        depth_map.read(args.workspace / "stereo" / "depth_maps" / f"{image.name}.geometric.bin")
        depth = depth_map.to_array()
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        visible, xyz = visible_indices(points, pose, dense.cameras[image.camera_id],
                                       depth, args.depth_tolerance)
        selected_mask = strict[visible]
        selected = visible[selected_mask]
        selected_xyz = xyz[selected_mask]
        original_image = original.images[image.image_id]
        u, v = project_original(original.cameras[original_image.camera_id], selected_xyz)
        matrix = np.asarray(loadmat(args.homographies / f"{stem}.mat")["H"], dtype=np.float64)
        ir_x, ir_y, finite = apply_homography(u, v, matrix)
        inside = (finite & (u >= 0) & (v >= 0) & (u < rgb.shape[1]) & (v < rgb.shape[0])
                  & (ir_x >= 0) & (ir_y >= 0) & (ir_x < ir.shape[1]) & (ir_y < ir.shape[0]))
        selected, u, v, ir_x, ir_y = (value[inside] for value in
                                      (selected, u, v, ir_x, ir_y))
        class_ids = cloud["defect_class"][selected]
        ix = np.rint(ir_x).astype(np.intp).clip(0, status.shape[1] - 1)
        iy = np.rint(ir_y).astype(np.intp).clip(0, status.shape[0] - 1)
        current_confirmed = status[iy, ix] == 1
        records.append({
            "image": image.name,
            "visible_strict_points": int(len(selected)),
            "currently_confirmed_points": int(current_confirmed.sum()),
            "current_view_agreement": float(current_confirmed.mean()) if len(selected) else 0.0,
            "u": u, "v": v, "ir_x": ir_x, "ir_y": ir_y,
            "classes": class_ids, "rgb": rgb, "ir": ir,
        })

    chosen = sorted(records, key=lambda item: item["visible_strict_points"], reverse=True)
    if args.images:
        chosen = chosen[:args.images]
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = []
    public_records = []
    for item in chosen:
        stem = Path(item["image"]).stem
        rgb_overlay = draw_points(item["rgb"], item["u"], item["v"],
                                  item["classes"], args.radius)
        ir_overlay = draw_points(item["ir"], item["ir_x"], item["ir_y"],
                                 item["classes"], args.radius)
        status_overlay = cv2.imread(str(args.fusion / stem / "final_status_overlay.png"))
        panels = [title(rgb_overlay, f"RGB <- strict 3D ({len(item['u'])} points)"),
                  title(ir_overlay, "IR <- strict 3D"),
                  title(status_overlay, "2D fusion status")]
        canvas = np.hstack(panels)
        cv2.imwrite(str(args.output / f"{stem}_audit.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        tiles.append(cv2.resize(canvas, (960, 256), interpolation=cv2.INTER_AREA))
        public_records.append({key: item[key] for key in
                               ("image", "visible_strict_points",
                                "currently_confirmed_points", "current_view_agreement")})
    if tiles:
        columns = 2
        blank = np.full_like(tiles[0], 255)
        while len(tiles) % columns:
            tiles.append(blank)
        rows = [np.hstack(tiles[index:index + columns])
                for index in range(0, len(tiles), columns)]
        cv2.imwrite(str(args.output / "contact_sheet.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = {"point_cloud": str(args.ply.resolve()), "audited_images": len(public_records),
              "images": public_records}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--homographies", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.images < 1 or args.radius < 1 or args.depth_tolerance <= 0:
        parser.error("Invalid audit settings")
    run(args)


if __name__ == "__main__":
    main()
