"""Generate source-JPEG/source-JSON overlays for every pair in a Hznu folder."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from seg_tasks.prepare_components_3class import COLORS, HZNU_MAP
from seg_tasks.review_components_3class import save_sheet


def make_panel(image_path: Path, annotation_path: Path, rotate_clockwise: bool = False) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if rotate_clockwise:
        image = image.transpose(Image.Transpose.ROTATE_270)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    mask = Image.new("L", image.size, 255)
    drawer = ImageDraw.Draw(mask)
    priority = {"building": 0, "sky": 1, "tree": 1, "car": 1, "window": 2, "door": 2}
    for shape in sorted(annotation.get("shapes", []),
                        key=lambda item: priority.get(item.get("label"), 3)):
        label = shape.get("label")
        points = shape.get("points", [])
        # Perspective-correction JSON contains zero-area contour fragments.
        if label not in HZNU_MAP or len(points) < 3:
            continue
        drawer.polygon([tuple(point) for point in points], fill=HZNU_MAP[label])

    image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    ids = np.asarray(mask)
    colored = Image.fromarray(COLORS[np.minimum(ids, 2)])
    blended = np.asarray(Image.blend(image, colored, 0.4)).copy()
    blended[ids == 255] = np.asarray(image)[ids == 255]
    overlay = Image.fromarray(blended)

    panel = Image.new("RGB", (1280, 520), "white")
    panel.paste(image, ((640 - image.width) // 2, 40 + (480 - image.height) // 2))
    panel.paste(overlay, (640 + (640 - overlay.width) // 2, 40 + (480 - overlay.height) // 2))
    draw = ImageDraw.Draw(panel)
    suffix = " (90deg CW diagnostic)" if rotate_clockwise else ""
    draw.text((12, 12), f"{image_path.stem} - source JPG{suffix}", fill="black")
    draw.text((652, 12), "source JSON: blue=surface, green=opening, yellow=other, plain=ignore",
              fill="black")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Folder containing same-stem .jpg and .json files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--names", nargs="*", default=[], help="Only process these exact stems")
    parser.add_argument("--rotate-cw", nargs="*", default=[],
                        help="Stems whose source pixels should be rotated 90 degrees clockwise")
    parser.add_argument("--skip-csv", nargs="*", type=Path, default=[],
                        help="CSV files whose name column contains already reviewed stems")
    args = parser.parse_args()
    if args.per_sheet < 1:
        parser.error("--per-sheet must be positive")

    images = sorted(args.input.glob("*.jpg"))
    labels = {path.stem: path for path in args.input.glob("*.json")}
    missing = [path.name for path in images if path.stem not in labels]
    extra = sorted(set(labels) - {path.stem for path in images})
    if missing or extra:
        raise ValueError(f"Unpaired Hznu files; missing JSON={missing[:10]}, extra JSON={extra[:10]}")
    if args.names:
        by_name = {path.stem: path for path in images}
        missing_names = sorted(set(args.names) - set(by_name))
        if missing_names:
            parser.error(f"Unknown image stems: {missing_names}")
        images = [by_name[name] for name in args.names]
    skipped: set[str] = set()
    for path in args.skip_csv:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "name" not in reader.fieldnames:
                raise ValueError(f"Skip CSV has no name column: {path}")
            skipped.update(row["name"] for row in reader)
    images = [path for path in images if path.stem not in skipped]
    if not images:
        parser.error(f"No JPG files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    pages = 0
    for image_path in images:
        panel = make_panel(image_path, labels[image_path.stem], image_path.stem in args.rotate_cw)
        panel.save(args.output / f"{image_path.stem}.jpg", quality=90)
        thumb = panel.copy()
        thumb.thumbnail((960, 390), Image.Resampling.LANCZOS)
        tiles.append(thumb)
        if len(tiles) == args.per_sheet:
            pages += 1
            save_sheet(tiles, args.output / f"contact_sheet_{pages:03d}.jpg")
            tiles.clear()
    if tiles:
        pages += 1
        save_sheet(tiles, args.output / f"contact_sheet_{pages:03d}.jpg")
    print(f"Saved {len(images)} source-based overlays to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} reviewed stems from {len(args.skip_csv)} CSV files")
    print(f"Contact sheets: {pages}")


if __name__ == "__main__":
    main()
