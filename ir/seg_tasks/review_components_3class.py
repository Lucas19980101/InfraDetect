"""Generate side-by-side RGB/label overlays from prepared three-class masks."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from seg_tasks.prepare_components_3class import hznu_mask


PALETTE = np.array([(0, 0, 255), (0, 255, 0), (255, 255, 0)], dtype=np.uint8)
EXIF_TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                  6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                  8: Image.Transpose.ROTATE_90}


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[index] for index in indices]


def select(rows: list[dict], source: str, count: int, names: list[str]) -> list[dict]:
    available = sorted((row for row in rows if row["source"] == source), key=lambda row: row["name"])
    if not available:
        raise ValueError(f"No rows for source {source}")
    if names:
        by_name = {row["name"]: row for row in available}
        missing = set(names) - set(by_name)
        if missing:
            raise ValueError(f"Unknown {source} image names: {sorted(missing)}")
        return [by_name[name] for name in names]
    if count < 1:
        raise ValueError("--count must be positive")
    if source == "hznu":
        camera = [row for row in available if row["name"].startswith("c_")]
        drone = [row for row in available if row["name"].startswith("d_")]
        first = min(len(camera), (count + 1) // 2)
        second = min(len(drone), count - first)
        chosen = evenly_spaced(camera, first) + evenly_spaced(drone, second)
        if len(chosen) < min(count, len(available)):
            remainder = [row for row in available if row not in chosen]
            chosen += evenly_spaced(remainder, min(count - len(chosen), len(remainder)))
        return chosen
    return evenly_spaced(available, min(count, len(available)))


def read_annotation(source_label: str) -> dict:
    if "::" in source_label:
        archive, member = source_label.split("::", 1)
        extracted = Path(archive).parent / member
        if extracted.is_file():
            return json.loads(extracted.read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive) as package:
            return json.loads(package.read(member))
    return json.loads(Path(source_label).read_text(encoding="utf-8"))


def original_image_path(source_label: str) -> Path:
    """Resolve a Hznu JPEG beside its source JSON in collected_datasets."""
    if "::" in source_label:
        archive, member = source_label.split("::", 1)
        extracted = Path(archive).parent / Path(member).with_suffix(".jpg")
        if extracted.is_file():
            return extracted
        raise FileNotFoundError(
            f"Extracted source JPEG required for review, but not found: {extracted}"
        )
    image = Path(source_label).with_suffix(".jpg")
    if not image.is_file():
        raise FileNotFoundError(f"Source JPEG not found beside JSON: {image}")
    return image


def preview(image_path: Path, mask_path: Path, max_size: tuple[int, int],
            annotation_source: str | None = None) -> Image.Image:
    with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
        orientation = image_file.getexif().get(274, 1)
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        if annotation_source is not None:
            annotation = read_annotation(annotation_source)
            mask = Image.fromarray(hznu_mask(annotation, image.size)[0])
        else:
            mask = mask_file.convert("L")
            if image_file.size != mask.size:
                raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
            if orientation in EXIF_TRANSPOSE:
                mask = mask.transpose(EXIF_TRANSPOSE[orientation])
        if image.size != mask.size:
            raise ValueError(f"EXIF-corrected image/mask size mismatch: {image_path}, {mask_path}")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    ids = np.asarray(mask)
    unexpected = set(np.unique(ids).tolist()) - {0, 1, 2, 255}
    if unexpected:
        raise ValueError(f"Unexpected mask values {unexpected}: {mask_path}")
    colors = Image.fromarray(PALETTE[np.minimum(ids, 2)])
    blended = np.asarray(Image.blend(image, colors, 0.4)).copy()
    blended[ids == 255] = np.asarray(image)[ids == 255]
    return Image.fromarray(blended)


def save_sheet(tiles: list[Image.Image], path: Path) -> None:
    sheet = Image.new("RGB", (max(tile.width for tile in tiles), sum(tile.height for tile in tiles)), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(path, quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/components_3class/manifest_with_splits.csv"))
    parser.add_argument("--source", choices=("hznu", "3dom", "cmp_base", "cmp_extended", "etrims8"), default="hznu")
    parser.add_argument("--count", type=int, default=20, help="Fixed, evenly spaced sample size")
    parser.add_argument("--all", action="store_true", help="Review every image in the selected source")
    parser.add_argument("--exif-only", action="store_true", help="Review the Hznu images listed by the EXIF orientation audit")
    parser.add_argument("--exif-pending", action="store_true",
                        help="Review only EXIF images neither accepted nor manually excluded")
    parser.add_argument("--excluded-only", action="store_true",
                        help="Review every excluded image for the selected source")
    parser.add_argument("--exif-audit", type=Path,
                        default=Path("data/components_3class/hznu_orientation_review.csv"))
    parser.add_argument("--exif-accepted", type=Path,
                        default=Path("data/components_3class/hznu_orientation_accepted.csv"))
    parser.add_argument("--exclusions", type=Path,
                        default=Path("data/components_3class/exclusions.csv"))
    parser.add_argument("--names", nargs="*", default=[], help="Exact stems to review instead of sampling")
    parser.add_argument("--output", type=Path, default=Path("data/components_3class/review"))
    args = parser.parse_args()
    if sum((bool(args.all), bool(args.names), bool(args.exif_only), bool(args.exif_pending),
            bool(args.excluded_only))) > 1:
        parser.error("--all, --names, --exif-only, --exif-pending and --excluded-only cannot be combined")
    if (args.exif_only or args.exif_pending or args.excluded_only) and args.source != "hznu":
        parser.error("Source-coordinate review modes are currently available only for --source hznu")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.excluded_only:
        chosen = sorted((row for row in rows if row["source"] == args.source and row["split"] == "exclude"),
                        key=lambda row: row["name"])
    elif args.exif_only or args.exif_pending:
        with args.exif_audit.open(newline="", encoding="utf-8") as handle:
            names = [row["name"] for row in csv.DictReader(handle) if row["source"] == "hznu"]
        if args.exif_pending:
            with args.exif_accepted.open(newline="", encoding="utf-8") as handle:
                accepted = {row["name"] for row in csv.DictReader(handle) if row["source"] == "hznu"}
            with args.exclusions.open(newline="", encoding="utf-8") as handle:
                excluded = {row["name"] for row in csv.DictReader(handle) if row["source"] == "hznu"}
            names = [name for name in names if name not in accepted and name not in excluded]
        chosen = select(rows, args.source, len(names), names)
    elif args.all:
        chosen = sorted((row for row in rows if row["source"] == args.source), key=lambda row: row["name"])
    else:
        chosen = select(rows, args.source, args.count, args.names)
    if not chosen:
        parser.error(f"No images found for source {args.source}")
    folder = args.output / args.source
    folder.mkdir(parents=True, exist_ok=True)
    tiles = []
    pages = 0
    for row in chosen:
        mask_path = Path(row["mask"])
        source_review = args.exif_only or args.exif_pending or args.excluded_only
        image_path = original_image_path(row["source_label"]) if source_review else Path(row["image"])
        with Image.open(image_path) as file:
            original = ImageOps.exif_transpose(file).convert("RGB")
            original.thumbnail((640, 480), Image.Resampling.LANCZOS)
        annotation_source = row["source_label"] if source_review else None
        overlaid = preview(image_path, mask_path, (640, 480), annotation_source)
        panel = Image.new("RGB", (1280, 520), "white")
        panel.paste(original, ((640 - original.width) // 2, 40 + (480 - original.height) // 2))
        panel.paste(overlaid, (640 + (640 - overlaid.width) // 2, 40 + (480 - overlaid.height) // 2))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), f"{args.source}/{row['name']} - original", fill="black")
        overlay_source = "source JSON" if source_review else "prepared mask"
        draw.text((652, 12), f"{overlay_source}: blue=surface, green=opening, yellow=other, plain=ignore", fill="black")
        panel.save(folder / f"{row['name']}.jpg", quality=90)
        thumb = panel.copy()
        thumb.thumbnail((960, 390), Image.Resampling.LANCZOS)
        tiles.append(thumb)
        if len(tiles) == 20:
            pages += 1
            name = "contact_sheet.jpg" if not (args.all or source_review) and len(chosen) <= 20 else f"contact_sheet_{pages:03d}.jpg"
            save_sheet(tiles, folder / name)
            tiles.clear()
    if tiles:
        pages += 1
        name = "contact_sheet.jpg" if not (args.all or source_review) and len(chosen) <= 20 else f"contact_sheet_{pages:03d}.jpg"
        save_sheet(tiles, folder / name)
    print(f"Saved {len(chosen)} paired overlays to {folder}")
    print(f"Contact sheets: {pages}")
    if not args.all and not args.exif_only and not args.exif_pending and not args.excluded_only:
        print("Names:", ", ".join(row["name"] for row in chosen))


if __name__ == "__main__":
    main()
