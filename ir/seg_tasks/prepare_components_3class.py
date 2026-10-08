"""Prepare auditable three-class component masks from existing facade datasets.

Output IDs: 0 building_surface, 1 opening, 2 other, 255 ignore.
Does not edit source data or create a training configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

NAMES = {0: "building_surface", 1: "opening", 2: "other", 255: "ignore"}
IRREGULAR_3DOM_GRAY_LABELS = {
    "Navona_image123": "Navona_image123_lpng.png",
    "Navona_image147_undist": "Navona_image147_l_colored_undist.png",
    "Navona_image148_undist": "Navona_image148_l_colored_undist.png",
    "Navona_image149_undist": "Navona_image149_l_colored_undist.png",
    "Navona_image151_undist": "Navona_image151_l_colored_undist.png",
    "Navona_image152_undist": "Navona_image152_l_colored_undist.png",
    "image115i": "image115_l.png",
}
COLORS = np.array([[0, 0, 255], [0, 255, 0], [255, 255, 0]], dtype=np.uint8)
CMP_MAP = {2: 0, 5: 0, 6: 0, 9: 0, 10: 0, 11: 0,
           3: 1, 4: 1, 8: 1, 1: 255, 7: 255, 12: 255}
ETRIMS_MAP = {1: 0, 3: 1, 8: 1, 2: 2, 4: 2, 5: 2, 6: 2, 7: 2}
DOM_MAP = {1: 0, 2: 1, 3: 2, 4: 2, 5: 1}
HZNU_MAP = {"building": 0, "window": 1, "door": 1,
            "sky": 2, "tree": 2, "car": 2}


def indexed_values(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "P":
            raise ValueError(f"Expected indexed PNG, got {image.mode}: {path}")
        return np.asarray(image, dtype=np.uint8)


def remap(values: np.ndarray, mapping: dict[int, int], source: str) -> np.ndarray:
    unexpected = set(np.unique(values).tolist()) - set(mapping) - {0}
    if unexpected:
        raise ValueError(f"Unexpected {source} label IDs: {sorted(unexpected)}")
    mask = np.full(values.shape, 255, dtype=np.uint8)
    for old, new in mapping.items():
        mask[values == old] = new
    return mask


def dom_mask(path: Path, image_path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
        raise ValueError(f"3DOM gray channels differ: {path}")
    with Image.open(image_path) as image:
        size = image.size
    mask = remap(rgb[..., 0], DOM_MAP, "3DOM")
    if mask.shape[::-1] == size:
        return mask
    if mask.shape == size:
        rotated = Image.fromarray(mask).transpose(Image.Transpose.ROTATE_90)
        if rotated.size == size:
            return np.asarray(rotated)
    raise ValueError(f"3DOM size mismatch: {image_path}, {path}")


def hznu_mask(annotation: dict, size: tuple[int, int]) -> tuple[np.ndarray, Counter]:
    width, height = size
    declared = (annotation.get("imageWidth"), annotation.get("imageHeight"))
    if all(value is not None for value in declared) and declared != size:
        raise ValueError(f"Hznu JSON dimensions {declared} differ from image {size}")
    canvas = Image.new("L", size, 255)
    drawer = ImageDraw.Draw(canvas)
    counts: Counter = Counter()
    shapes = annotation.get("shapes", [])
    # Semantic priority: expose facade first, then exclusion regions, then openings.
    priority = {"building": 0, "sky": 1, "tree": 1, "car": 1, "window": 2, "door": 2}
    for shape in sorted(shapes, key=lambda item: priority.get(item.get("label"), 3)):
        label = shape.get("label")
        counts[label] += 1
        if label not in HZNU_MAP:
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Hznu polygon has fewer than 3 points: {label}")
        drawer.polygon([tuple(point) for point in points], fill=HZNU_MAP[label])
    return np.asarray(canvas, dtype=np.uint8), counts


def save_pair(mask: np.ndarray, image_path: Path, mask_path: Path, preview_path: Path) -> dict[str, int]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != mask.shape[::-1]:
        raise ValueError(f"Image/mask size mismatch: {image_path}")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    # Preview is downscaled only after the full-resolution label is saved.
    image.thumbnail((900, 650))
    preview_mask = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
    ids = np.asarray(preview_mask)
    colored = Image.fromarray(COLORS[np.minimum(ids, 2)])
    overlay = Image.blend(image, colored, 0.4)
    overlay_arr = np.asarray(overlay).copy()
    original_arr = np.asarray(image)
    overlay_arr[ids == 255] = original_arr[ids == 255]
    Image.fromarray(overlay_arr).save(preview_path, quality=88)
    return {NAMES[value]: int(np.count_nonzero(mask == value)) for value in NAMES}


def prepare(root: Path, output: Path, preview_limit: int) -> None:
    if preview_limit < 0:
        raise ValueError("preview_limit must be >= 0")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    anomalies: Counter = Counter()
    by_source: Counter = Counter()
    previews: Counter = Counter()

    def cached_mask(source: str, name: str, build) -> np.ndarray:
        path = output / "masks" / source / f"{name}.png"
        # Hznu JPEGs may use EXIF rotation while their JSON polygons use the
        # visually upright coordinate system. Always rebuild these masks so a
        # stale mask made against raw JPEG dimensions cannot survive a rerun.
        if path.exists() and source != "hznu":
            with Image.open(path) as saved:
                return np.asarray(saved, dtype=np.uint8)
        return build()

    def add(source: str, name: str, image: Path, original_label: str, mask: np.ndarray) -> None:
        mask_path = output / "masks" / source / f"{name}.png"
        preview_path = output / "previews" / source / f"{name}.jpg"
        counts = save_pair(mask, image, mask_path, preview_path) if previews[source] < preview_limit else None
        if counts is None:
            with Image.open(image) as picture:
                if picture.size != mask.shape[::-1]:
                    raise ValueError(f"Image/mask size mismatch: {image}")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_path)
            counts = {NAMES[value]: int(np.count_nonzero(mask == value)) for value in NAMES}
        else:
            previews[source] += 1
        rows.append({"source": source, "name": name, "image": str(image.resolve()),
                     "source_label": original_label, "mask": str(mask_path.resolve()), **counts})
        by_source[source] += 1

    dom = root / "3DOM"
    dom_images = {path.stem: path for path in (dom / "data").iterdir() if path.suffix.lower() == ".jpg"}
    dom_labels = {path.stem[:-2]: path for path in (dom / "labels_gray").glob("*_l.png")}
    dom_labels.update({name: dom / "labels_gray" / filename for name, filename in IRREGULAR_3DOM_GRAY_LABELS.items()})
    if missing := dom_images.keys() - dom_labels.keys():
        raise ValueError(f"3DOM images without labels: {sorted(missing)}")
    for name in sorted(dom_images):
        add("3dom", name, dom_images[name], str(dom_labels[name].resolve()),
            cached_mask("3dom", name, lambda name=name: dom_mask(dom_labels[name], dom_images[name])))

    for folder in ("base", "extended"):
        source_dir = root / folder
        images = sorted(source_dir.glob("*.jpg"))
        for image in images:
            label = image.with_suffix(".png")
            if not label.exists():
                raise FileNotFoundError(label)
            add("cmp_" + folder, image.stem, image, str(label.resolve()),
                cached_mask("cmp_" + folder, image.stem,
                            lambda label=label: remap(indexed_values(label), CMP_MAP, "CMP")))

    etrims = root / "etrims-db_v1"
    image_dir = etrims / "images" / "08_etrims-ds"
    label_dir = etrims / "annotations" / "08_etrims-ds"
    images = sorted(image_dir.glob("*.jpg"))
    if {path.stem for path in images} != {path.stem for path in label_dir.glob("*.png")}:
        raise ValueError("eTRIMS-8 image/label names differ")
    for image in images:
        label = label_dir / f"{image.stem}.png"
        add("etrims8", image.stem, image, str(label.resolve()),
            cached_mask("etrims8", image.stem,
                        lambda label=label: remap(indexed_values(label), ETRIMS_MAP, "eTRIMS-8")))

    hznu_root = root / "Hznu Facade Dataset" / "original"
    archive = hznu_root / "hznu_facade_dataset_original.zip"
    extracted = hznu_root / "all_adjust_zhedang"

    def process_hznu(names, read_json, copy_image, original_label) -> None:
        for name in names:
            stem = Path(name).stem
            json_name = str(Path(name).with_suffix(".json"))
            image = output / "images" / "hznu" / f"{stem}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            if not image.exists():
                copy_image(name, image)
            annotation = json.loads(read_json(json_name))
            with Image.open(image) as picture:
                orientation = picture.getexif().get(274, 1)
                corrected = ImageOps.exif_transpose(picture).convert("RGB")
                size = corrected.size
                declared = (annotation.get("imageWidth"), annotation.get("imageHeight"))
                if all(value is not None for value in declared) and declared != size:
                    raise ValueError(
                        f"Hznu {stem}: EXIF-corrected JPEG {size}, JSON {declared}"
                    )
                if orientation != 1:
                    corrected.save(image, quality=95)
            counts = Counter(shape.get("label") for shape in annotation.get("shapes", []))
            anomalies.update({label: count for label, count in counts.items() if label not in HZNU_MAP})
            mask = cached_mask("hznu", stem, lambda: hznu_mask(annotation, size)[0])
            add("hznu", stem, image, original_label(json_name), mask)

    if extracted.is_dir():
        names = sorted(path.name for path in extracted.glob("*.jpg"))
        process_hznu(names,
                     lambda name: (extracted / name).read_bytes(),
                     lambda name, target: shutil.copyfile(extracted / name, target),
                     lambda name: str((extracted / name).resolve()))
    elif archive.is_file():
        with zipfile.ZipFile(archive) as package:
            members = {item.filename for item in package.infolist()}
            names = sorted(name for name in members if name.lower().endswith(".jpg"))

            def copy_zip_image(name: str, target: Path) -> None:
                with package.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

            process_hznu(names, package.read, copy_zip_image,
                         lambda name: f"{archive.resolve()}::{name}")
    else:
        raise FileNotFoundError(f"Neither extracted Hznu directory nor ZIP found under {hznu_root}")

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "name", "image", "source_label", "mask",
                                                  *NAMES.values()))
        writer.writeheader()
        writer.writerows(rows)
    report = {"classes": NAMES, "counts": dict(by_source), "anomalous_hznu_labels": dict(anomalies),
              "mapping": {"3dom": DOM_MAP, "cmp": CMP_MAP, "etrims8": ETRIMS_MAP, "hznu": HZNU_MAP},
              "unmapped_cmp_ids": [0, 1, 7, 12],
              "note": "No training started. CMP unlabeled/background/balcony/shop are ignored; no BFDD component truth is assumed."}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../collected_datasets"))
    parser.add_argument("--output", type=Path, default=Path("data/components_3class"))
    parser.add_argument("--previews-per-source", type=int, default=6)
    args = parser.parse_args()
    prepare(args.root, args.output, args.previews_per_source)


if __name__ == "__main__":
    main()
