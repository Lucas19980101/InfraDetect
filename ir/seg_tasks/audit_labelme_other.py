"""Render LabelMe ``other`` annotations over their source RGB images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def shape_points(shape: dict) -> np.ndarray:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    kind = shape.get("shape_type", "polygon")
    if kind == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if len(points) < 3:
        raise ValueError(f"{kind} needs at least 3 polygon points after conversion")
    return np.rint(points).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for annotation in sorted(args.annotations.glob("*.json")):
        data = json.loads(annotation.read_text(encoding="utf-8"))
        image_path = (annotation.parent / data["imagePath"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {image_path}")
        expected = (int(data["imageHeight"]), int(data["imageWidth"]))
        if image.shape[:2] != expected:
            raise ValueError(f"Size mismatch for {annotation.name}: {image.shape[:2]} != {expected}")

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        labels, errors = set(), []
        for index, shape in enumerate(data.get("shapes", [])):
            label = str(shape.get("label", "")).strip()
            labels.add(label)
            if label != "other":
                errors.append(f"shape {index}: unexpected label {label!r}")
                continue
            try:
                cv2.fillPoly(mask, [shape_points(shape)], 255)
            except ValueError as exc:
                errors.append(f"shape {index}: {exc}")

        overlay = image.copy()
        tint = np.zeros_like(image); tint[:, :] = (0, 255, 255)
        selected = mask > 0
        overlay[selected] = cv2.addWeighted(image, 0.45, tint, 0.55, 0)[selected]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        cv2.putText(overlay, f"{annotation.stem}: other", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"{annotation.stem}_overlay.png"), overlay)
        cv2.imwrite(str(args.output / f"{annotation.stem}_other.png"), mask)
        records.append({"name": annotation.stem, "image": str(image_path),
                        "shapes": len(data.get("shapes", [])), "labels": sorted(labels),
                        "other_pixels": int(selected.sum()), "errors": errors})

    (args.output / "summary.json").write_text(
        json.dumps({"samples": records}, indent=2), encoding="utf-8")
    print(json.dumps({"annotations": len(records),
                      "nonempty": sum(row["other_pixels"] > 0 for row in records),
                      "errors": sum(bool(row["errors"]) for row in records),
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
