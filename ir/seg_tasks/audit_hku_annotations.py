"""Render HKU LabelMe component or air-conditioner annotations for review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .audit_labelme_other import shape_points


COLORS = {
    "opening": (255, 80, 40),
    "other": (0, 220, 255),
    "air_conditioner": (210, 50, 230),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    records = []
    for name in ids:
        image_path = args.annotations / f"{name}.png"
        annotation = args.annotations / f"{name}.json"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        overlay = image.copy()
        counts: dict[str, int] = {}
        errors: list[str] = []
        if annotation.exists():
            data = json.loads(annotation.read_text(encoding="utf-8"))
            for index, shape in enumerate(data.get("shapes", [])):
                label = str(shape.get("label", "")).strip()
                if label not in COLORS:
                    errors.append(f"shape {index}: unexpected label {label!r}")
                    continue
                try:
                    points = shape_points(shape)
                except ValueError as exc:
                    errors.append(f"shape {index}: {exc}")
                    continue
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [points], 255)
                tint = np.full_like(image, COLORS[label])
                selected = mask > 0
                overlay[selected] = cv2.addWeighted(
                    overlay, 0.5, tint, 0.5, 0)[selected]
                cv2.polylines(overlay, [points], True, COLORS[label], 2)
                counts[label] = counts.get(label, 0) + 1
        status = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        if not annotation.exists():
            status = "no shapes"
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(overlay, f"{name}  {status}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"{name}_overlay.jpg"), overlay)
        records.append({"name": name, "annotation": annotation.exists(),
                        "shapes": counts, "errors": errors})
    summary = {"samples": len(records),
               "annotated": sum(row["annotation"] for row in records),
               "errors": sum(bool(row["errors"]) for row in records),
               "records": records}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("samples", "annotated", "errors")}, indent=2))


if __name__ == "__main__":
    main()
