"""List Hznu images whose EXIF rotation needs image/mask alignment review."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def run(source: Path, output: Path) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(source)
    rows = []
    status_counts: Counter = Counter()
    for image_path in sorted(source.glob("*.jpg")):
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            size = image.size
        if orientation in (1, None):
            continue
        annotation = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        declared = (annotation.get("imageWidth"), annotation.get("imageHeight"))
        if declared == (None, None):
            status = "missing_json_size"
        elif declared == size[::-1]:
            status = "swapped_json_size"
        elif declared == size:
            status = "raw_json_size"
        else:
            status = "other_json_size"
        rows.append({"source": "hznu", "name": image_path.stem,
                     "reason": f"EXIF orientation {orientation}; {status}; image/mask alignment not visually confirmed",
                     "orientation": orientation, "json_size_status": status})
        status_counts[status] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "name", "reason", "orientation", "json_size_status"))
        writer.writeheader(); writer.writerows(rows)
    return {"flagged": len(rows), "json_size_status": dict(status_counts), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("../collected_datasets/Hznu Facade Dataset/original/all_adjust_zhedang"))
    parser.add_argument("--output", type=Path, default=Path("data/components_3class/hznu_orientation_review.csv"))
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
