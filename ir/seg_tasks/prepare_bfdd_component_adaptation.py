"""Append reviewed BFDD component drafts to the grouped component manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("data/components_3class/manifest_grouped_splits.csv"))
    parser.add_argument("--bfdd-manifest", type=Path, default=Path("data/bfdd_split.csv"))
    parser.add_argument("--drafts", type=Path,
                        default=Path("runs/components_all_3class/bfdd_component_drafts"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/components_3class/manifest_bfdd_adaptation.csv"))
    args = parser.parse_args()

    with args.base.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if fields is None:
        raise ValueError(f"Empty base manifest: {args.base}")
    corrections = json.loads((args.drafts / "corrections.json").read_text(encoding="utf-8"))
    corrected = {row["name"]: row for row in corrections["samples"]}
    with args.bfdd_manifest.open(newline="", encoding="utf-8") as handle:
        bfdd = {row["name"]: row for row in csv.DictReader(handle) if row["name"] in corrected}
    if set(bfdd) != set(corrected):
        raise ValueError(f"BFDD manifest mismatch: {sorted(set(corrected) - set(bfdd))}")

    # Entire capture groups stay together. These two held-out groups are close
    # to 15% each; every remaining reviewed group is training-only.
    split_for_group = {"32": "val", "38": "test"}
    for name in sorted(corrected):
        group = bfdd[name]["group"]
        split = split_for_group.get(group, "train")
        image = Path(corrected[name]["rgb"])
        mask = (args.drafts / "masks" / f"{name}.png").resolve()
        values = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if values is None or values.ndim != 2 or set(map(int, np.unique(values))) - {0, 1, 2, 255}:
            raise ValueError(f"Invalid BFDD component draft: {mask}")
        counts = {label: int((values == value).sum()) for value, label in
                  ((0, "building_surface"), (1, "opening"), (2, "other"), (255, "ignore"))}
        rows.append({"source": "bfdd_rgb", "name": name, "split": split,
                     "group": f"bfdd/{group}", "image": str(image.resolve()),
                     "source_label": str((args.drafts / "corrections.json").resolve()),
                     "mask": str(mask), **counts})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    counts = Counter((row["source"], row["split"]) for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    for (source, split), count in sorted(counts.items()):
        if source == "bfdd_rgb":
            print(f"{source}/{split}: {count}")


if __name__ == "__main__":
    main()
