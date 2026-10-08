"""Prepare selected BFDD RGB/IR pairs as same-name PNG inputs for MFDN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfdd-root", type=Path, default=Path("../collected_datasets/Dataset_1x"))
    parser.add_argument("--output", type=Path, default=Path("MFDN/BFDD_Pairs"))
    parser.add_argument("--names", nargs="+", required=True, help="Exact pair stems")
    args = parser.parse_args()
    vis_dir, ir_dir = args.output / "test" / "VIS", args.output / "test" / "IR"
    vis_dir.mkdir(parents=True, exist_ok=True); ir_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.names:
        rgb_matches = [path for path in (args.bfdd_root / "RGB").glob(f"{name}.*")
                       if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        ir_matches = [path for path in (args.bfdd_root / "IR").glob(f"{name}.*")
                      if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(rgb_matches) != 1 or len(ir_matches) != 1:
            raise ValueError(f"Expected one RGB and IR file for {name}: {rgb_matches}, {ir_matches}")
        rgb, ir = cv2.imread(str(rgb_matches[0])), cv2.imread(str(ir_matches[0]))
        if rgb is None or ir is None:
            raise OSError(f"Could not read pair {name}")
        if not cv2.imwrite(str(vis_dir / f"{name}.png"), rgb):
            raise OSError(f"Could not write VIS {name}")
        if not cv2.imwrite(str(ir_dir / f"{name}.png"), ir):
            raise OSError(f"Could not write IR {name}")
        records.append({"name": name, "rgb": str(rgb_matches[0].resolve()),
                        "ir": str(ir_matches[0].resolve()), "rgb_size": [rgb.shape[1], rgb.shape[0]],
                        "ir_size": [ir.shape[1], ir.shape[0]]})
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} BFDD pairs in {args.output}")


if __name__ == "__main__":
    main()
