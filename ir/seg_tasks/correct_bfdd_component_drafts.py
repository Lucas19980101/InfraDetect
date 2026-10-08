"""Apply reported BFDD component corrections and create before/after review panels.

These are coarse human-guided draft labels, not final pixel-accurate ground truth.
Class IDs: 0 building_surface, 1 opening, 2 other.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PALETTE_BGR = np.array(((255, 0, 0), (0, 255, 0), (0, 255, 255)), np.uint8)


YELLOW_TO_BLUE = {
    # 2025-06-24
    *(f"DJI_20250624{time}_{number:04d}" for time, number in (
        ("181912", 12), ("181915", 13), ("181917", 14),
        ("183102", 66), ("183104", 67), ("183107", 68), ("183110", 69),
        ("183112", 70), ("183114", 71), ("183124", 73))),
    # 2025-06-25
    "DJI_20250625184150_0015", "DJI_20250625185217_0001",
    "DJI_20250625173615_0046", "DJI_20250625173640_0047",
    "DJI_20250625175111_0126", "DJI_20250625175117_0128",
    "DJI_20250625175120_0129", "DJI_20250625175124_0130",
    "DJI_20250625175135_0133",
    # 2025-06-26
    "DJI_20250626182715_0036", "DJI_20250626182737_0037",
    "DJI_20250626182806_0038", "DJI_20250626182842_0043",
    "DJI_20250626182845_0044",
}

ALL_BLUE = {
    # 2025-06-25
    "DJI_20250625182155_0091",
    *(f"DJI_202506251852{time}_{number:04d}" for time, number in (
        ("22", 2), ("26", 3), ("28", 4), ("34", 5), ("41", 6),
        ("45", 7), ("48", 8), ("52", 9))),
    "DJI_20250625185307_0011",
    # 2025-06-26
    "DJI_20250626182648_0033",
    *(f"DJI_2025062618{time}_{number:04d}" for time, number in (
        ("3727", 87), ("3743", 88), ("3755", 89), ("3802", 90),
        ("3808", 91), ("3813", 92), ("3824", 93), ("3834", 94),
        ("3840", 95), ("3846", 96), ("3857", 97), ("3926", 98),
        ("3938", 99), ("3951", 100), ("3954", 101), ("3959", 102),
        ("4007", 103), ("4018", 105), ("4021", 106), ("4029", 107),
        ("4031", 108), ("4038", 109), ("4046", 110), ("4049", 111))),
}

GREEN_TO_BLUE = {
    "DJI_20250626183210_0060", "DJI_20250626183213_0061",
    "DJI_20250626183215_0062", "DJI_20250626183221_0063",
}


def blend(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = PALETTE_BGR[mask]
    return cv2.addWeighted(image, 0.6, color, 0.4, 0)


def apply(name: str, mask: np.ndarray) -> tuple[np.ndarray, str]:
    corrected = mask.copy()
    rules: list[str] = []
    if name in YELLOW_TO_BLUE:
        corrected[corrected == 2] = 0
        rules.append("other->building_surface")
    if name in GREEN_TO_BLUE:
        corrected[corrected == 1] = 0
        rules.append("opening->building_surface")
    if name in ALL_BLUE:
        corrected[:] = 0
        rules.append("entire image->building_surface")
    if name == "DJI_20250625173611_0045":
        cutoff = round(mask.shape[1] * 0.85)
        region = corrected[:, :cutoff]
        region[region == 2] = 0
        rules.append("left 85% other->building_surface")
    if name == "DJI_20250626182554_0026":
        corrected[:round(mask.shape[0] * 0.75), :] = 0
        rules.append("top 75%->building_surface")
    if name == "DJI_20250626182558_0027":
        corrected[corrected == 2] = 0
        corrected[:, :round(mask.shape[1] * 0.5)] = 0
        corrected[:, round(mask.shape[1] * 0.75):] = 0
        rules.append("other->building_surface; left 50% and right 25%->building_surface")
    return corrected, "; ".join(rules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=Path("runs/components_all_3class/bfdd_rgb_audit"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/components_all_3class/bfdd_component_drafts"))
    parser.add_argument("--per-sheet", type=int, default=5)
    args = parser.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text(encoding="utf-8"))
    rgb_paths = {row["name"]: Path(row["rgb"]) for row in summary["samples"]}
    requested = YELLOW_TO_BLUE | GREEN_TO_BLUE | ALL_BLUE | {
        "DJI_20250625173611_0045", "DJI_20250626182554_0026", "DJI_20250626182558_0027"}
    missing = sorted(requested - set(rgb_paths))
    if missing:
        raise ValueError(f"Corrections refer to images absent from audit summary: {missing}")
    (args.output / "masks").mkdir(parents=True, exist_ok=True)
    (args.output / "panels").mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    records = []
    for name in sorted(requested):
        image = cv2.imread(str(rgb_paths[name]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(args.audit / f"{name}_component_ids.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape:
            raise ValueError(f"Missing or mismatched RGB/mask for {name}")
        corrected, rule = apply(name, mask)
        cv2.imwrite(str(args.output / "masks" / f"{name}.png"), corrected)
        views = [image, blend(image, mask), blend(image, corrected)]
        views = [cv2.resize(view, (320, 240), interpolation=cv2.INTER_AREA) for view in views]
        panel = np.hstack(views)
        cv2.putText(panel, f"{name} | RGB | before | corrected draft", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output / "panels" / f"{name}.jpg"), panel)
        panels.append(panel)
        records.append({"name": name, "rgb": str(rgb_paths[name]), "rule": rule,
                        "changed_pixels": int(np.count_nonzero(mask != corrected))})
    for page, start in enumerate(range(0, len(panels), args.per_sheet), 1):
        page_panels = panels[start:start + args.per_sheet]
        canvas = np.vstack(page_panels)
        cv2.imwrite(str(args.output / f"contact_sheet_{page:03d}.jpg"), canvas)
    (args.output / "corrections.json").write_text(json.dumps({
        "status": "coarse human-guided drafts; visually verify before training",
        "classes": {"0": "building_surface", "1": "opening", "2": "other"},
        "samples": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} corrected drafts and {len(records)} comparison panels to {args.output}")
    print(f"Contact sheets: {(len(panels) + args.per_sheet - 1) // args.per_sheet}")


if __name__ == "__main__":
    main()
