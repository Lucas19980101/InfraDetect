from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


CLASS_NAMES = (
    "crack",
    "spalling",
    "exposed_rebar",
    "rust",
    "efflorescence",
    "wetspot",
)
COLORS_BGR = np.asarray(
    [
        (40, 40, 255),
        (0, 165, 255),
        (255, 80, 40),
        (30, 90, 180),
        (220, 255, 80),
        (255, 180, 30),
    ],
    dtype=np.uint8,
)
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen RGB 6ML checkpoint on arbitrary images."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument(
        "--sixml-src",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "RGB" / "6ML" / "src",
    )
    return parser.parse_args()


def find_images(root: Path, limit: int) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if root.is_file():
        return [root]
    images = sorted(path for path in root.iterdir() if path.suffix.lower() in suffixes)
    return images[:limit] if limit > 0 else images


def make_overlay(image: np.ndarray, masks: np.ndarray, alpha: float) -> np.ndarray:
    overlay = image.copy()
    for index, color in enumerate(COLORS_BGR):
        selected = masks[index]
        if selected.any():
            blended = (
                overlay[selected].astype(np.float32) * (1.0 - alpha)
                + color.astype(np.float32) * alpha
            )
            overlay[selected] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay


def make_contact_sheet(panels: list[np.ndarray], labels: list[str]) -> np.ndarray:
    width, height = 480, 384
    rendered = []
    for panel, label in zip(panels, labels):
        tile = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (width, 32), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(tile)
    rows = []
    for start in range(0, len(rendered), 4):
        row = rendered[start:start + 4]
        row.extend([np.zeros_like(rendered[0])] * (4 - len(row)))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.sixml_src.resolve()))
    from rgb_multilabel.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # RGB/6ML was trained with SMP 0.5+, while the reusable DeepLabs+ venv may
    # contain an older decoder whose forward signature expects unpacked
    # encoder features. Keep inference compatible without changing either env.
    if checkpoint["config"]["model"]["architecture"].lower() == "deeplabv3plus":
        original_forward = model.forward

        def compatible_forward(images: torch.Tensor):
            try:
                return original_forward(images)
            except TypeError as error:
                if "conv2d() received an invalid combination" not in str(error):
                    raise
                features = model.net.encoder(images)
                decoder = model.net.decoder(*features)
                return model.net.segmentation_head(decoder)

        model.forward = compatible_forward
    size = int(checkpoint["config"]["data"]["crop_size"])
    thresholds = np.asarray(
        [checkpoint["thresholds"][name] for name in CLASS_NAMES], dtype=np.float32
    )

    images = find_images(args.input, args.max_images)
    if not images:
        raise FileNotFoundError(f"No supported images found under {args.input}")
    masks_dir = args.output / "masks"
    overlays_dir = args.output / "overlays"
    probabilities_dir = args.output / "probabilities"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probabilities:
        probabilities_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    panels: list[np.ndarray] = []
    labels: list[str] = []
    with torch.inference_mode():
        for path in images:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read {path}")
            height, width = bgr.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
            normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
            probabilities_small = model(tensor).sigmoid()[0].cpu().numpy()
            probabilities = np.stack(
                [cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                 for channel in probabilities_small]
            )
            masks = probabilities >= thresholds[:, None, None]
            packed = np.zeros((height, width), dtype=np.uint8)
            for index in range(len(CLASS_NAMES)):
                packed |= masks[index].astype(np.uint8) << index
            cv2.imwrite(str(masks_dir / f"{path.stem}_mask_bits.png"), packed)
            overlay = make_overlay(bgr, masks, args.alpha)
            cv2.imwrite(str(overlays_dir / f"{path.stem}_overlay.jpg"), overlay)
            if args.save_probabilities:
                np.savez_compressed(
                    probabilities_dir / f"{path.stem}_probabilities.npz",
                    probabilities=probabilities.astype(np.float16),
                    class_names=np.asarray(CLASS_NAMES),
                    thresholds=thresholds,
                )
            row: dict[str, object] = {"image": path.name}
            row.update({f"{name}_pixels": int(masks[index].sum())
                        for index, name in enumerate(CLASS_NAMES)})
            rows.append(row)
            panels.append(overlay)
            labels.append(path.stem)

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(args.output / "contact_sheet.jpg"), make_contact_sheet(panels, labels))
    print(f"Processed {len(images)} images on {device}; output: {args.output.resolve()}")
    print("Thresholds:", dict(zip(CLASS_NAMES, thresholds.tolist())))


if __name__ == "__main__":
    main()
