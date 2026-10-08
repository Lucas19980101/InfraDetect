from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .inference import sliding_window_predict
from .model import build_model


def load_runtime(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], pretrained=False).to(device); load_checkpoint(model, checkpoint_path, device)
    return cfg, device, model


def predict_image(model, image_bgr, cfg, device, features=False):
    inf, data = cfg["inference"], cfg["data"]
    return sliding_window_predict(model, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), data["tile_size"], inf["stride"],
                                  cfg["model"]["num_classes"], device, inf["batch_size"], features)


def run(args) -> None:
    cfg, device, model = load_runtime(args.config, args.checkpoint)
    source, output = Path(args.input), Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if args.video:
        capture = cv2.VideoCapture(str(source)); fps = capture.get(cv2.CAP_PROP_FPS) or 25
        if not capture.isOpened(): raise FileNotFoundError(source)
        frame_dir = output / "mask_frames"; frame_dir.mkdir(exist_ok=True)
        writer = None; index = 0
        palette = np.array([[0, 0, 0], [0, 0, 255], [0, 165, 255]], dtype=np.uint8)
        while True:
            ok, frame = capture.read()
            if not ok: break
            logits = predict_image(model, frame, cfg, device)
            mask = logits.argmax(0).byte().numpy()
            cv2.imwrite(str(frame_dir / f"{index:06d}.png"), mask)
            preview = palette[mask]
            if writer is None: writer = cv2.VideoWriter(str(output / "prediction.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (mask.shape[1], mask.shape[0]))
            writer.write(preview); index += 1
        capture.release()
        if writer: writer.release()
        return
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(source)
    result = predict_image(model, image, cfg, device, args.save_features)
    logits, features = result if args.save_features else (result, None)
    cv2.imwrite(str(output / f"{source.stem}_mask.png"), logits.argmax(0).byte().numpy())
    if args.save_logits: np.save(output / f"{source.stem}_logits.npy", logits.numpy().astype(np.float32))
    if features is not None: torch.save(features, output / f"{source.stem}_decoder_features.pt")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions"); parser.add_argument("--video", action="store_true")
    parser.add_argument("--save-logits", action="store_true"); parser.add_argument("--save-features", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
