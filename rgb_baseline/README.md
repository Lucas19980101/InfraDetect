# InfraDetect RGB baseline

Minimal multiclass semantic-segmentation baseline using SMP DeepLabV3+ with a ResNet50 ImageNet encoder.

## Data

Place matching images and single-channel PNG masks under:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

Mask values are `0=background`, `1=crack`, `2=spalling`, and `255=ignore`. Split original UAV images or flight sequences **before** producing tiles, so related tiles cannot leak across splits.

Image and mask stems must match. For example:

```text
dataset/images/train/bridge_001.jpg
dataset/masks/train/bridge_001.png
```

The folders can be initialized with:

```bash
mkdir -p dataset/{images,masks}/{train,val,test}
```

When using the provided `crack_org` and `spalling_org` archives under `dataset/raw/`, generate leakage-safe 1024 px tiles with:

```bash
rgb-prepare --raw-root dataset/raw --output dataset --tile-size 1024 --stride 768
```

This splits original images before tiling, keeps every positive training tile, samples 25% of purely-background training tiles, and writes `dataset/manifest.csv`. Validation and test retain their full tile grids.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

First, populate the dataset described above and train the model:

```bash
rgb-train --config configs/baseline.yaml
```

Training displays per-batch progress, loss, running-average loss, learning rate, and ETA. It saves `last.pt` after every epoch, `best.pt` on improved validation mIoU, and an `epoch_XXX.pt` snapshot every five epochs (keeping the latest three). Resume an interrupted run with:

```bash
rgb-train --config configs/baseline.yaml --resume runs/rgb_baseline/last.pt
```

Only after training completes and creates `runs/rgb_baseline/best.pt`, run evaluation:

```bash
rgb-evaluate --config configs/baseline.yaml --checkpoint runs/rgb_baseline/best.pt
```

Prediction also requires that checkpoint and a real input path:

```bash
rgb-predict --checkpoint runs/rgb_baseline/best.pt --input full_resolution.jpg --save-logits
rgb-predict --checkpoint runs/rgb_baseline/best.pt --input flight.mp4 --video
```

The virtual environment must be activated again after opening a new terminal:

```bash
source .venv/bin/activate
```

If `python3 -m venv .venv` reports that `ensurepip` is unavailable, install the Ubuntu venv package first:

```bash
sudo apt update
sudo apt install python3-venv
```

The commands can also be invoked as Python modules, which avoids shell `PATH` issues:

```bash
python3 -m rgb_baseline.train --config configs/baseline.yaml
python3 -m rgb_baseline.evaluate --config configs/baseline.yaml --checkpoint runs/rgb_baseline/best.pt
python3 -m rgb_baseline.predict --checkpoint runs/rgb_baseline/best.pt --input full_resolution.jpg --save-logits
python3 -m rgb_baseline.predict --checkpoint runs/rgb_baseline/best.pt --input flight.mp4 --video
```

Inference blends overlapping **logits** with a Hann window before `argmax`. Add `--save-features` to save each tile's decoder feature tensor and spatial origin for later XYZ projection/fusion.

Video inference writes lossless class-index PNG masks to `mask_frames/` and a color preview MP4; the MP4 is visualization only.

The training dataset accepts already-created tiles and lightly augments them. Tile extraction/splitting is intentionally kept outside training because dataset-specific flight/original-image grouping is required to prevent leakage.
