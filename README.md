# InfraDetect

Reproducible research code for UAV-based building-facade inspection using RGB
segmentation, thermal-IR evidence, RGB-IR registration, and 2D-to-3D semantic
fusion.

## Repository layout

- `rgb_baseline/`: three-class RGB DeepLabV3+ baseline.
- `rgb_6ml/`: shared six-label DeepLabV3+ and SegFormer pipeline.
- `ir/`: component segmentation, IR segmentation, fusion, and evaluation code.
- `mfdn/`: RGB-IR feature extraction and registration code derived from MFDN.
- `colmap_integration/`: RGB model comparison and dense 2D-to-3D fusion tools.
- `fast_livo2_integration/`: native environment, data-flow notes, and helper tools.
- `docs/`: model, dataset, and checkpoint handoff information.

## Start here

Team members who only need inference should read
[`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md). Training data locations, label
semantics, and redistribution constraints are documented in
[`docs/DATASETS.md`](docs/DATASETS.md).

Large datasets, checkpoints, virtual environments, generated predictions,
COLMAP workspaces, point clouds, and ROS bags are intentionally not stored in
Git. Approved checkpoints will be linked from the model zoo.

## Current status

- RGB three-class and six-label models: trained and evaluated.
- Building component models: trained and evaluated.
- Legacy six-class IR baseline: trained and retained for comparison.
- Specialized IR `Background / Hollow Areas / Stains` model: training config,
  leakage-resistant split, and code are ready; checkpoint and final metrics are
  pending GPU training.
- RGB-IR registration and 2D-to-3D fusion: integration code and reproducibility
  notes are included.

Predictions are inspection candidates, not confirmed engineering defects.

