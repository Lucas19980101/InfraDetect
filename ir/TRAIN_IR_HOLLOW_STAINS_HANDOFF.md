# IR Hollow Areas and Stains Training Handoff

This handoff is for Codex on the NVIDIA-equipped primary workstation. Continue the InfraDetect IR work from the existing repository. Do not overwrite or delete the earlier six-class IR experiment in `runs/ir_bfdd_weighted/`.

## Objective

Train a three-class IR semantic-segmentation model using the BFDD pseudo-color IR images:

| New ID | Class |
|---:|---|
| 0 | Background |
| 1 | Hollow Areas |
| 2 | Stains |
| 255 | Ignore |

The original BFDD labels are mapped as follows:

```text
0 Background    -> 0 Background
1 Cracks        -> 255 Ignore
2 Peeling       -> 255 Ignore
3 Hollow Areas  -> 1 Hollow Areas
4 Stains        -> 2 Stains
5 Erosion       -> 255 Ignore
```

Keep the name `Stains`. Do not rename it to moisture or water seepage.

## Existing files

- Configuration: `configs/ir_bfdd_hollow_stains.yaml`
- Leakage-resistant manifest: `data/bfdd_hollow_stains_split.csv`
- Split implementation: `seg_tasks/splits.py::bfdd_specialized_rows`
- Label-map implementation: `seg_tasks/data.py::load_ir_mask`
- New output directory: `runs/ir_bfdd_hollow_stains/`

The new manifest contains 570 train, 44 validation, and 224 test images. It uses eleven complete acquisition sessions separated by gaps longer than five minutes. No session or acquisition date crosses a split: June 24-25 are train, June 28 is validation, and June 26 is test.

## First read-only checks

From the repository root:

```bash
cd ~/projects/InfraDetect
nvidia-smi
git status --short 2>/dev/null || true
```

Confirm that the expected files exist:

```bash
test -f IR/configs/ir_bfdd_hollow_stains.yaml
test -f IR/data/bfdd_hollow_stains_split.csv
test -d collected_datasets/Dataset_1x/IR
test -d collected_datasets/Dataset_1x/Label
```

Do not regenerate or modify the manifest unless the local BFDD collection differs from the reviewed 838-image dataset.

## Environment

Prefer the environment previously used to train `ir_bfdd_weighted`. Verify its packages:

```bash
python - <<'PY'
import cv2
import numpy
import segmentation_models_pytorch
import torch
import yaml

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If `segmentation_models_pytorch` is missing, install the version expected by this project inside the selected environment:

```bash
python -m pip install "segmentation-models-pytorch>=0.3.3,<0.4"
```

Do not begin the full run unless `torch.cuda.is_available()` is `True`.

## Validate the split and label mapping

Run the focused tests if pytest is available:

```bash
cd ~/projects/InfraDetect/IR
python -m pytest tests/test_seg_tasks.py -q
```

Then perform a short real-data smoke test:

```bash
python -m seg_tasks.smoke \
  --config configs/ir_bfdd_hollow_stains.yaml \
  --steps 5 \
  --height 128 \
  --width 160 \
  --batch-size 2
```

Stop and diagnose before full training if:

- a mask contains a value other than `0`, `1`, `2`, or `255` after mapping;
- the split count differs from 570/44/224;
- CUDA is unavailable;
- the smoke-test loss is non-finite;
- the smoke test cannot reduce loss.

## Split-quality decision

The repository's existing Dataset_1x COLMAP reconstruction covers only June 24, so it cannot provide shared 3D-point statistics for sessions 7 and 8. Contact-sheet review showed that both June 26 sessions repeatedly contain the same gray facade, green roof tiles, photovoltaic panels, eaves, and damage patterns. They must remain in the same split.

The resulting date-disjoint split deliberately accepts a small 44-image validation set rather than leaking the same facade across validation and test. Validation still contains 15 Hollow Areas-positive and 39 Stains-positive images. Do not move individual frames merely to make the split sizes closer to 70/15/15.

## Start training

```bash
cd ~/projects/InfraDetect/IR
python -m seg_tasks.train \
  --config configs/ir_bfdd_hollow_stains.yaml \
  2>&1 | tee runs/ir_bfdd_hollow_stains_train.log
```

Expected configuration:

- DeepLabV3+ with ResNet50 encoder;
- ImageNet encoder initialization;
- input size 512 x 640;
- batch size 4;
- 50 epochs;
- AdamW, learning rate `1e-4`;
- CE plus Dice loss;
- class weights `[1.0, 8.0, 6.0]`;
- mixed precision on CUDA.

Monitor GPU memory during the first epoch:

```bash
watch -n 2 nvidia-smi
```

If CUDA runs out of memory, reduce only `batch_size` from 4 to 2 in a copied configuration. Do not reduce the input resolution for the primary result unless batch size 1 also fails.

## Resume an interrupted run

The trainer writes `last.pt` atomically after every completed epoch. Resume with:

```bash
cd ~/projects/InfraDetect/IR
python -m seg_tasks.train \
  --config configs/ir_bfdd_hollow_stains.yaml \
  --resume runs/ir_bfdd_hollow_stains/last.pt \
  2>&1 | tee -a runs/ir_bfdd_hollow_stains_train.log
```

Before resuming, confirm that the checkpoint configuration still uses the same task, class names, label map, class weights, and manifest.

## Final evaluation

Evaluate the fixed test split only after training and model selection are complete:

```bash
cd ~/projects/InfraDetect/IR
python -m seg_tasks.evaluate \
  --config configs/ir_bfdd_hollow_stains.yaml \
  --checkpoint runs/ir_bfdd_hollow_stains/best.pt \
  --split test \
  --output runs/ir_bfdd_hollow_stains/test \
  2>&1 | tee runs/ir_bfdd_hollow_stains_test.log
```

Report all of the following:

- Background, Hollow Areas, and Stains IoU and Dice;
- mean IoU both including and excluding Background;
- per-class precision and recall derived from the confusion matrix;
- binary anomaly precision, recall, IoU, and Dice when Hollow Areas and Stains are merged;
- the number of test images and positive images per class;
- representative successes and failures;
- confirmation that the old six-class results were not overwritten.

Do not describe a wall-supported prediction as a confirmed engineering defect. Use `IR candidate`, `Hollow Areas candidate`, or `Stains candidate` until multi-view or field validation is available.

## Connect the model to fusion

The two-dimensional fusion path now reads IR class names, colors, and ground-truth mapping from the selected config. Pass the specialized config and checkpoint explicitly:

```bash
python -m seg_tasks.inspect_pair \
  --rgb path/to/RGB.JPG \
  --ir path/to/IR.png \
  --ir-label path/to/Label.png \
  --ir-config configs/ir_bfdd_hollow_stains.yaml \
  --ir-checkpoint runs/ir_bfdd_hollow_stains/best.pt \
  --output runs/hollow_stains_pair_check
```

When projecting a specialized fusion run into COLMAP, also pass:

```text
--ir-config configs/ir_bfdd_hollow_stains.yaml
```

If `--ir-config` is omitted from `project_colmap_fragment`, it intentionally uses the legacy five-defect metadata. Numeric status ID 1 remains backward-compatible, but its preferred external meaning is `wall_supported_candidate`. The legacy file `confirmed_defect_ids.png` is still written alongside `wall_supported_candidate_ids.png` for existing consumers.

## Current baseline for comparison

The earlier six-class model produced the following test IoU values on the old, partially leaky split:

| Class | IoU |
|---|---:|
| Background | 0.9254 |
| Cracks | 0.0384 |
| Peeling | 0.0033 |
| Hollow Areas | 0.2106 |
| Stains | 0.2703 |
| Erosion | 0.0816 |

These values are context only, not a strict head-to-head baseline, because the new model uses a different, session-disjoint split and a different label space.
