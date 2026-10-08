# Model zoo and team handoff

Use the approved `best.pt` checkpoint with its matching YAML configuration.
Checkpoint downloads will be added separately; model binaries are not committed
to normal Git history.

| Module | Output | Model | Training data | Recommended checkpoint | Result / status |
|---|---|---|---|---|---|
| RGB baseline | background, crack, spalling | DeepLabV3+ / ResNet50 | Current-building tiles, 4286/840/840 | `rgb_baseline/runs/rgb_baseline/best.pt` | mIoU 0.7227; crack IoU 0.5492; spalling IoU 0.6303 |
| RGB six-label general model | crack, spalling, exposed rebar, rust, efflorescence, wetspot | SegFormer-B0 | DACL10K, 6241/694/975 | `rgb_6ml/runs/segformer_b0_6ml/best.pt` | six-label mIoU 0.4135 |
| RGB six-label comparison | same six labels | DeepLabV3+ / ResNet50 | Same DACL10K split | `rgb_6ml/runs/deeplabv3plus_6ml/best.pt` | six-label mIoU 0.3994 |
| Current-building RGB | crack and spalling partial-label adaptation | SegFormer-B0 | DACL10K then current-building data | `rgb_6ml/runs/segformer_b0_6ml_building/best.pt` | crack IoU 0.4850; spalling IoU 0.6216 |
| Current-building RGB comparison | same | DeepLabV3+ / ResNet50 | Same adaptation data | `rgb_6ml/runs/deeplabv3plus_6ml_building/best.pt` | crack IoU 0.5077; spalling IoU 0.5982 |
| Five-class components | facade, window, obstacle, sky, door | SegFormer-B0 | 3DOM, 183/22/22 | `ir/runs/components_3dom/best.pt` | mIoU 0.5059; facade IoU 0.8946 |
| Three-class building regions | building surface, opening, other | SegFormer-B0 | 3DOM + CMP + eTRIMS + HZNU | `ir/runs/components_all_3class/best.pt` | mIoU 0.8253 |
| Legacy IR baseline | background + five defect labels | DeepLabV3+ / ResNet50 | BFDD / Dataset_1x | `ir/runs/ir_bfdd_weighted/best.pt` | mIoU 0.2550; comparison only |
| Specialized IR | background, hollow areas, stains | DeepLabV3+ / ResNet50 | BFDD, 570/44/224 date-disjoint split | **Pending** | **Checkpoint and metrics to be updated after training** |
| RGB-IR registration | keypoints, descriptors, homography/warp | MFDN | UST Campus pretrained model | `mfdn/Pretrained/UST_Campus.pth` | Upstream pretrained checkpoint; download separately |

## Key training settings

| Model | Resolution | Batch | Epochs | Optimizer / LR | Loss / notes |
|---|---:|---:|---:|---|---|
| RGB baseline | 1024 tiles | 4 | 50 | AdamW, 1e-4 | 0.5 CE + 0.5 Dice; ImageNet ResNet50 |
| RGB six-label models | 512x512 | 4, accumulation 2 | 50 | AdamW, 6e-5, cosine | masked BCE + Dice; independent sigmoid outputs |
| Current-building adaptation | 512x512 | inherited | 5 | AdamW, 1e-5 | frozen encoder; partial-label masked loss |
| Five-class components | 512x768 | 2 | 50 | AdamW, 6e-5 | ImageNet MiT-B0 |
| Three-class building regions | 512x768 | 2 | 20 | AdamW, 2e-5 | ImageNet MiT-B0; rotation augmentation |
| Legacy IR | 512x640 | 4 | 50 | AdamW, 1e-4 | CE + Dice; weights `[1,7,8,8,7,6]` |
| Specialized IR | 512x640 | 4 | 50 | AdamW, 1e-4 | CE + Dice; weights `[1,8,6]`; other defects ignored |
| MFDN retraining | paired RGB-IR | 2 | 50 | configured by `train.py` | Prefer pretrained inference unless adaptation is required |

The exact source of truth is always the matching YAML file. Do not tune on the
frozen test split. IR outputs should be described as candidates rather than
confirmed defects.

