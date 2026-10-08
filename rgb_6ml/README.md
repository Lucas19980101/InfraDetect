# InfraDetect RGB 6ML

This project provides one shared data, loss, threshold-selection, evaluation,
and visualization pipeline for DeepLabV3+-6ML and SegFormer-B0-6ML.

The accepted three-class RGB baseline in `../../DeepLabs+` is frozen and is not
modified by this project. The legacy SegFormer/MMCV checkout in `../SegFormer`
is reference material only.

## Output channels

1. `crack` (`Crack` union `ACrack`)
2. `spalling` (`Spalling`)
3. `exposed_rebar` (`ExposedRebars`)
4. `rust` (`Rust`)
5. `efflorescence` (`Efflorescence`)
6. `wetspot` (`Wetspot`)

There is no background output channel. All six logits are independent and are
trained with a masked multi-label loss. Pixels positive for a selected class
remain supervised; pixels labeled only as excluded DACL10K classes are ignored;
unannotated pixels are valid negative background.

## Data audit

From this directory, using the accepted baseline environment:

```bash
PYTHONPATH=src ../../DeepLabs+/.venv/bin/python scripts/audit_dacl10k.py \
  --data-root ../datasets/dacl10k/raw/dacl10k_v2_devphase \
  --output-dir outputs/audit
```

The audit writes `dacl10k_audit.json` and `dacl10k_class_statistics.csv`.

## Frozen experiment split

The official training set is deterministically divided into experiment train
and threshold-tuning validation subsets. The official validation set remains a
frozen local test set:

```bash
PYTHONPATH=src:. ../../DeepLabs+/.venv/bin/python scripts/create_split_manifest.py \
  --data-root ../datasets/dacl10k/raw/dacl10k_v2_devphase \
  --output data/split_manifest.csv \
  --seed 42
```

## Mask cache

Convert the JSON polygons once into lossless one-byte PNG masks. Bits 0-5 hold
the six targets and bit 6 holds the valid-loss mask:

```bash
PYTHONPATH=src ../../DeepLabs+/.venv/bin/python scripts/prepare_dacl10k_masks.py \
  --data-root ../datasets/dacl10k/raw/dacl10k_v2_devphase \
  --cache-root ../datasets/dacl10k/cache/labels_6ml \
  --workers 4
```

## Training

Both models inherit the same data and training configuration. Set `HF_HOME` so
MiT-B0 weights remain inside this project:

```bash
HF_HOME="$PWD/.cache/huggingface" PYTHONPATH=src \
  ../../DeepLabs+/.venv/bin/python -m rgb_multilabel.train \
  --config configs/deeplabv3plus_6ml.yaml

HF_HOME="$PWD/.cache/huggingface" PYTHONPATH=src \
  ../../DeepLabs+/.venv/bin/python -m rgb_multilabel.train \
  --config configs/segformer_b0_6ml.yaml
```
