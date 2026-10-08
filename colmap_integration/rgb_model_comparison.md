<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Shared DeepLabV3+ / SegFormer evaluation

Run from the COLMAP repository root. The comparison uses the **same image
split, resize (512×512), batch size, and metric definitions** for both models.
Each checkpoint contributes its frozen per-class validation threshold; test
data are never used to tune thresholds. The output is one JSON report with
per-class IoU, Dice, Precision, Recall, crack relaxed F1, measured inference
speed, parameter count, model state size, GPU peak allocated memory (when run
on CUDA), and optional point-level evaluation.

Install the local requirements, including DeepLab's optional backend:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python \
  -r requirements-rgb-integration.txt
```

## 2D comparison on the existing DACL10K test split

```bash
.venv/bin/python scripts/compare_rgb_models.py \
  --checkpoint segformer=../RGB/6ML/runs/segformer_b0_6ml/best.pt \
  --checkpoint deeplab=../RGB/6ML/runs/deeplabv3plus_6ml/best.pt \
  --device cuda --batch-size 4 \
  --output benchmark/reconstruction/runs/rgb-dacl-comparison.json
```

For a quick CPU check, add `--device cpu --max-samples 2 --batch-size 1
--warmup 0 --repeats 1`, and use a separate output filename. Such a partial
run is **not** an accuracy or speed comparison. The current-building split can
be selected with `--dataset building --data-root ../DeepLabs+/dataset` and the
two `*_building/best.pt` checkpoints. That split supervises crack/spalling
only; missing class truth is reported as `null`, not a zero score.

## 2D to 3D for both models

Export six probability maps per original image into separate folders. The
exporter uses the same inference adapter for both architectures and can resume
when run again with the same checkpoint and settings:

```bash
.venv/bin/python scripts/export_rgb_probabilities.py \
  --checkpoint ../RGB/6ML/runs/segformer_b0_6ml/best.pt \
  --output benchmark/reconstruction/runs/south-rgb-segformer --device cuda

.venv/bin/python scripts/export_rgb_probabilities.py \
  --checkpoint ../RGB/6ML/runs/deeplabv3plus_6ml/best.pt \
  --output benchmark/reconstruction/runs/south-rgb-deeplab --device cuda
```

Fuse each model's probabilities against the **same** COLMAP reconstruction:

```bash
.venv/bin/python scripts/dense_semantic_fusion.py \
  --classes crack spalling exposed_rebar rust efflorescence wetspot \
  --probabilities benchmark/reconstruction/runs/south-rgb-segformer/probabilities_512 \
  --output benchmark/reconstruction/runs/south-building-dense/segformer_labeled.ply

.venv/bin/python scripts/dense_semantic_fusion.py \
  --classes crack spalling exposed_rebar rust efflorescence wetspot \
  --probabilities benchmark/reconstruction/runs/south-rgb-deeplab/probabilities_512 \
  --output benchmark/reconstruction/runs/south-building-dense/deeplab_labeled.ply
```

Pass those point clouds to the shared evaluator with the same two checkpoints
and DACL test data as the first command, adding:

```bash
  --point-cloud segformer=benchmark/reconstruction/runs/south-building-dense/segformer_labeled.ply \
  --point-cloud deeplab=benchmark/reconstruction/runs/south-building-dense/deeplab_labeled.ply
```

The script verifies identical source geometry and fusion settings. Without
point-level ground truth, the 3D section reports **consistency only**: the
fraction of point-view votes agreeing with the final point label, plus the
same agreement restricted to predicted-positive points. The all-point number
can be inflated by background and must not be called accuracy. For point-level
IoU/Dice/Precision/Recall, also pass `--point-ground-truth PATH`; this must be
an index-aligned binary PLY with `x`, `y`, `z`, `label_bits` and optionally
`valid_bits` (which classes have been checked for each point). No ground truth
means the point-accuracy field remains `null`.

## External datasets, including DamCrack

Prepare the locally extracted DamCrack refined masks with:

```bash
.venv/bin/python scripts/prepare_damcrack.py \
  --data-root "../collected_datasets/DamCrack/Damage Segmentaion" \
  --output-root benchmark/datasets/damcrack
```

This converts red refined-mask pixels to `crack`, blue pixels to `spalling`,
and preserves images containing both as multilabel samples. The remaining four
classes have blank manifest fields and are therefore unsupervised rather than
incorrectly treated as background. `summary.json` records counts and the label
mapping. DamCrack does not provide patch-to-source-image provenance, so the
default deterministic 80/10/10 split is a patch-level development split and
may contain correlated crops from a common source photograph. Report that
limitation, and prefer an author-provided source mapping if one becomes
available.

After downloading a new dataset, create a CSV manifest. Paths are relative to
the CSV file (absolute paths also work). Example:

```csv
split,image,crack,spalling,exposed_rebar,rust,efflorescence,wetspot
test,images/view001.jpg,masks/view001_crack.png,masks/view001_spall.png,,,,
test,images/view002.jpg,masks/view002_crack.png,masks/view002_spall.png,,,,
```

Each class mask is grayscale: zero = negative, nonzero = positive. A blank
class path means **unlabeled**, not negative. An optional `valid_<class>` mask
column can restrict which pixels have reliable truth. Masks must match their
original image's dimensions. Then run the same comparison with
`--dataset manifest --manifest path/to/manifest.csv`. If the dataset also has
overlapping original photographs, generate its COLMAP dense workspace and pass
its image folder/workspace/probabilities/PLY paths to the export and fusion
commands. The dataset-specific conversion into the CSV manifest and, for 3D
accuracy, an aligned point-ground-truth PLY still needs to be done after the
actual files are available.

For DeepLabV3+, the local evaluator calls the underlying
`segmentation_models_pytorch` model's top-level forward method; this bridges
the current backend API difference without editing RGB/6ML checkpoints.
