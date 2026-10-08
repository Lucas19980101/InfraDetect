<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Dense 2D-to-3D semantic fusion

Run from the COLMAP repository root with the local `.venv`:

```bash
.venv/bin/python scripts/dense_semantic_fusion.py \
  --limit-points 100000 --limit-images 10 \
  --output benchmark/reconstruction/runs/south-building-dense/defects_dense_smoke.ply

.venv/bin/python scripts/dense_semantic_fusion.py
```

The second command uses all 5.5 million points and all 128 registered images.
It writes `defects_dense.ply` and `defects_dense.summary.json` beside the original
`fused.ply`; the original is never overwritten. The smoke-test output is partial
and should not be used to assess detection accuracy.

## Inputs

- `--dataset`: original COLMAP `sparse/` reconstruction with distorted cameras.
- `--workspace`: undistorted COLMAP `sparse/` reconstruction and geometric depth
  maps from `patch_match_stereo`.
- `--ply`: binary `fused.ply` produced by `stereo_fusion`.
- `--probabilities`: 16-bit grayscale PNGs named `<image-name>.<class>.png`.
  Each PNG is a probability map aligned to the *original* photograph, with
  0–65535 representing 0–1. Maps may be lower resolution than the photo.
  The existing RGB script supplies `crack` and `spalling` maps at 512×512.

To add the other classes, generate corresponding maps for every registered
image, then pass, for example:

```bash
.venv/bin/python scripts/dense_semantic_fusion.py \
  --classes crack spalling exposed_rebar rust efflorescence wetspot
```

Missing class maps cause an error; the script does not invent predictions.
The supported original camera models are `SIMPLE_PINHOLE`, `PINHOLE`,
`SIMPLE_RADIAL`, `RADIAL`, and `OPENCV`. Dense cameras must be `PINHOLE`.
Original and dense reconstructions must share the same coordinate frame.

## Method and outputs

For each image, the script projects points into the undistorted camera, checks
agreement with the geometric depth map (`--depth-tolerance`, default 2%), then
projects visible points into the original camera and bilinearly samples each
class probability map. A class requires at least `--min-views` positive views
(default 2) and `--min-fraction` of all visible views (default 0.3). Multiple
classes can pass independently. The class with the largest positive-view score
weighted by support fraction becomes the *primary* class for display only.

PLY `label` is the primary display label: `0` means no accepted defect, and
`1..N` follow the order of `--classes`. `label_bits` preserves *all* accepted
classes: bit 0 corresponds to the first class, bit 1 to the second, and so on.
For example, with `--classes crack spalling`, `label_bits=3` means both classes
are present (`1 | 2`). The explicit `is_<class>` fields contain the same
independent 0/1 decisions and are easier to inspect in a PLY viewer. Defect
points are colored by their primary class; color alone is not the complete
multi-label annotation. Other points keep their source RGB.

The PLY also stores `valid_views`, `prob_<class>` (mean over *all* visible
views), `support_<class>` (mean over positive views), and
`positive_views_<class>`. The JSON summary records independent per-class counts,
primary-label counts, multi-label point count, thresholds, and per-image
visible-point totals. If a prior RGB `summary.json` exists beside
the probability folder, its per-class thresholds are reused; otherwise 0.5 is
used. `--threshold` overrides all classes with one value.

This is a geometric/semantic integration framework, not a validation of the
detector. Thin cracks can be missed at 512×512 probability resolution; the
current South Building model has known domain mismatch and no ground-truth
defect labels. Inspect positive points against their source photos before
using them for measurement or reporting.
