# Dataset handoff

Full datasets are not redistributed in this Git repository. This document
records expected layouts, label semantics, and committed split manifests.

| Dataset | Purpose | Size / split | Labels | Repository artifact |
|---|---|---|---|---|
| Current-building RGB | RGB crack/spalling training and adaptation | 4286 train, 840 val, 840 test tiles | background, crack, spalling, ignore | `rgb_baseline/dataset/manifest.csv` when supplied locally |
| DACL10K | General RGB six-label segmentation | 6241 train, 694 experiment-val, 975 frozen local test | crack, spalling, exposed rebar, rust, efflorescence, wetspot | `rgb_6ml/data/split_manifest.csv` |
| BFDD / Dataset_1x | IR segmentation and RGB-IR pairs | 838 paired RGB, pseudo-colour IR, and masks | background, cracks, peeling, hollow areas, stains, erosion | `ir/data/bfdd_split.csv`, `ir/data/bfdd_hollow_stains_split.csv` |
| 3DOM | Building component segmentation | 227 images; 183/22/22 | facade, window, obstacle, sky, door | `ir/data/3dom_split.csv` |
| Combined component data | Robust building-region segmentation | 3DOM, CMP, eTRIMS, HZNU sources | building surface, opening, other | `ir/data/components_3class/manifest_grouped_splits.csv` |
| RGB-IR registration benchmarks | MFDN evaluation/adaptation | UST Campus, MSRS, VIS-IR as available | correspondences, landmarks or homographies | preparation scripts under `mfdn/` |

## Important rules

- Preserve committed manifests when comparing with reported results.
- Split source photographs or acquisition sessions before producing tiles.
- A blank partial-label field means **unlabelled**, not negative.
- In specialized IR training, original labels 1, 2, and 5 map to ignore value
  255; they are not healthy background.
- Check each third-party dataset license before downloading or redistributing
  its files.
- Review self-collected imagery for faces, plates, location information, and
  other sensitive content before sharing outside the group.

