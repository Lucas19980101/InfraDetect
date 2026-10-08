#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_dir/../DeepLabs+/.venv/bin/python"
dataset_dir="$repo_dir/../collected_datasets/Dataset_1x"
homography_dir="$repo_dir/MFDN/results/BFDD_All/homo_ir2vis/reproj/4096"
output_dir="$repo_dir/runs/fusion_status_dataset1x_hku_border_safe"
failure_log="$output_dir/failures.txt"

mkdir -p "$output_dir"
touch "$failure_log"
total=$(find "$dataset_dir/RGB" -maxdepth 1 -type f -name '*.JPG' | wc -l)
index=0

for rgb in "$dataset_dir"/RGB/*.JPG; do
  index=$((index + 1))
  stem="$(basename "$rgb" .JPG)"
  ir="$dataset_dir/IR/$stem.png"
  label="$dataset_dir/Label/$stem.png"
  homography="$homography_dir/$stem.mat"
  sample_output="$output_dir/$stem"
  if [[ -f "$sample_output/summary.json" ]]; then
    printf '[%d/%d] skip %s\n' "$index" "$total" "$stem"
    continue
  fi
  if [[ ! -f "$ir" || ! -f "$label" || ! -f "$homography" ]]; then
    printf '%s missing input\n' "$stem" | tee -a "$failure_log"
    continue
  fi
  printf '[%d/%d] %s\n' "$index" "$total" "$stem"
  if ! "$python_bin" -m seg_tasks.inspect_pair \
    --rgb "$rgb" \
    --ir "$ir" \
    --ir-label "$label" \
    --rgb-to-ir "$homography" \
    --component-config "$repo_dir/configs/components_hku_adapted.yaml" \
    --component-checkpoint "$repo_dir/runs/components_hku_adapted/best.pt" \
    --ir-config "$repo_dir/configs/ir_bfdd_weighted.yaml" \
    --ir-checkpoint "$repo_dir/runs/ir_bfdd_weighted/best.pt" \
    --ac-config "$repo_dir/configs/bfdd_ac_hku_adapted.yaml" \
    --ac-checkpoint "$repo_dir/runs/bfdd_ac_hku_adapted/best.pt" \
    --ac-threshold 0.78 \
    --ac-min-component-area 200 \
    --ac-local-low-threshold 0.40 \
    --ac-local-radius 24 \
    --review-linear-artifacts \
    --line-ac-threshold 0.45 \
    --line-proximity-radius 16 \
    --line-min-length 20 \
    --line-max-gap 6 \
    --line-thickness 5 \
    --edge-radius 8 \
    --ac-influence-radius 24 \
    --output "$sample_output"; then
    printf '%s failed\n' "$stem" | tee -a "$failure_log"
  fi
done

completed=$(find "$output_dir" -mindepth 2 -maxdepth 2 -type f -name summary.json | wc -l)
failures=$(wc -l < "$failure_log")
printf 'completed=%d total=%d failures=%d output=%s\n' \
  "$completed" "$total" "$failures" "$output_dir"
