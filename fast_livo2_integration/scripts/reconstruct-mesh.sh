#!/usr/bin/env bash
set -eo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$repo_dir/Log/pcd/all_downsampled_points.pcd}"
name="${2:-reconstruction}"
method="${3:-gp3}"
neighbors="${NORMAL_K:-20}"
depth="${POISSON_DEPTH:-8}"
radius="${GP3_RADIUS:-0.35}"
mu="${GP3_MU:-2.5}"
output_dir="$repo_dir/Log/mesh"

if [ ! -f "$input" ]; then
  echo "Point cloud not found: $input" >&2
  exit 1
fi

source "$repo_dir/scripts/native-env.sh"
mkdir -p "$output_dir"

normals="$output_dir/${name}_normals.pcd"
echo "Estimating normals (k=$neighbors)..."
pcl_normal_estimation "$input" "$normals" -k "$neighbors"

case "$method" in
  gp3)
    vtk="$output_dir/${name}_gp3.vtk"
    ply="$output_dir/${name}_gp3.ply"
    echo "Running local GP3 triangulation (radius=$radius, mu=$mu)..."
    pcl_gp3_surface "$normals" "$vtk" -radius "$radius" -mu "$mu"
    ;;
  poisson)
    vtk="$output_dir/${name}_depth${depth}.vtk"
    ply="$output_dir/${name}_depth${depth}.ply"
    echo "Running closed Poisson reconstruction (depth=$depth)..."
    pcl_poisson_reconstruction "$normals" "$vtk" -depth "$depth"
    ;;
  *)
    echo "Unknown method '$method'; use gp3 or poisson." >&2
    exit 2
    ;;
esac

echo "Writing PLY..."
pcl_vtk2ply "$vtk" "$ply"

echo "Mesh for pcl_viewer: $vtk"
echo "Mesh for Blender/MeshLab/CloudCompare: $ply"
