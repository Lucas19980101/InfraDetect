#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$repo_dir/.native/env"
native_dir="$repo_dir/.native"
jobs="${BUILD_JOBS:-2}"
python "$repo_dir/scripts/native-patch-deps.py"
cmake -S "$native_dir/src/Sophus" -B "$native_dir/build/Sophus" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
cmake --build "$native_dir/build/Sophus" --target Sophus -j "$jobs"
cmake --install "$native_dir/build/Sophus"
cmake -S "$native_dir/src/Livox-SDK" -B "$native_dir/build/Livox-SDK" \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-Wno-error=c++20-compat
cmake --build "$native_dir/build/Livox-SDK" --target livox_sdk_static -j "$jobs"
cmake --install "$native_dir/build/Livox-SDK"
cd "$native_dir/ws"
catkin_make -j"$jobs" -l"$jobs" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSophus_DIR="$native_dir/build/Sophus" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
