#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p .native/src .native/ws/src
fetch_source() {
  local url="$1" dest="$2" revision="$3"
  if [ ! -d "$dest/.git" ]; then
    git clone "$url" "$dest"
    git -C "$dest" checkout "$revision"
  fi
  if [ "$(git -C "$dest" rev-parse HEAD)" != "$revision" ]; then
    echo "Unexpected revision in $dest; preserve local changes and inspect manually." >&2
    exit 1
  fi
}
fetch_source https://github.com/strasdat/Sophus.git .native/src/Sophus a621ff2e56c56c839a6c40418d42c3c254424b5c
fetch_source https://github.com/xuankuzcr/rpg_vikit.git .native/ws/src/rpg_vikit 6c886c8e5d83997806e00294826d528cea3581dd
fetch_source https://github.com/Livox-SDK/livox_ros_driver.git .native/ws/src/livox_ros_driver 3d240d5666129e1a3052e78ee8487a04b08fdda3
fetch_source https://github.com/Livox-SDK/Livox-SDK.git .native/src/Livox-SDK 9306596a2bf15c1343bc023b497465ed0a32909d
if [ ! -e .native/ws/src/FAST-LIVO2 ]; then
  ln -s ../../.. .native/ws/src/FAST-LIVO2
fi
