#!/usr/bin/env bash
set -euo pipefail

root="${1:-vendor}"
gvhmr_commit="${GVHMR_COMMIT:-6ec3ca39336c50492c0fae65fba2fb831fc7d866}"
gmr_commit="${GMR_COMMIT:-bb1bbe40774794fceb2a7c579a3464a28e68c844}"
beyondmimic_commit="${BEYONDMIMIC_COMMIT:-2184fa93e950960733d7024930a2c79616d295a0}"

mkdir -p "$root"
clone_pinned() {
  local url="$1" name="$2" commit="$3" dest="$root/$2"
  if [[ -e "$dest" ]]; then
    echo "Refusing to reuse existing checkout: $dest" >&2
    exit 2
  fi
  git clone "$url" "$dest"
  git -C "$dest" checkout --detach "$commit"
  git -C "$dest" status --short --branch
}

clone_pinned "https://github.com/zju3dv/GVHMR.git" GVHMR "$gvhmr_commit"
clone_pinned "https://github.com/YanjieZe/GMR.git" GMR "$gmr_commit"
clone_pinned "https://github.com/HAOTianGa03/GAOTIANHAO-G1-BeyondMimic.git" BeyondMimic "$beyondmimic_commit"
echo "Pinned source checkouts created under $root"
