#!/usr/bin/env bash
set -euo pipefail

repo_owner="UnKaiFoo"
repo_name="HERO_Calib"
repo_full_name="${repo_owner}/${repo_name}"
repo_https_url="https://github.com/${repo_full_name}.git"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$script_dir"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is not installed; installing the Ubuntu package..."
  sudo apt update
  sudo apt install -y gh
fi

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  echo "Opening GitHub's web-login flow..."
  gh auth login --hostname github.com --web
fi

gh auth setup-git --hostname github.com

if ! gh repo view "$repo_full_name" >/dev/null 2>&1; then
  gh repo create "$repo_full_name" \
    --private \
    --description "CARLA 0.9.16 Toyota Prius wheel-based driver-eye calibration"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$repo_https_url"
else
  git remote add origin "$repo_https_url"
fi

git push -u origin main
gh repo view "$repo_full_name" --web

echo "Published privately: https://github.com/${repo_full_name}"
