#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
road_result_dir="${1:-results/pure_rerun}"
/usr/bin/python3 tools/five_video_classical.py --backend evidence --output "$road_result_dir"
/usr/bin/python3 tools/evaluate_lane_review.py \
  --labels results/classical_five_videos/lane_review_labels_v4.json \
  --predictions "$road_result_dir/predictions.json" \
  --output "$road_result_dir/review_metrics.json" --require-all-pass
