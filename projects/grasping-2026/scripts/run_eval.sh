#!/usr/bin/env bash
# Convenience wrapper for running grasping evaluations inside Docker without
# repetitive docker-compose CLI boilerplate or accidental 10-seed VLM API bills.
#
# Usage:
#   ./scripts/run_eval.sh                      # default: canary seeds 2,6 (all policies, HTML + GIFs)
#   ./scripts/run_eval.sh --seeds 2            # single seed
#   ./scripts/run_eval.sh --no-vlm             # offline only (heuristic,ppo,diffusion) — zero VLM API risk
#   ./scripts/run_eval.sh --retrain-diffusion  # collect planar cage demos & retrain diffusion_v2.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

SEEDS="2,6"
PLANNERS="vlm,heuristic,ppo,diffusion"
RETRAIN_DIFF=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --planners)
      PLANNERS="$2"
      shift 2
      ;;
    --no-vlm|--offline)
      PLANNERS="heuristic,ppo,diffusion"
      shift
      ;;
    --retrain-diffusion)
      RETRAIN_DIFF=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${RETRAIN_DIFF}" -eq 1 ]]; then
  echo "[run_eval] Collecting planar cage demos on seeds 0-99..."
  docker compose run --rm -v "${PROJECT_DIR}:/workspace" --entrypoint python demo \
    scripts/collect_demos.py --demonstrator cage --seeds 0-99 --n-objects 4 --keep fc \
    --out data/demos_cage_planar
  echo "[run_eval] Training models/diffusion_v2.pt..."
  docker compose run --rm -v "${PROJECT_DIR}:/workspace" --entrypoint python demo \
    scripts/train_diffusion.py --demos data/demos_cage_planar --epochs 120 \
    --min-steps-per-epoch 60 --out models/diffusion_v2
fi

echo "[run_eval] Comparing planners (${PLANNERS}) on seeds (${SEEDS})..."
docker compose run --rm -v "${PROJECT_DIR}:/workspace" --entrypoint python demo \
  scripts/compare_planners.py --seeds "${SEEDS}" --planners "${PLANNERS}" \
  --gifs --html "${EXTRA_ARGS[@]}"
