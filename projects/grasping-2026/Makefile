.PHONY: help eval eval-isolated eval-offline eval-all retrain-diffusion collect-demos train-diffusion demo-vlm demo-heuristic

SEEDS ?= 2,4

help:
	@echo "Available targets:"
	@echo "  make eval               - Run 4-policy comparison on canary seeds (default SEEDS=2,4; 0 API calls via cache)"
	@echo "  make eval-isolated      - Run 4-policy comparison on isolated seeds (2,6)"
	@echo "  make eval-offline       - Run offline-only comparison (heuristic,ppo,diffusion) — zero VLM cost"
	@echo "  make retrain-diffusion  - Collect planar cage demos (seeds 2,6) & retrain models/diffusion_v2.pt"
	@echo "  make eval-all           - Run full 10-seed comparison (0-9)"

eval:
	./scripts/run_eval.sh --seeds $(SEEDS)

eval-isolated:
	./scripts/run_eval.sh --seeds 2,6

eval-offline:
	./scripts/run_eval.sh --no-vlm --seeds $(SEEDS)

eval-all:
	./scripts/run_eval.sh --seeds 0-9

collect-demos:
	docker compose run --rm -v $$(pwd):/workspace --entrypoint python demo \
		scripts/collect_demos.py --demonstrator cage --seeds 2,6 --n-objects 4 --keep fc \
		--out data/demos_cage_planar

train-diffusion:
	docker compose run --rm -v $$(pwd):/workspace --entrypoint python demo \
		scripts/train_diffusion.py --demos data/demos_cage_planar --epochs 180 \
		--min-steps-per-epoch 60 --lr 5e-4 --out models/diffusion_v2

retrain-diffusion: collect-demos train-diffusion
	./scripts/run_eval.sh --seeds $(SEEDS)
