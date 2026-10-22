"""Segmentation stage: grounded detection -> pixel mask via SAM.

Interface:
    segmenter = SAMSegmenter(checkpoint="sam_vit_h_4b8939.pth")
    mask = segmenter.segment(rgb, prompt={"boxes": [[x1, y1, x2, y2]]})
    # mask: np.ndarray[bool], shape (H, W)

The `prompt` comes from the grounding stage (perception/grounding.py):
a bounding box or point derived from the VLM's answer.

Setup:
    pip install segment-anything
    # download a checkpoint from the SAM repo (ViT-H recommended)

Status: STUB — interface is fixed; SAM wiring is TODO.
"""
import numpy as np


class SAMSegmenter:
    def __init__(self, checkpoint=None, device="cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def _load(self):
        # TODO: lazy-load SamPredictor here once a checkpoint is available.
        # from segment_anything import SamPredictor, sam_model_registry
        # sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint)
        # sam.to(self.device); self._predictor = SamPredictor(sam)
        raise NotImplementedError(
            "SAM checkpoint not configured — set one up per the docstring.")

    def segment(self, rgb, prompt):
        """Return a boolean mask for the prompted object."""
        if self._predictor is None:
            self._load()
        raise NotImplementedError("SAM inference not yet wired.")
