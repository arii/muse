"""Grasp synthesis: object mask (+ depth) -> ranked grasp candidates.

Approach (classical, no learning required for v1):
  1. Extract the object contour from the SAM mask.
  2. Sample antipodal point pairs on the contour (normals ~opposed).
  3. Score each pair by: gripper width fit, approach clearance, and
     distance from the mask centroid (prefer central grasps).
  4. Return grasps ranked by score.

Grasp representation: dict(x, y, angle, width, score) in image frame;
callers project to the robot frame using the camera model.

Status: STUB — interface and scoring sketch are fixed; the sampler is TODO.
"""
import numpy as np


def synthesize_grasps(mask, depth=None, n_candidates=50):
    """Rank grasp candidates on a boolean object mask.

    Returns a list of dicts sorted by descending score. Empty list if
    no valid candidate is found.
    """
    # TODO:
    #  - contour extraction (cv2.findContours on mask)
    #  - antipodal pair sampling with normal opposition check
    #  - width/approach scoring; optional depth-based refinement
    raise NotImplementedError("Grasp sampler not yet implemented.")


def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
