# CV submission form — copy-paste draft

Video: upload `video/out/demo_video.mp4` as unlisted YouTube first,
paste that URL in the video field.

## Title

Whole-Arm Grasping, Revisited: 2014 Thesis Meets 2026 Frontier Models

## Track / problem statement

Indie · Push the Frontier

## Short description (1–2 sentences)

Language-conditioned whole-arm grasping in simulation: name a target
in words, and SAM 3.1 segments it from the robot's camera. The arm
grasps it — benchmarked across seven planner families on five clutter
scenes.

## What we built (new work, Oct 27 – Nov 4)

- SAM 3.1 perception stage via Meta Model API: text prompt → pixel
  mask, verified live at IoU 0.77 vs sim ground truth
- End-to-end pipeline: grounding → segmentation → antipodal grasp
  synthesis → motion → MuJoCo execution
- Full benchmark: PPO from scratch (40/50), BC/diffusion cloning with
  a fair val-split pipeline (BC ties its teacher 40–40 at 2× data),
  VLM planning with ladder/attribution studies, scripted baselines —
  all on identical seeds × five scenes.
- Artifacts: results site, HuggingFace demo exporter, captioned demo
  video. 115+ commits in-window; only the repo scaffold, base sim,
  and stub files pre-date Oct 27.

## Meta models used

SAM 3.1 through the Meta Model API (perception stage; the load-bearing
component: boxes touch in clutter, bounding boxes go ambiguous, pixel
masks don't). Developed with Muse Code.

## Links

- Repo (public): https://github.com/arii/muse (project in
  `projects/grasping-2026/`, submission notes in `SUBMISSION.md`)
- Video: (paste unlisted YouTube URL)
