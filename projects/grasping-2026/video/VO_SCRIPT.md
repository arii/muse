# Demo video caption script (was TTS narration, now burned-in captions)

Source of the `<Captions>` cue text in `src/demo.tsx` (verbatim; long
lines are paged across cues in the composition, never rewritten).
Numbers spelled out as read ("seventy-seven percent", "forty of fifty").

## title (0:00–0:10)
Tell a three-jointed arm what to grasp, in words. It figures out the rest.

## pipeline (0:10–0:25)
Six stages turn language into motion. A vision model grounds the instruction.
SAM three point one segments the target. A sampler proposes grasps.
And MuJoCo executes.

## sam (0:25–1:10)
Here is the perception stage, live. The prompt is red box.
Watch SAM three point one find it among the clutter and trace its exact outline.
We checked the mask against simulator ground truth: seventy-seven percent overlap.
Ground truth never produced it. The model saw only the image.
This mask is what the grasp sampler reasons over.
Boxes touch and overlap in clutter. Bounding boxes go ambiguous. Pixel masks do not.
That is why segmentation is load-bearing for this task.

## lineup (1:10–1:25)
Seven planners attempt the same scenes. Scripts, reinforcement learning,
cloning, and vision-language planning, side by side.

## ppowin (1:25–1:37)
The PPO policy wins forty of fifty episodes, the best overall,
by learning restraint: approach gently, disturb nothing.

## matrix (1:37–1:52)
Across five scene types, learned policies match or beat every script.
Cloning at double data ties its own teacher, forty to forty.

## close (1:52–2:02)
A robot arm that grasps what you name.
Built with SAM three point one on the Meta Model API.
