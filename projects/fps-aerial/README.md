# Relay Yard

A first-person collection game with a following aerial camera. Collect the eight
amber signals in a seeded, walled arena, then choose **Run again** to restart.

## Run

```sh
cd projects/fps-aerial
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. Any static host can serve this directory as-is.

## Controls

| Input | Action |
| --- | --- |
| Click arena, then move mouse | Capture pointer and look around in first person |
| Escape | Release pointer |
| WASD or arrow keys | Move relative to heading |
| Shift | Sprint |
| V or view button | Switch between first-person and aerial views |
| Left joystick on touch screens | Move |
| Drag right half on touch screens | Turn and look |

Movement continues in aerial view. The mint marker and arrow show position and
heading. Input hides the initial controls hint. Losing focus clears movement.

## What I learned

### Controls

Pointer lock supplies relative mouse deltas even at a screen edge. I apply yaw
to both movement and the camera, and clamp pitch short of a vertical view.
Keyboard and joystick input become a normalized direction before converting to
velocity, so diagonal movement stays at the selected speed. Axis-separated
circle/box collision lets a blocked player slide. The simulation clamps elapsed
time to 100 ms and substeps travel to prevent sprinting through thin obstacles
after a stalled frame; a long pause deliberately advances less game time.

### Cameras

First person places a perspective camera at eye height. Aerial view places that
camera directly above the player, looking down with north fixed at the top.
I interpolate position and quaternion rotation over 400 ms with smoothstep
easing. A single perspective camera keeps the projection continuous during the
switch. The marker appears during ascent and rotates with the player's heading.

### Rendering

I build the scene from a floor, grid, walls, boxes, lit amber pickups and a mint
player marker. A fixed seed shuffles obstacle cells and pickup locations while
leaving clear grid corridors. The frame loop integrates input, detects pickups,
updates the camera, and renders. Resize updates canvas dimensions and camera
aspect; the renderer caps device pixel ratio at 2 to limit mobile GPU work.

## Dependency

Three.js **0.160.1** is vendored at `vendor/three.module.min.js` and loaded through
the import map in `index.html`. Source:
<https://unpkg.com/three@0.160.1/build/three.module.min.js>.
The MIT license is in `vendor/LICENSE`.
I used a library instead of raw WebGL for this time-boxed skill build to focus
the learning on controls and cameras.

## Tests

With Node.js 22 or newer:

```sh
node --test
```

`logic.mjs` contains the seeded generator, movement integration, collision and
pickup detection, independently of the DOM and renderer. Tests exercise sliding,
corners, sprint substeps, time clamping, pickup radius and level reachability.
The path-scoped GitHub Actions workflow runs these tests with Node.js 22.
