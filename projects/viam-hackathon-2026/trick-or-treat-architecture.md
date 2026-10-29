# Trick or Treat — architecture notes (Viam hackathon project)

Decisions are final unless marked stretch. Product details are not finalized.
Event: Viam × MongoDB hackathon, Oct 29–30. Stack: Viam-native (D1).

## Idea

Halloween arm demo. A visitor at a tablet next to the arm requests TRICK (a
choreographed routine) or TREAT (the arm picks a candy and drops it in the
visitor's bucket). Every run is logged.

## Components

```
Input modalities ──► Request event ──► Orchestrator ──► Trick player
(button / voice /       {kind:            (main.py)          OR Treat pipeline
 gesture)              trick|treat}                          (perceive → plan → act)
                                              │                    │
                                              ▼                    ▼
                                        Tablet feedback      Arm + gripper
                                        (state display)      (via motion service)
                                              │
                                              ▼
                                        Episode log → Atlas
```

- **Input:** button, voice (browser Web Speech), gesture (webcam +
  detector) all produce one event, `Request{kind: trick|treat}`. Nothing
  downstream knows the modality. Button first; voice/gesture only after the
  button path works end to end.
- **Orchestrator:** `main.py`, the only component that commands the arm (D2).
- **Tablet:** input + state display — idle prompt, "doing a trick…",
  "getting your treat…", done/failed reasons (D3).
- **Data:** Viam capture + direct Atlas docs (D10). Vector search is a Day-2
  stretch (D11).

## Decisions

### D1. Stack: Viam-native

The event asks teams to build an end-to-end application using Viam, and
judges reward creative Viam + Atlas use. Pure ROS 2 via rosbridge risks
judging favor. ROS enters only inside Viam components (the
`viam-ros2-integration` module) if a specific node is ever needed. No
rosbridge tablet path is planned.

Viam's shape, for reference: the machine runs viam-server; JSON config
declares components (hardware) and services (motion, vision, data
management); your code is an SDK client calling standard APIs; the frame
system holds spatial relationships for planning; data syncs to MongoDB
(tabular) + blob storage (binary).

### D2. Orchestrator: plain asyncio state machine

`main.py`. States: IDLE, DETECTING, MOVING, MANIPULATING, DANCING, LOGGING,
FAULT. One `asyncio.Lock()` held per cycle: one routine at a time; input
while busy gets visible busy feedback, no queueing in v1. FAULT stops the
arm, homes when safe, posts the reason to the tablet, returns to IDLE. Every
routine ends DONE or FAILED with a tablet-visible reason. Speed limits and
workspace bounds live in the motion layer, not in routine code. No framework:
SM libraries buy nothing here. Behavior trees deferred — routines are short
sequences, SM handlers fit the async SDK directly, and a tree maps 1:1 onto
these states if routines later grow reactive.

### D3. Tablet: served by `main.py`

`main.py` serves the page; buttons POST to the orchestrator; the orchestrator
drives the machine via the Python SDK. Keys stay server-side, logic stays
out of the browser, logs live in one process. Alternative (documented, not
default): Vite + React SPA on Cloudflare Pages with the TypeScript SDK
direct to the machine — it needs the orchestrator exposed as a module
service (or logic in the browser), puts keys in browser code, and depends on
venue internet for the UI.

### D4. Perception: color segmenter + depth + frames

Vision Service color segmenter on bright wrappers (orange/green/purple). HSV
tuned on site under venue light — lab values won't survive hall lighting.
2D detection + RealSense depth at the centroid + frame system (camera-arm
transform, calibrated on site) gives XYZ in the base frame. The arm camera is
the grasp camera; the overview webcam is verification and recording. No
candy detected → FAILED with tablet feedback, never a blind grasp.
Stretch (Day 2, only if treat works): custom detector (e.g., YOLOv5 exported
to ONNX/TFLite via the ML model service) labeling `candy_bar` vs
`empty_wrapper`, trained on Day-1 captures.

### D5. Bucket: fixed drop zone

Tape mark on the table, constant in config, returned by a drop-target
provider interface (v1 = constant; a detected pose is a later implementation
behind the same interface). Upgrade: AprilTag on/near the bucket (6-DOF pose
from tag size + intrinsics, ~10 ms/frame), fixed zone as fallback on
occlusion. No moving-bucket tracking in v1.

### D6. Candy grasp: two-stage, verified

1. Detect candy in the candy zone (bowl ROI constrains the search).
2. Pre-grasp 12 cm directly above the candy (starting value, validate on
   site). Lateral moves happen high, so the gripper never sweeps candy.
3. Cartesian descent to grasp height. Below the D435's ~0.28 m minimum depth,
   depth is invalid: descend dead-reckoned from last good depth + a fixed
   distance.
4. Close to a fixed width slightly under candy size (position-limited, not
   max force — 30 N crushes candy).
5. Lift, verify via gripper position: fully closed (≈0) = missed → one
   retry, then FAILED with tablet feedback.
6. Motion-service transport with table, bowl, drop zone as obstacles;
   release over the drop zone; home.

### D7. Trick: joint arrays

Pre-recorded joint-position arrays stepped with small sleeps
(`asyncio.sleep`) via the arm API. Choreography, not planning: no motion
service, no IK. First dance: 4–5 poses (wave, bow, rhythmic move). Juggling
is a stretch (closed-loop throwing dynamics).

### D8. Simulation: fakes tonight, hardware Day 1

Viam fake components (fake arm/camera/gripper) via `fake-machine.json`:
tonight, CI, and parallel dev while someone else has the arm. Viam app 3D
visualization for frame/path sanity checks. No MuJoCo, no Isaac Sim (setup
cost exceeds value with hardware available Day 1 10:30). Gazebo via
`xarm_ros2` only if the ROS-inside-Viam path is ever used — not planned.
Sim cannot validate perception (fake camera = static gradient), grasp
physics, or lighting. Those three are hardware-only, scheduled Day 1.

### D9. Operating parameters

Verified (module/spec docs):

| Parameter | Value |
|---|---|
| xArm reach / repeatability | 700 mm / ±0.1 mm |
| xArm joint speed | max 180°/s; Viam default 30°/s — use ~30 for demo-safe motion |
| xArm gripper | 84 mm range, 30 N max, position range 0–840 |
| xArm6 payload | 5 kg (candy is negligible) |
| D435 depth | min ~0.28 m, ideal 0.3–3 m; RGB 1920×1080@30; depth up to 1280×720@90 |
| Control-box network | default segment 192.168.1.x; Studio proxy port 18333 |

Set on site (not guessable now): zone coordinates, camera-arm transform,
candy HSV, gripper close width, pre-grasp height, descend distance, demo
speed near visitors. Treat cycle target: under ~60 s (a target, not a
promise). Safety: one routine at a time; bounds in the motion layer; Viam
stops the arm if the connection drops.

### D10. Data: dual path, async worker

- **(a) Viam data-management capture** (arm joints, both cameras) → Viam
  cloud (tabular → MongoDB, images → blob). On from the first real run.
- **(b) Direct Atlas docs**, written by an asyncio worker draining a queue
  (DB work never blocks motion):

```json
{
  "timestamp": "2026-10-29T10:00:00Z",
  "mode": "treat",
  "target_class": "chocolate_bar",
  "coordinates": {"x": 0.25, "y": -0.12, "z": 0.05},
  "success": true,
  "duration_seconds": 12.4,
  "joint_config": [0.1, -0.5, 0.3, 0.0, 0.2, 0.0],
  "grip_width": 42,
  "candy_features": "orange wrapper, ~5cm"
}
```

Grip width is recorded, not pressure — the stock gripper is
position-controlled and reports no pressure sensor. Day-1 use: MQL queries
between runs (success by candy color, failure modes, timing) to tune the
approach.

### D11. Vector search: grasp memory, Day-2 stretch

Store an embedding per grasp attempt. On 3 consecutive failures in one table
quadrant, retrieve past successful grasps and apply their grip angle as a
bounded offset on the next attempt, shown on the tablet. The trigger is a
plain aggregation; vectors are only retrieval (HNSW index, cosine,
`$vectorSearch`; the same model must embed stored docs and queries).
Lowest-friction embedding: Atlas autoEmbed (Voyage) on a text description of
each attempt. Image embeddings need a vision model + custom writer. Viam
sync does not auto-embed. Cold start: useless until dozens of runs are
logged. Do not build before the treat pipeline is demo-solid.

### Alternative stack (documented, not chosen)

ROS 2 + rosbridge: `rosbridge_suite` exposes topics over websocket; the
tablet publishes a Request topic via roslibjs; a dispatcher node drives a
trick player and a treat pipeline (perception → MoveIt → driver). It buys
rviz, rosbag, MoveIt, and Ari's fluency. It costs full bringup + drivers +
MoveIt config in 2 days, plus a Viam bridge (or custom writer) for the Atlas
story.

## Build order

1. Trick playback. 2. Treat detection on a static zone, then the full
pipeline. 3. Tablet buttons + state display. 4. Data capture + Atlas docs
from the first real run. 5. Voice/gesture. 6. Vector grasp-memory (only
after treat is demo-solid).

## 2-day schedule

- **Day 1 AM:** network (hotspot fallback ready) → dashboard mapping → frame
  calibration → first motion from code → capture ON.
- **Day 1 PM:** color segmenter → two-stage approach → full treat pipeline.
- **Day 1 evening:** tablet page → dance script (4–5 poses) → spam test
  (hammer the buttons, prove the Lock holds).
- **Day 2 AM:** Atlas docs flowing → MQL tuning → vector memory if treat is
  solid.
- **Day 2 PM:** final-hour color re-sample under showroom lights → dress
  rehearsal (trick + treat + failure paths) → demo.

## Judging narrative

Edge perception, cloud memory: Viam computes perception locally; episode
metadata flows to Atlas; live telemetry is visible during the demo; the
3-failure vector adjustment is the proof past data changes physical behavior.
