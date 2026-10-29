# Trick or Treat — architecture notes (Viam hackathon project)

Index + decisions: see `README.md` in this directory and the "Hard decisions"
section below. Product details are deliberately NOT finalized; architecture
decisions are.

Date: 2026-10-28. Status: architecture sketch only. Per Ari: do NOT finalize
product details or design requirements yet — this file covers components and
interfaces, not the final interaction design.

## The idea (one paragraph)

Halloween-themed arm demo. A visitor at a monitor/tablet next to the arm
requests TRICK or TREAT. Trick: the arm performs a pre-choreographed routine
(dancing is the feasible example; juggling is a stretch — see below). Treat:
the arm picks up a candy and drops it into the visitor's bucket. Every visitor
interaction is a complete, demoable story, and every run is logged.

## Component diagram (stack-independent)

```
Input modalities ──► Request event ──► Dispatcher ──► Trick player
(button / voice /       {kind:            (state         OR Treat pipeline
 gesture)              trick|treat}        machine)           (perceive → plan → act)
                                              │                    │
                                              ▼                    ▼
                                        Tablet feedback      Arm + gripper
                                        (state display)      (via motion service)
                                              │
                                              ▼
                                        Episode log → Atlas
```

### 1. Input layer: many modalities, one event type

Button tap, voice command, and gesture recognition are interchangeable
producers of a single event: `Request{kind: trick | treat}`. Nothing
downstream knows or cares which modality produced it. This is what keeps the
modality choice unfinalized without blocking the rest of the build:

- **Button:** trivial in both stacks (a web page button).
- **Voice:** can run entirely in the tablet browser (Web Speech API) in both
  stacks — no server-side speech component needed. Browser hears a phrase,
  emits the same `Request` event.
- **Gesture:** needs a vision component (webcam + a gesture detector /
  classifier). This is the only modality that adds a real component; the
  others are UI code.

### 2. Dispatcher: the only component that commands the arm

A small state machine in `main.py` (states refined in R1: IDLE, DETECTING,
MOVING, MANIPULATING, DANCING, LOGGING, FAULT, guarded by one
`asyncio.Lock()`). Rules:

- One routine at a time. A request arriving while busy gets visible "busy"
  feedback (queueing is an option, not a decision).
- Every routine ends in DONE or FAILED with a tablet-visible reason, and the
  arm returns to a home pose on either outcome. No silent hangs, no infinite
  retries (treat: one retry on a missed grasp, then FAILED).
- Arm speed limits and workspace bounds live in the motion layer, not in
  routine code, so a bad routine can't exceed them.

### 3. Trick player: open-loop trajectory playback

Tricks are recorded joint trajectories played back, not planned live:

- Record by teaching: move the arm by hand / jog it through the routine,
  save poses + timing, play back as a joint-position sequence. First dance:
  4–5 poses (wave, bow, rhythmic move); more poses only after the first
  routine works.
- Dancing fits this exactly: choreographed joint trajectory, no perception.
- **Juggling is a stretch, not a plan.** It needs high-frequency closed-loop
  control and throwing dynamics — a different control problem from trajectory
  playback. Keep it as an aspiration; the feasible trick library is dances
  and waves.
- Trick routines need no camera, no planning, and no candy. They are the
  safe fallback if perception isn't working yet — build a trick first.

### 4. Treat pipeline: closed-loop perceive → plan → act

1. **Detect:** vision service finds a candy in the candy zone (arm camera or
   overview cam). Failure mode is explicit: "no candy seen" → FAILED with
   tablet feedback, not a blind grasp.
2. **Plan:** motion service plans a collision-free path to the grasp pose,
   with the table, candy bowl, and bucket zone as declared obstacles.
3. **Grasp:** gripper closes (position-limited — candy is crushable, so
   close to a set width, not max force).
4. **Transport + release:** move to the drop target, open gripper, return home.
- **Zones, calibrated on site:** candy zone, drop target (bucket spot),
  home pose. Whether the bucket sits in a fixed spot or is held by the
  visitor is a product detail — architecturally the drop target is just a
  configured zone, optionally detected.
- **Candy supply is a human ops task.** The routine assumes candy is in the
  zone; refilling the bowl is not a robot component.

### 5. Tablet: input + feedback in one screen

The monitor/tablet next to the arm is both the input device and the
state display: idle prompt, "doing a trick…", "getting your treat…",
done/failed reasons. Visitors need to see state, not just press a button.
The tablet runs a plain web page served by `main.py` (see D3); the
TS-SDK-direct alternative is documented in R6.

### 6. Data layer: the "Data" in Physical AI & Data

Log every request, routine, outcome, timing, and camera frames via Viam data
management → Atlas (see research notes). This is demo evidence and the
judging story for the data half of the theme. Viam stays in the loop for
capture by decision (D1); the direct Atlas docs are specified in D10/R4.

## Path A: Viam-native (the stack — see D1)

- Tablet page served by `main.py` (same pattern as the S5 buttons page);
  buttons POST to the orchestrator, which calls the Python SDK. Voice =
  browser Web Speech → same POST. Gesture = webcam + vision-service
  detector → same POST. The TS-SDK-direct alternative is documented in R6.
- Dispatcher: Python app on the laptop or Linux box using `Arm`, `Gripper`,
  `Camera`, `VisionClient`, and the motion service.
- Trick = recorded joint-position sequence via `move_to_joint_positions`.
- Treat = vision detections → motion-service move → gripper open/close.
- Data capture is built in (data management service).
- Open question: is there a registry voice/gesture component worth using,
  or is browser-side voice + a small custom gesture detector enough?
  Default to browser-side voice; gesture is the only custom vision work.

## Path B: ROS 2 + rosbridge (documented alternative — NOT the stack, see D1)

The mechanism, kept for reference in case the team ever needs it: the tablet
talks over rosbridge.

- `rosbridge_suite` on the Linux box exposes ROS topics/services over a
  websocket. The tablet page connects with roslibjs and publishes a
  `Request` topic (or calls a service) — JSON over websocket, no app server
  to write. Button tap = publish. Voice = browser Web Speech → publish.
  Gesture = a perception node publishing the same topic.
- ROS graph: request topic → dispatcher node (state machine) → trick player
  node (trajectory playback, e.g. recorded joint states) / treat pipeline
  nodes (perception → MoveIt planning → arm driver + gripper action).
- What ROS 2 buys: rviz debugging, rosbag recording/replay, MoveIt, and
  Ari's existing fluency. What it costs: full ROS 2 bringup + arm drivers
  (xArm / UR) + MoveIt config on the provided box, inside a 2-day window.
- **The data catch:** if Viam leaves the loop, the Atlas story needs a
  custom MongoDB writer. The fix is running Viam alongside via the
  `viam-ros2-integration` module so ROS topics still land in Atlas capture —
  but that is a second integration to debug, not a freebie.

## What the stack choice changed (decided: Viam-native — kept for reference)

| | Viam-native | ROS 2 + rosbridge |
|---|---|---|
| Tablet transport | SDK call / local server | rosbridge websocket + roslibjs |
| Arm control | Viam modules + motion service | Arm driver + MoveIt |
| Trick playback | Joint-position sequence | Recorded joint states / trajectory |
| Treat perception | Vision service | Perception node |
| Data → Atlas | Built in | Via integration module, or custom writer |
| First-90-min risk | Module config | ROS bringup + drivers |

Decided (D1): Viam-native. The table above is kept for reference — it shows
what the choice would have changed. The components and the `Request` event
are the same either way; only the first 90 minutes of Day 1 differ (module
config vs ROS bringup).

## Build order — superseded by the revised build order at the end of the Hard decisions section.

## Hard decisions (Oct 28 evening, incl. external feedback)

The feedback on this project settles the stack question and sharpens several
components. Product details remain unfinalized; these are architecture
decisions, and they supersede the "both paths open" framing above.

### D1. Stack: Viam-native (decided)

The event is hosted by Viam and MongoDB and asks teams to build an end-to-end
application using Viam as the core platform. Pure ROS 2 via rosbridge is
technically possible but risks judging favor, since judges evaluate creative
uses of Viam and MongoDB Atlas. So: Viam-native is the stack. ROS 2 survives
only as Viam's built-in ROS 2 integration — wrapping a specific ROS node
inside a Viam component — if a concrete need for it appears. It is not the
backbone, and no rosbridge tablet path is planned.

Viam's recommended architecture, for reference: one machine runs viam-server
(the RDK); everything is a resource in JSON config (components = hardware,
services = software like motion, vision, data management); your code is a
client (Python/Go/TypeScript SDK) that calls standard component/service APIs;
the frame system holds spatial relationships for motion planning; data syncs
to MongoDB (tabular) + blob storage (binary).

### D2. Orchestrator: plain asyncio state machine, not a behavior tree (decided)

The feedback proposes a central orchestrator (`main.py`) with states IDLE,
PROCESSING_TREAT, DANCING, LOGGING. Adopted (states refined in R1), with one reconciliation to the
earlier state-machine-vs-behavior-tree question:

- **Decision: hand-written asyncio state machine, no framework.** The
  routines here are short sequences (treat is ~8 steps); a state machine's
  handlers are naturally async functions, which fits the Viam Python SDK
  directly. Behavior trees (py_trees) are the stronger tool for long-running,
  reactive autonomy with deep fallback logic — and Ari knows them — but they
  add a library plus a sync-tick/async-SDK bridging problem for logic this
  short. If routines later grow reactive (e.g., dynamic re-planning
  mid-grasp), the tree structure maps 1:1 onto these states and can replace
  the routine internals without touching the orchestrator's interface.
- Off-the-shelf SM libraries (`transitions`, `python-statemachine`,
  `aiostate`) exist and support asyncio, but buy nothing essential here:
  declarative states and diagrams don't justify a dependency in a 2-day
  sprint. Plain Python it is.
- Tablet requests arriving while busy get visible "busy" feedback; no
  queueing in v1.

### D3. Tablet wiring: local server, not SDK-direct (decided)

Two Viam-native wirings exist. The feedback's "tablet talks directly to the
orchestrator via WebRTC/SDK" is roughly the second one; the decision is the
first:

- **Chosen: `main.py` serves the tablet page and exposes endpoints.**
  Button tap → HTTP POST to the orchestrator → orchestrator drives the
  machine via the Python SDK. Rationale: API keys stay server-side (never in
  a tablet browser page), routine logic stays out of the browser (a crashed
  tab can't kill the demo), and all logs live in one process.
- **Not chosen: TypeScript SDK direct from the tablet to the machine.**
  Fewer moving parts, but keys in browser code and demo logic in a web page.
  Revisit only if the team strongly prefers zero server code.

### D4. Perception: color segmenter + depth + frames → XYZ (decided)

- Viam Vision Service with a color segmenter on brightly colored wrappers
  (orange/green). HSV ranges tuned on site under venue lighting — lab-tuned
  thresholds will not survive hall lighting.
- 2D detection + RealSense depth at the detection centroid + the frame
  system (camera-arm transform, calibrated on site) yields XYZ in the arm
  base frame, which is what the motion service consumes. The feedback's
  "Transform Camera converts 2D to 3D automatically" is simplified: the
  conversion is detection + depth + frames, and the exact wiring (vision
  detections combined with the camera's point cloud / depth frame) is
  verified on site, not assumed.
- Overview webcam is the second viewpoint (verification, demo recording);
  the arm camera is the grasp camera. If no candy is detected, the treat
  routine FAILS with tablet feedback — no blind grasp into the bowl.

### D5. Bucket: fixed drop zone (decided, unchanged)

Fixed coordinate above the bucket mark (tape on the table, constant in
config), returned by the drop-target provider interface. Upgrade path:
AprilTag on/near the bucket (`apriltag-python`, tag36h11, ~10 ms per still
frame) giving full 6-DOF pose from known tag size + camera intrinsics, with
the fixed zone as fallback when the tag is occluded. No moving-bucket
tracking in v1 — a held bucket is out of scope.

### D6. Candy grasp: vision-guided top-down, verified (decided)

1. Detect candy in the candy zone (bowl ROI constrains the search area).
2. Pre-grasp above the candy, descend to grasp height. Below the D435's
   minimum depth (~0.28 m) depth is invalid, so the final descent is
   dead-reckoned from the last good depth + a fixed descend distance.
3. Close the two-finger gripper to a fixed width (slightly less than the
   candy size, measured on site) — position-limited, not max force, because
   30 N crushes candy.
4. Lift, verify via gripper position feedback: fully closed (≈0) means it
   grabbed nothing → one retry, then FAILED with tablet feedback.
5. Transport via the motion service with table, bowl, and bucket zone as
   declared obstacles; release over the drop zone; return home.

### D7. Trick: joint arrays, raw control (decided, unchanged)

Pre-recorded joint-position arrays stepped with small sleeps through the arm
API. This is choreography, not planning — no motion service, no IK. First
dance: 4–5 poses (wave, bow, rhythmic move). Dance
first; juggling stays a stretch (closed-loop throwing dynamics, different
control problem).

### D8. Simulation: fakes tonight, hardware from Day 1 10:30 (decided)

- **Tonight + CI + parallel dev:** Viam fake components (fake arm, fake
  camera, fake gripper) via `fake-machine.json`, plus the Viam app's 3D
  visualization to sanity-check frames and planned paths before executing.
- **ROS path only:** Gazebo via `xarm_ros2` (`xarm_gazebo`,
  `xarm_moveit_config` fake/Gazebo launches) — not planned, since the stack
  is Viam-native.
- **Not used:** MuJoCo (UR5e/xArm7 models exist, and Ari's grasping sim is
  MuJoCo, but adapting a scene costs more than it saves when real hardware
  is available from 10:30 AM Day 1) and Isaac Sim (GPU + setup cost, no
  verified fast path for this hardware).
- **What sim cannot validate:** perception (the fake camera returns a static
  gradient image — it cannot test candy detection), grasp physics (the fake
  gripper has no contact), and lighting. Those three are hardware-only tests
  and must be scheduled on Day 1, not discovered during the demo.

### D9. Operating parameters

Verified values (from module/spec docs):

| Parameter | Value |
|---|---|
| xArm reach / repeatability | 700 mm / ±0.1 mm |
| xArm joint speed, max | 180°/s (Viam module default `speed_degs_per_sec`: 30 — use ~30°/s for demo-safe motion) |
| xArm gripper | 84 mm range, 30 N max clamping, position range 0–840 |
| xArm payload (6) | 5 kg (candy is negligible) |
| D435 depth | min ~0.28 m, ideal 0.3–3 m; RGB 1920×1080@30; depth up to 1280×720@90 |
| xArm control-box network | default segment 192.168.1.x; Studio proxy port 18333 |

Set on site (not guessable now): zone coordinates (candy, drop, home);
camera-arm transform; candy HSV ranges under venue light; gripper close
width for the actual candy; pre-grasp height and descend distance; demo
speed scaling near visitors. Cycle-time target for a treat run: under ~60 s
(a target to validate, not a promise). Safety: one routine at a time;
workspace bounds as motion constraints; operator stop via the Viam app;
Viam stops the arm if the connection drops.

### D10. Data: dual path, async worker (decided)

Two complementary MongoDB paths — do not conflate them:

- **(a) Viam data-management capture** (automatic): arm joints + both cameras,
  synced to Viam's cloud (tabular → MongoDB, images → blob). Turn it on
  before the first real run; it is the sensor record.
- **(b) Direct Atlas docs** (team-written): every episode writes a document
  to the team's provided Atlas cluster via `pymongo` — timestamp, choice
  (trick/treat), success, duration, joint configurations, candy features.
  Writes go through an asyncio queue drained by a background worker, so
  database work never blocks arm motion (the feedback's telemetry pipeline).
- **Day 1 use:** MQL queries over (b) between runs — success rate by candy
  color, failure modes, timing — to tune the approach. This is exactly the
  event's "query it to tune your approach between runs" suggestion, and it
  needs no embeddings at all.

### D11. Vector search: grasp memory, Day-2 stretch (decided)

Adopted from the feedback, with its costs stated plainly. The hook: store an
embedding with each grasp attempt (candy features + grip width + approach
angle + outcome); when the robot struggles with an odd-shaped candy, query
Atlas Vector Search for the nearest past successful grasps and reuse their
grip configuration. Mechanics: a vector index on the embedding field
(HNSW, cosine similarity) + `$vectorSearch` aggregation; the same model must
embed stored docs and queries.

- **Lowest-friction path:** Atlas Automated Embedding (Voyage models) on a
  text field — embed a text description of each attempt on insert, no
  external pipeline. Image embeddings need a vision embedding model and a
  custom writer; Viam's sync does not auto-embed anything.
- **Cold start:** vector search is useless until dozens of runs are logged.
  It only pays off by demo afternoon, and only if the core demo already
  works. Build order: episode logging first (Day 1), vector grasp-memory
  only after the treat pipeline is demo-solid.
- **What it is not:** a replacement for the Day-1 MQL tuning loop, and not a
  reason to delay the core demo. It is the differentiator slide, not the
  foundation.

### Build order (revised)

1. Trick playback (works with no perception).
2. Treat detection on a static candy zone, then the full pipeline.
3. Tablet page with button input + state display.
4. Data capture + Atlas docs from the first real run.
5. Voice / gesture input (button path must work end to end first).
6. Vector grasp-memory (only after the treat pipeline is demo-solid).

## Feedback round 2 — refinements (Oct 28 evening)

A second feedback round (execution roadmap) is incorporated here. Most of it
confirms D1–D11; the items below refine or add to them. One conflict is
reconciled explicitly rather than papered over.

### R1. Orchestrator states, refined (replaces the D2 state list)

States: IDLE → DETECTING → MOVING → MANIPULATING → LOGGING → FAULT, plus
DANCING for the trick routine (IDLE → DANCING → LOGGING). FAULT is an
explicit state, not just an outcome: on entry it stops the arm, returns home
when safe, posts the reason to the tablet, then transitions back to IDLE.
Atomicity mechanism (adopted): a single `asyncio.Lock()` held for the whole
cycle — one routine at a time, and tablet input arriving while the lock is
held gets "busy" feedback instead of queueing. Phase 5 test: spam the UI
buttons rapidly and prove the lock holds without fracturing a motion.

### R2. Two-stage approach: 12 cm pre-grasp (refines D6)

The treat approach is two stages, not one continuous move: (1) motion-service
move to a pre-grasp waypoint 12 cm directly above the candy, (2) straight
Cartesian descent to grasp height. The 12 cm value is a starting value to
validate on site (bowl depth and candy height vary); the principle is what
matters — the lateral move happens high, so the gripper never sweeps candy
across the table. Descent below the D435's ~0.28 m minimum depth remains
dead-reckoned, per D6.

### R3. Perception upgrade: custom detector, Day 2 (extends D4)

Day-1 perception is the color segmenter (D4). The upgrade, only if time
permits: a custom detector (e.g., YOLOv5 exported to ONNX/TFLite) deployed
via the ML model service + vision service, trained to label `candy_bar` vs
`empty_wrapper` so the arm ignores junk. Training data comes free from Day
1: the data-management capture already records every frame. This is a
stretch, not a Day-1 dependency — the color segmenter must work first.

### R4. Atlas schema (extends D10)

Episode documents in the team's Atlas cluster use this shape (merged from
the feedback's example and D10's fields):

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

Note the correction: the schema records gripper width (position), not grip
pressure — the stock two-finger gripper is position-controlled and reports
no pressure sensor.

### R5. Vector closed loop: the 3-failure trigger (makes D11 concrete)

The vector grasp-memory hook gets a specific trigger rule: if a treat fails
3 times consecutively in the same table quadrant, the next attempt queries
Atlas Vector Search for past successful grasps in similar situations and
applies their grip angle as an offset to the grasp target angle. Bounds on
the correction: clamp the offset (small, e.g., a few degrees — set on site),
apply it to one attempt, and show on the tablet that the angle was adjusted
from past data (that visibility is judging gold). The trigger itself needs
no vectors — it is a plain aggregation over the episode docs; vectors are
only the retrieval step. Cold-start and Day-2-only caveats from D11 still
apply.

### R6. UI conflict, reconciled (D3 stands; SPA documented as alternative)

The feedback proposes a Vite + React SPA hosted on Cloudflare Pages, talking
to the machine directly via the TypeScript SDK over WebRTC. The docs keep
the D3 default (page served by `main.py`, buttons POST to the orchestrator)
for these reasons:

- The SPA path has a hidden requirement: "query your script actions" from a
  browser only works if the orchestrator's functions are exposed through the
  machine API — i.e., the orchestrator becomes a Viam module service — or
  if the routine logic moves into the browser. Module packaging (or
  browser-resident logic) is extra work in a 2-day sprint, and it is not
  free.
- A Cloudflare-hosted page puts the API key in browser code and depends on
  venue internet for the UI path; the feedback itself warns about corporate
  WiFi pitfalls. The local-server page works on the LAN with keys
  server-side.
- What both paths agree on (and what matters for the demo): two buttons +
  state display. Only transport and hosting differ.

The SPA path remains documented as the alternative if the team prefers it —
it is a legitimate Viam-native choice, just not the default here.

### R7. Networking fallback + final-hour calibration (extends the runbooks)

- If venue WiFi blocks outbound traffic or Viam cloud access, fall back to
  a phone hotspot for the Linux box's internet link. The arm's Ethernet link
  is separate and unaffected — the two links must stay on different
  segments (see the setup doc).
- Final-hour calibration is scheduled, not "if time": re-sample wrapper
  color values under the showroom spotlights in the last hour before
  judging. Hall lighting changes color detection more than any other
  single factor.

### 2-day schedule (from the feedback, mapped to the decisions)

- **Day 1 morning — bring-up:** network (hotspot fallback ready) → Viam
  dashboard mapping (arm, gripper, RealSense) → spatial calibration
  (camera mounting offset in the frame system) → first motion from code →
  data capture ON.
- **Day 1 afternoon — vision + grasp:** color segmenter → two-stage
  approach (12 cm pre-grasp) → full treat pipeline on the real candy zone.
- **Day 1 evening — UI + trick:** tablet page + buttons → dance script
  (4–5 joint poses, `asyncio.sleep` steps) → spam test for the lock.
- **Day 2 morning — data loop:** Atlas docs flowing → MQL tuning queries →
  vector grasp-memory only if the treat pipeline is demo-solid.
- **Day 2 afternoon — judging calibration:** final-hour color re-sample →
  full dress rehearsal (trick + treat + failure paths) → demo.

### Judging narrative (one paragraph, for the demo)

Edge perception, cloud memory: Viam computes perception transforms locally
on the machine while episode metadata flows to Atlas; live telemetry is
visible during the demo; and the 3-failure vector adjustment (R5) is the
proof that past data changes physical behavior — the closed loop the event
is named for.
