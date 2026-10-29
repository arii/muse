# Viam hackathon prep — Trick or Treat (index)

Project: Halloween Trick or Treat arm demo. Event: Viam × MongoDB Robot
Hackathon, Oct 29–30, Robotics Center of Silicon Valley. Team hardware:
UFactory xArm or UR arm + two-finger gripper, Linux machine, RealSense D435
or Orbbec Astra 2 camera fixed to the arm, webcams, free Viam account,
MongoDB Atlas cluster.

Read order: this index → `trick-or-treat-architecture.md` (decisions) →
`viam-mongodb-ros2-research.md` (background, examples) →
`hackathon-team-setup.md` (runbooks). Product details are deliberately NOT
finalized; architecture decisions are.

## Decisions at a glance

| Decision | Verdict |
|---|---|
| Stack | Viam-native. The event is Viam + MongoDB hosted and asks for an end-to-end Viam app; pure ROS 2 via rosbridge risks judging favor. ROS only inside Viam components, if a specific node is ever needed. |
| Orchestrator | Central `main.py`, plain asyncio state machine (IDLE, DETECTING, MOVING, MANIPULATING, DANCING, LOGGING, FAULT) guarded by one `asyncio.Lock()`. No framework. Behavior trees deferred (see architecture doc). |
| Tablet wiring | Tablet page served by the `main.py` process; buttons POST to the orchestrator; orchestrator drives the machine via the Python SDK. Keys stay server-side, logic stays out of the browser. |
| Perception | Viam Vision Service color segmenter on bright wrappers + RealSense depth at centroid + frame system → XYZ in base frame. |
| Bucket | Fixed drop zone (tape mark, constant in config) behind a drop-target provider interface. AprilTag upgrade path, fixed fallback. No moving-bucket tracking in v1. |
| Candy grasp | Vision-guided top-down grasp; verify via gripper position (fully closed = missed); one retry, then FAILED with tablet feedback. |
| Trick | Pre-recorded joint arrays, raw joint control. Dance first; juggling is a stretch. |
| Simulation | Viam fake components tonight and in CI; real hardware is the primary test from Day 1 10:30 AM. No MuJoCo / Isaac Sim for this project. |
| Data | Dual path: Viam data-management capture (joints, cameras) + direct async Atlas docs (choice, timestamp, success, duration, joint configs) via a worker that never blocks motion. |
| Vector search | Day-2 stretch: grasp memory — retrieve past successful grasps for similar candy. Useless until dozens of runs are logged; do not build before the core demo works. |

## Shared vocabulary (used identically in all docs)

- **Request**: one event, `{kind: trick | treat}`, produced by any input
  modality (button, voice, gesture). Nothing downstream knows the modality.
- **Orchestrator**: the `main.py` process and its state machine. The only
  component that commands the arm.
- **Zones**: fixed, marked, measured-on-site areas — **candy zone** (bowl),
  **drop zone** (bucket mark), **home** pose. All are config constants.
- **Drop-target provider**: the interface that returns the drop pose. v1
  implementation returns the fixed drop-zone constant; a detected pose is a
  later implementation behind the same interface.
- **Episode**: one visitor interaction (request → routine → outcome), logged
  to Atlas with its camera frames and joint data.

## Tonight vs on site

- **Tonight (no hardware):** team repo skeleton, CI (ruff + pytest), fake
  machine config, orchestrator + state machine + trick playback against
  fakes, tablet page skeleton, `.env.example`. See the setup doc's tonight
  runbook.
- **On site, in order:** machine live → arm network (ping) → modules →
  cameras → frame check → first motion → data capture on → Pick and Place
  pathway → trick recording → treat calibration (zones, colors, gripper
  width). See the setup doc's hardware runbook.

## File map

- `trick-or-treat-architecture.md` — components, interfaces, hard decisions,
  operating parameters, MongoDB / vector-search role, build order.
- `viam-mongodb-ros2-research.md` — hardware-matched examples, Viam +
  MongoDB mechanics, ROS 2 assessment (kept as the documented alternative).
- `hackathon-team-setup.md` — team repo skeleton, CI, hooks, Docker verdict,
  networking, secrets, tonight + hardware runbooks.

## Round 2 refinements (Oct 28 evening, from external feedback)

- Orchestrator states refined: IDLE, DETECTING, MOVING, MANIPULATING,
  DANCING, LOGGING, FAULT — guarded by one `asyncio.Lock()`; spam test is a
  scheduled Day-1-evening item.
- Two-stage approach: pre-grasp waypoint 12 cm above the candy (starting
  value, validate on site), then Cartesian descent.
- Perception upgrade (stretch): custom detector (`candy_bar` vs
  `empty_wrapper`) via the ML model service, trained on Day-1 captures.
- Atlas schema fixed (see architecture doc R4); gripper width recorded, not
  pressure (no pressure sensor on the stock gripper).
- Vector closed loop made concrete: 3 consecutive failures in one quadrant
  → retrieve past successful grasps → bounded angle offset, shown on the
  tablet.
- UI: local-server default stands; React SPA + TS SDK direct is documented
  as the alternative with its real costs (module-service exposure, keys in
  browser code, venue-network dependence).
- Schedule: Day 1 morning bring-up, afternoon vision + grasp, evening UI +
  trick; Day 2 morning data loop, afternoon judging calibration (including
  a scheduled final-hour color re-sample under showroom lights).
