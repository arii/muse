# Viam hackathon prep — Trick or Treat (index)

Halloween Trick or Treat arm demo. Event: Viam × MongoDB Robot Hackathon,
Oct 29–30, Robotics Center of Silicon Valley. Hardware: UFactory xArm or UR
arm + two-finger gripper, Linux machine, RealSense D435 or Orbbec Astra 2
camera fixed to the arm, webcams, free Viam account, MongoDB Atlas cluster.

Read order: this index → `trick-or-treat-architecture.md` (decisions) →
`viam-mongodb-ros2-research.md` (background, examples) →
`hackathon-team-setup.md` (runbooks).

## Decisions at a glance

| Decision | Verdict |
|---|---|
| Stack | Viam-native. Pure ROS 2 via rosbridge risks judging favor; ROS only inside Viam components. |
| Orchestrator | `main.py`, plain asyncio state machine (IDLE, DETECTING, MOVING, MANIPULATING, DANCING, LOGGING, FAULT) + one `asyncio.Lock()`. No framework; behavior trees deferred. |
| Tablet | Page served by `main.py`; buttons POST to the orchestrator. Keys stay server-side. React SPA + TS SDK direct is the documented alternative. |
| Perception | Vision Service color segmenter + RealSense depth + frames → XYZ. Detector (`candy_bar` vs `empty_wrapper`) is a Day-2 stretch. |
| Bucket | Fixed drop zone behind a drop-target provider interface. AprilTag upgrade, fixed fallback. |
| Candy grasp | Two-stage (12 cm pre-grasp, then descent), gripper-position verify, one retry. |
| Trick | Pre-recorded joint arrays, 4–5 poses first. Juggling is a stretch. |
| Simulation | Viam fakes tonight + CI; hardware is the primary test from Day 1 10:30. No MuJoCo / Isaac Sim. |
| Data | Viam capture + direct async Atlas docs (never blocking motion). MQL tuning Day 1. |
| Vector search | Day-2 stretch: 3 failures in one quadrant → bounded angle offset from retrieved successful grasps. Cold start until dozens of runs logged. |

## Shared vocabulary

- **Request:** one event, `{kind: trick | treat}`, from any modality. Nothing
  downstream knows the modality.
- **Orchestrator:** the `main.py` process and its state machine. The only
  component that commands the arm.
- **Zones:** fixed, marked, measured-on-site areas — **candy zone** (bowl),
  **drop zone** (bucket mark), **home** pose. All are config constants.
- **Drop-target provider:** returns the drop pose. v1 = fixed constant; a
  detected pose is a later implementation behind the same interface.
- **Episode:** one visitor interaction (request → routine → outcome), logged
  to Atlas with camera frames and joint data.

## Tonight vs on site

- **Tonight (no hardware):** repo skeleton, CI (ruff + pytest), fake machine
  config, orchestrator + trick playback against fakes, tablet skeleton,
  `.env.example`. See the setup doc's tonight runbook.
- **On site:** machine live → arm network (ping) → modules → cameras →
  frame check → first motion → capture on → Pick and Place pathway → trick
  recording → treat calibration. See the setup doc's hardware runbook.
- **Schedule:** Day 1 AM bring-up, PM vision + grasp, evening UI + trick +
  spam test; Day 2 AM data loop, PM final-hour color re-sample + rehearsal.

## File map

- `trick-or-treat-architecture.md` — components, decisions, parameters,
  build order, schedule, judging narrative.
- `viam-mongodb-ros2-research.md` — hardware map, examples, MongoDB
  mechanics, ROS 2 assessment, connect pattern, open on-site questions.
- `hackathon-team-setup.md` — repo skeleton, CI, hooks, Docker verdict,
  secrets, workflow, runbooks, networking checklist.
