# Hackathon team repo setup — Viam hackathon prep

Index + decisions: see `README.md` in this directory. Runbooks are at the end
of this file.

Date: 2026-10-28. Situation: Ari will most likely create a new git repo with his
2–3 person Viam hackathon team. Note: the Muse GitHub connector cannot create
repos — Ari creates the repo himself (public or private; either works, keys
never go in it).

Principle for a 2-day hackathon: everything that doesn't need hardware gets set
up **tonight**; everything that needs hardware happens **on site, in order**.
CI and hooks are guardrails, not process — keep them fast or teammates will
route around them.

## Tonight (no hardware needed)

### Repo skeleton

```
.
├── README.md                  # hardware map, quickstart, demo command
├── pyproject.toml             # viam-sdk, pymongo, pytest, ruff
├── .env.example               # API_KEY, API_KEY_ID, MACHINE_ADDRESS, ATLAS_URI (placeholders only)
├── .gitignore                 # .env, .venv, __pycache__, data/, *.mp4, *.bag
├── .pre-commit-config.yaml    # ruff only
├── .github/workflows/ci.yml   # ruff + pytest, no hardware
├── Dockerfile                 # app only, optional (see below)
├── config/
│   ├── machine.example.json   # reference copy of the real machine config (no secrets)
│   ├── fake-machine.json      # fake arm/camera/gripper for CI + offline dev
│   └── zones.json             # candy zone, drop zone, home pose (measured on site)
├── src/
│   ├── main.py                # orchestrator: asyncio state machine + Lock, serves tablet page
│   ├── connect.py             # env-based RobotClient connect (see research notes)
│   ├── treat.py               # detect → approach → grasp → transport → release
│   ├── trick.py               # recorded joint-sequence playback
│   ├── vision.py              # vision-service helpers
│   └── atlas_worker.py        # async queue drain writing episode docs to Atlas
├── tablet/
│   └── index.html             # Trick / Treat buttons + state display (no API keys)
├── scripts/
│   ├── run.sh                 # venv + env + run main.py
│   └── check_config.py        # json.tool validation of config files
└── tests/
    ├── test_main.py           # state machine + Lock logic, no hardware
    └── test_treat.py          # treat pipeline logic against fakes/mocks, no hardware
```

### GitHub Actions: one minimal workflow

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff pytest viam-sdk pymongo
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: pytest tests -q
```

Deliberately NOT in CI: anything touching hardware, Docker builds, or a live
Viam machine. CI must stay under a few minutes or it stops being consulted.
Viam's built-in `fake` models (fake arm, fake camera returning a static
gradient image + point cloud, fake gripper, fake board) exist exactly so logic
and motion-planning code can run without hardware — the motion-planning
quickstart itself uses fake components. Use `fake-machine.json` for any test
that loads a machine config.

### Git hooks: ruff only, via pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.0
    hooks:
      - id: ruff
      - id: ruff-format
```

Every teammate runs `pre-commit install` after cloning. Hooks are local-only —
they don't travel with the clone — so CI is the backstop, not the hook.
Do NOT put tests, Docker builds, or config validation in hooks: a slow or
flaky hook gets uninstalled mid-hackathon, and then you have neither the hook
nor the discipline.

### Docker: app container optional, viam-server stays native

- **viam-server: install natively on the provided Linux box** (Viam's
  recommended path is `viam-agent`). Do not run viam-server in Docker at the
  event: USB camera passthrough (`--device=/dev/video0` or `--device=/dev/bus`)
  breaks if a camera is unplugged after container start, hot-plugged cameras
  need cgroup rules, and `--privileged` is often the only thing that works.
  That is debugging time the team doesn't have.
- **Team app: Docker is optional and safe**, because the app talks to
  viam-server over the network via `RobotClient` — it needs no USB devices and
  no Ethernet link to the arm. If containerized, the Dockerfile is just
  `python:3.11-slim` + `pip install viam-sdk pymongo` + copy `src/`, with credentials
  passed as environment variables at run time, never baked into the image.
- **Default recommendation: skip Docker for the app too.** A venv on the
  laptop or the Linux box is one fewer moving part. Add the Dockerfile only
  if a teammate's environment differs enough to justify it.

### Secrets

- Real values live in `.env` (gitignored) on each person's machine, or in
  GitHub Secrets if CI ever needs a live machine (it shouldn't).
- `.env.example` carries the variable names and placeholder values only.
- Source of truth for the real machine config is the Viam app, not the repo:
  the repo's `machine.example.json` is a backup/reference. If the config
  changes on site, copy it back to the repo at the end of the day.
- Viam API keys come from the machine's CONNECT tab in the Viam app.

### Workflow for 2 days, 2–3 people

- Trunk-based on `main` with CI on push; small, frequent commits. Branch
  protection and mandatory review cost more than they save in 48 hours.
- Exception: machine-config changes get announced in the team channel (or a
  quick PR) before landing — one bad config edit breaks the robot for
  everyone, and config is the shared state.
- Hardware contention is the real bottleneck, not merge conflicts: only one
  person drives the arm at a time. Agree on arm time slots; everyone else
  develops against the fake config.

## On site, in order (first 60–90 min, Day 1)

1. **Machine live.** Provided Linux box: viam-server installed, machine shows
   Live in the Viam app. Team's laptops need nothing but the repo + `.env`.
2. **Arm network.** Ethernet from control box to the Linux box. Arm IP from
   the sticker on the control box (xArm default segment is 192.168.1.x).
   Give the Linux box an IP on the same subnet and verify:
   `sudo ip addr add 192.168.1.10/24 dev eth0 && ping <arm-ip>`.
   In the Viam xArm config, `host` = the arm's IP, not the box's own IP.
   If the box also uses WiFi (for Viam cloud + laptop access), its WiFi and
   Ethernet addresses must be on different segments.
3. **Arm in the app.** Add the xArm module (`ufactory/xArm6` etc.), set `host`,
   test joint motion from the CONTROL tab. The module also offers a
   UFactory Studio proxy (port 18333) for the arm's web UI.
4. **Cameras in the app.** Arm camera via the RealSense module (or the Astra 2
   module — verify the name on site); webcam via built-in `camera/webcam`.
   Name them distinctly (`arm-cam`, `overview-cam`); set explicit `video_path`
   values with two USB cameras. Verify both feeds in CONTROL.
5. **First motion from code.** Run the repo's connect script from a laptop
   (remote `RobotClient` connection, no code on the box needed) and move the
   arm to a safe pose. If this works, the whole architecture is proven.
6. **Data capture on.** Enable the data management service for arm joints and
   both cameras before real runs start — capture is off by default, and
   episodes from hour one are the demo's evidence and the Atlas story.
7. **Then the Pick and Place pathway** (xArm6 + RealSense D435, ~90 min) as
   the first working vertical slice, before custom features.

## Networking checklist (copy to the team channel)

- [ ] Arm control-box IP read off the sticker; Linux box on the same subnet
- [ ] `ping <arm-ip>` succeeds from the Linux box
- [ ] Viam machine shows Live; laptops connect via `RobotClient` from `.env`
- [ ] Both camera feeds visible in the Viam app CONTROL tab
- [ ] Data management service enabled for arm + both cameras
- [ ] `.env` present on each laptop, real keys in no committed file
- [ ] CI green on `main` before the first on-robot run

## Runbooks

### Tonight runbook (no hardware; do in order)

```bash
# 1. Ari creates the team repo (connector cannot create repos); teammates clone it
git clone <team-repo-url> && cd <repo>

# 2. Python env + deps (py_trees NOT needed: the orchestrator is a plain
#    asyncio state machine; pymongo is for the direct Atlas docs)
python -m venv .venv && source .venv/bin/activate
pip install viam-sdk pymongo pytest ruff
pip install pre-commit && pre-commit install

# 3. Skeleton per this doc, then fake machine config for offline dev
#    config/fake-machine.json: fake arm, fake camera, fake gripper
#    (fake camera returns a static gradient image: it exercises the
#    failure path — detection finds nothing — which is itself a useful test)

# 4. Prove the loop without hardware
pytest tests -q          # logic + fake-config tests
ruff check src tests && ruff format --check src tests
python src/main.py --config config/fake-machine.json
#    Expect: tablet page serves, trick playback runs on the fake arm,
#    treat attempt FAILS cleanly (no candy in a gradient image) with
#    tablet feedback. If any of that fails, fix it tonight — not on site.

# 5. Tablet page skeleton: two buttons (Trick / Treat) + state display.
#    Buttons POST to the orchestrator; no API keys in the page.
```

Cannot do tonight (needs the event): the Atlas connection string (provided
at the event — put it in `.env`, never commit it), the arm's IP, camera
modules, zone coordinates, candy colors, gripper width.

### First hardware run runbook (Day 1, in order)

```bash
# 1. Machine live: viam-server on the provided Linux box shows Live in the app
# 2. Arm network: Ethernet control box -> Linux box; IP off the sticker
sudo ip addr add 192.168.1.10/24 dev eth0   # adjust interface + subnet to the sticker
ping <arm-ip>                               # must succeed before touching the app
# 3. In the Viam app: add xArm module (host = arm IP), RealSense module,
#    webcam component (explicit video_path values for both cameras)
# 4. CONTROL tab: move each joint a little, open/close the gripper,
#    confirm both camera feeds
# 5. Frame check: confirm the camera-arm transform in the frame system
#    (measure on site; a wrong transform sends every grasp to the wrong place)
# 6. First motion from code: run main.py from a laptop (remote RobotClient
#    connection via .env) and move to a safe pose — this proves the architecture
# 7. Data capture ON (data management service: arm joints + both cameras)
#    BEFORE real runs start — hour-one episodes are the demo evidence
# 8. Pick and Place pathway (xArm6 + RealSense D435, ~90 min) as the first
#    vertical slice, then trick recording, then treat calibration:
#    zones (tape marks), candy HSV under venue light, gripper close width
```

If step 6 fails, stop and debug connectivity (arm IP, subnet, .env) before
writing more code — every later step assumes it.

### Round 2 additions (from feedback)

- **Network fallback:** if venue WiFi blocks outbound traffic or Viam cloud
  access, put the Linux box's internet link on a phone hotspot. The arm's
  Ethernet link is separate — keep the two links on different segments.
- **Spam test (Day 1 evening):** hammer the Trick/Treat buttons rapidly and
  confirm the orchestrator's `asyncio.Lock()` holds: one routine at a time,
  busy feedback, no fractured motion.
- **Final-hour calibration (Day 2 afternoon, scheduled):** re-sample wrapper
  color values under the showroom spotlights before judging. Do not leave
  this to "if time" — hall lighting is the biggest single threat to color
  detection.
