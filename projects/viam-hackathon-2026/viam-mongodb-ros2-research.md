# Viam + MongoDB + ROS 2 research — Viam hackathon prep

Index + decisions: see `README.md` in this directory. This file is background
and examples; the architecture doc holds the decisions.

Date: 2026-10-28. Event: Viam Robot Hackathon, Oct 29–30, Robotics Center of Silicon Valley.
Team hardware (per event listing): UFactory xArm or UR arm + two-finger gripper,
Linux machine, RealSense D435 or Orbbec Astra 2 camera fixed to the arm, webcams,
free Viam account, MongoDB Atlas cluster.

## Hardware map: event hardware → Viam component

| Hardware | Viam path | Notes |
|---|---|---|
| xArm 6 / 7 / 850 / Lite 6 | `viam-modules/viam-ufactory-xarm` registry module | Add via app CONFIGURE tab: search `ufactory/xArm6` (or xArm7 / xArm850 / lite6). Only required attribute: `host` = control-box IP (sticker). Gripper models ship in the same module. |
| UR arm | Built into the RDK (`arm/universalrobots`) | Confirm the exact model name on site; not verified from the registry listing here. |
| RealSense D435 (D415 / D435i / D455 also listed) | `viam-modules/viam-camera-realsense` (C++ module) | Ubuntu recommended: `librealsense` is officially supported on Ubuntu only. Python: `Camera.from_robot(robot, "myRealSense")`. |
| Orbbec Astra 2 | Viam case study confirms Astra 2 works with Viam | Exact registry module name NOT verified here — check on site. Without a module, RGB via the `webcam` model may work, but depth needs the proper module. |
| Webcam (second camera) | Built-in `camera/webcam` model, no module needed | Leave `video_path` blank for a single webcam. With two USB cameras, set explicit `video_path` values so identities don't swap. |

## Start-here examples (exact hardware match)

- **Viam "Pick and Place" pathway** — Python SDK + vision models: an arm
  detects, picks up, and places blocks on its own. Hardware: uFactory xArm6 +
  gripper + Intel RealSense D435. ~90 min. Closest existing example to the
  hackathon setup; run it first on site.
  <https://docs.viam.com/operate/hello-world/first-project/>
- **Cube-sorter tutorial** — Lite 6 + vacuum gripper + RealSense 435, color
  detection + motion planning + custom web app. Useful for the detect →
  plan → place loop and the demo UI pattern.
  <https://www.viam.com/post/use-viam-to-build-a-robot-that-can-detect-pick-up-and-sort-colored-blocks-full-tutorials>
- **xArm module reference** (attributes, networking, frame system, gripper,
  UFactory Studio proxy on port 18333):
  <https://pkg.go.dev/github.com/viam-modules/viam-ufactory-xarm>
- **Vision service pattern** — `VisionClient.get_detections_from_camera()` /
  `get_classifications_from_camera()`, used to gate arm motion on detections:
  <https://docs.viam.com/data-ai/ai/act/>

## MongoDB path: what "Viam + MongoDB" actually means

No custom pipeline needed. Viam's data management service is the integration:

1. **Capture on the machine.** Enable the service; choose components (arm
   joints, camera frames) and frequency. Capture is off until configured.
2. **Sync to the cloud.** A separate process uploads at a configurable
   interval, then deletes local files. Offline data buffers locally and syncs
   when connectivity returns.
3. **Store.** Tabular data (sensor readings, joint positions) lands in
   MongoDB; binary data (images, point clouds) lands in blob storage. Both
   are indexed and queryable.
4. **Use.** Query from the Viam app with SQL or MQL, or export to your own
   database for ML training datasets.
   <http://docs.viam.com/data/overview/>

Direct Atlas access: once data has synced, configure a read-only database
user for the org (via the Viam CLI) and query the Atlas Data Federation
instance from mongosh, Compass, or any MQL-compatible client.

Boundary: **Atlas logging is free with data management. Vector search is
not automatic** — embeddings must be written to Atlas separately (a stretch
goal). Same for SAM wiring: a hackathon plan, not a built feature.

## Two-camera plan

- **Arm camera (RealSense / Astra 2):** depth + RGB for perception and grasp
  targeting.
- **Webcam:** fixed overview camera for verification, demo recording, and a
  fallback RGB stream. Both are plain Viam camera components, so both can
  feed the vision service and both can be captured to Atlas.
- Name them distinctly (`arm-cam`, `overview-cam`) with explicit
  `video_path` values — auto-detect can assign two USB cameras in the wrong
  order after a reboot.

## ROS 2: verdict

**Decided (D1 in the architecture doc): Viam-native, not ROS 2.** Reasons,
specific to this event:

- The exact hardware already has Viam modules (xArm, RealSense, webcam). The
  kickoff crash course teaches Viam, the mentors support Viam, and Viam's
  data management is the event's data story.
- Viam's Python SDK + motion service + vision service cover the whole loop
  (perceive → plan → execute) with no driver work.
- ROS 2 cost in a 2-day window: install ROS 2 + drivers on the provided box,
  debug the bringup, then still bridge into Viam for data capture. A day of
  setup risk for a team with no existing ROS 2 code to reuse here.

**ROS 2 is a bridge, not a replacement.** Viam's `viam-ros2-integration`
module exposes ROS 2 topics as Viam components (config is mostly just the
topic name). Use it only if a teammate already has ROS 2 code worth reusing,
or the team needs a specific ROS tool (rviz, rosbag, MoveIt) Viam doesn't
provide. In that case run ROS 2 *alongside* Viam, not instead of it. Ari's
ROS depth still pays off on the Viam path: frames, calibration, driver
debugging, and knowing what the motion planner does underneath.
<https://www.viam.com/post/creating-a-ros2-integration-module-and-deploying-to-a-robot>

## Connect pattern (Python SDK)

Machine CONNECT tab in the Viam app gives the address, API key, and API key
ID. Env-var names below are the team's own convention, not Viam's:

```python
import asyncio, os
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.getenv("API_KEY") or "",
        api_key_id=os.getenv("API_KEY_ID") or "",
    )
    return await RobotClient.at_address(os.getenv("MACHINE_ADDRESS") or "", opts)

async def main():
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm-1")
        cam = Camera.from_robot(machine, "arm-cam")
        print(await arm.get_end_position())
```

<https://python.viam.dev/autoapi/viam/robot/client/index.html>

## Open questions for on-site check

- UR arm: exact model name in the configure search (if the team gets a UR).
- Orbbec Astra 2: exact registry module name (if the team gets an Astra 2).
- Venue network: venue WiFi details, whether the provided Linux box has both
  WiFi and Ethernet, and the arm control-box IP (sticker) for the team's arm.
- Frame calibration: arm-camera transform for the team's mounting (measure on
  site; Viam frame system holds it).
