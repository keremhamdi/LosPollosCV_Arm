# LosPollosCV_Arm

Text-prompted object detection, segmentation and grasp proposals for an ElRobot
with **two continuously connected cameras: one fixed standing camera and one
mounted on the gripper**. Combines [Robot_Vision](https://github.com/keremhamdi/Robot_Vision)
with the integrated [lospollos-arm](https://github.com/ciloglu44/lospollos-arm) controller.

## What it does

1. Read a text description, such as `pick up the white cup`.
2. Fetch the latest standing and hand images from the NormaCore Station bridge.
3. Run **SAM 3 (`facebook/sam3`)** on each image to detect matching objects and
   produce their segmentation masks. The default uses SAM confidence for object
   selection and mask shape for posture. Missing or ambiguous objects do not trigger a pick.
4. Highlight the object region and report visible area in pixels, a proposed
   grasp point, jaw contacts, and a top or side approach.
5. With measured calibration and object dimensions, propose robot coordinates.
   `--go` explicitly requests one supported pickup after the checks pass.

The controller keeps acquiring both camera feeds. SAM runs on snapshots
for each text request; this is **not continuous visual servo control**. The fixed
camera supplies table localization. The moving hand camera currently provides
an independent semantic check and close-up proposal, not calibrated depth.

**Current limitations:** no hardware pickup has been validated here. Top-grasp
execution is blocked because wrist-roll control is not implemented. Side pickup
uses approximate table geometry and requires physical calibration, dimensions,
clearance and reachability checks. Two cameras alone do not supply 3D geometry.

## Linux setup

The full sequence, including the robot's table setup, is in
[LINUX_HANDOFF.md](LINUX_HANDOFF.md). Your friend needs the robot's working
NormaCore checkout, generated protobuf modules, Station and camera configuration;
those components and SAM model weights are not bundled in this repository.

```bash
git clone https://github.com/keremhamdi/LosPollosCV_Arm.git
cd LosPollosCV_Arm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest -q test_vision test_integration test_voice_pick test_text_pick test_calibration_report
```

These dependency versions match the development environment. Install a compatible
PyTorch CPU/CUDA build for your Linux hardware if necessary. See
[PyTorch installation](https://pytorch.org/get-started/locally/) for wheel selection.

For SAM 3, request access to [facebook/sam3](https://huggingface.co/facebook/sam3),
then authenticate with `hf auth login` on the Linux laptop. The first inference
downloads the weights. **Ollama, Qwen, an API key and a separate model server are
not required.** SAM 3 runs directly in Python through PyTorch and Transformers.
`--vlm none` is the default for text, voice, selection and two-view entry points.

Optional checker adapters remain available only when explicitly selected;
see [optional checkers](docs/OPTIONAL_CHECKERS.md). They are outside the default
installation. Microphone/speech packages are also optional.

SAM confidence is not a calibrated probability or an independent identity check.
Without a second checker, exact labels, similar objects and posture inferred from
an angled silhouette need particular evaluation on your own camera images.

## Start both camera feeds

First preserve or measure the **actual robot's table calibration** as described
in the setup guide. The bridge needs `controller/calib/table.json` to start.
With Station already running, in a separate terminal:

```bash
cd LosPollosCV_Arm/controller
export NORMA_CORE=/absolute/path/to/norma-core
../.venv/bin/python -m arm.click --stand-index 0 --hand-index 1 --read-only
```

Confirm the camera indices in the Station feed; select the fixed stand camera
on `http://localhost:8890`. If using direct Linux camera devices instead of the
Station feed, see the guide's `--camera` / `--hand-camera` alternative.

## Type an object request

From the repository root:

```bash
.venv/bin/python -B text_pick.py \
  --arm-url http://localhost:8890 \
  --text "pick up the white cup" --category cup --vlm none
```

Omit `--text` to type at a prompt. Results go into a new directory under
`review_outputs/text_pick/`: annotated images, exact masks, input frames and
`result.json`. No arm movement occurs without `--go`, and the controller's
`--read-only` mode rejects all motion requests. See the guide before a physical trial.

Saved images work without the robot:

```bash
.venv/bin/python -B text_pick.py \
  --image /path/to/stand.jpg --hand-image /path/to/hand.jpg \
  --text "pick up the white cup" --category cup --vlm none
```

## Calibration and accuracy

Read [CALIBRATION.md](CALIBRATION.md) for table-plane, camera-to-table and moving
hand-camera calibration. The known files currently have centimetre-scale errors;
there is no validated millimetre accuracy or physical grasp-success claim.

Audit your own camera and table files without moving the robot:

```bash
.venv/bin/python -B calibration_report.py \
  --camera controller/calib/camera.json --table controller/calib/table.json
```

The checkerboard calibration helper is in `controller/vision/calibrate_board.py`.
Real calibration is specific to the robot, mount, table and camera settings and
is intentionally not committed. Synthetic fixtures are labelled tests only.

## Project map

| File | Purpose |
| --- | --- |
| `text_pick.py` | Combined text, two-camera, planning and optional pickup workflow |
| `target_selector_v5.py` | SAM 3 masks and optional identity verification |
| `grasp_planner_v3.py`, `pick_v5.py` | Silhouette grasp candidates and approximate table mapping |
| `arm_bridge.py` | Versioned HTTP contract and one-attempt controller submission |
| `controller/arm/` | NormaCore connection, kinematics, controller and camera feed |
| `calibration_report.py` | Read-only diagnostics for measured calibration files |
| `two_camera_setup.json` | Incomplete template for measured two-camera geometry |
| `calibrated_pair.py`, `multiview_geometry.py` | Offline calibrated triangulation utilities |

Provenance is in [NOTICE.md](NOTICE.md) and `controller/UPSTREAM.json`.
Development details are in `INTEGRATION.md` and `docs/DEVELOPMENT_NOTES.md`.
See [next improvements](docs/NEXT_STEPS.md) for the SAM-only photo findings and
the recommended order of calibration, workspace filtering and grasp-control work.
