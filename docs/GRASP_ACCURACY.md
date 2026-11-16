# Current pickup: tapered compensation and local wrist alignment

Branch: `feel-grab-reachy`. Running controller build: **`descent-taper-v1`**.
Both standing and wrist guidance are enabled by default in `grab_text.py`.
Use `--no-hand` only for an explicit standing-only comparison.

## What changed

- Guarded descent preserves its initial servo reference, then tapers the previous
  pose's sag compensation to zero at the closing-height endpoint. It does not
  apply fresh sag-correction pulses just before closing. Existing settling and
  blocked-joint relief remain. Joint reference speed is bounded.
- Normal pickup retains its numeric closing height and needs a complete planned
  descent. It does not require table contact to close. The separate below-table
  touch/search tool may still return a partial path, now labelled as partial.
- Wrist selection excludes matches more than a quarter image-width from the
  expected object before comparing confidence. A second look crops around the
  expected object using the same text prompt. It no longer substitutes an
  arbitrary "small white object" when the requested object is missing.
- Wrist corrections are limited to 2 cm per move, with at most five looks and
  another observation after every correction. Lost or unconfirmed targets revert
  to the standing aim. Confirmed wrist alignment skips the inherited extra 1 cm
  outward offset. That offset remains for standing-only/fallback aiming.
- Each controller grab writes a JSON report under `config/local/grasp_runs`,
  including failures during motion. Reports contain the build, request, nominal
  heights, compensated motor-target heights, measured encoder/FK heights and
  shoulder compensation. Calibration files and motor gains are not changed.

## Evidence and verification

The supplied bottle report was from the older `feel-stop-v2` controller. Its
nominal endpoint was -2.50 cm; replaying logged motor goals gave +0.36 cm, and
its final measured model height was +0.84 cm. Reconstructing nominal IK identified
a -57-step shoulder offset, consistent with carried-over hover compensation.
This was not a truncated nominal path.

The updated descent was exercised with the supplied geometry and inferred old
compensation in three ideal-servo simulations. Requested heights of 1.5, 2.7
and 2.0 cm produced motor-target heights of 1.529, 2.696 and 1.990 cm. These are
software checks, **not measured physical accuracy**. Real load, calibration and
motor response still need a physical test.

The offline suite passes **135 tests**, covering descent reference continuity,
endpoint compensation, speed bounds, contact backoff, partial-path labelling,
default camera mode, distant wrist detections, local crop coordinates, stale
frames, target loss, correction limits, close/lift without contact, and report
persistence on motion errors. SAM outputs and hardware feedback are mocked;
no local robot or GPU inference was used.

## Linux update and start

With the arm stationary, stop the old pickup program and controller. Keep
Station running. From the existing checkout:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
git switch feel-grab-reachy
git pull --ff-only
git log -1 --oneline
```

If Git reports a conflict, preserve local changes; do not reset the checkout.
Controller terminal:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm/controller"
python3 -m arm.click --camera /dev/video4 --hand-camera /dev/video6
```

Select follower **Self-controlled** in Station and **Hold arm** on the arm page
at `http://localhost:8890`. Do not press the camera-calibration Reset button.
Pickup terminal:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
.venv/bin/python -u -B grab_text.py --hand 2>&1 | tee pickup_test.log
```

Type an object at `What now?`. It should print `Controller build: descent-taper-v1`
and `standing camera + wrist alignment ON`. A confirmed wrist correction prints
`wrist alignment confirmed; no extra outward offset`. If wrist detection is
unavailable or uncertain, the terminal explicitly reports standing-camera fallback.
Pulling does not update running Python processes; restart both programs.

## A focused physical test within an hour

Use one isolated known object in a small central pickup patch. The supplied
plane disagreed with one touch near the bottle by 1.6 cm; software cannot turn
that into a correct physical measurement. Recheck the table before relying on
the new height endpoint. Keep both camera mounts fixed.

With controller/pickup stopped, Station running, and the follower controlled by
the leader in mirroring mode, back up and re-teach the local Z plane:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm/controller"
cp calib/table.json "calib/table.backup-$(date +%Y%m%d-%H%M%S).json"
python3 -m arm.reach table
```

Follow the four prompts: closed fingertips pointing down, barely touching the
bare table at four corners around the intended pickup area. Do not press down.
Then check an additional point in the middle, not one of the fitted corners:

```bash
python3 -m arm.reach where
```

With a light touch there, the displayed height should be near zero. Aim for
agreement within roughly 0.3 cm for the 2 cm test cylinder; this is a practical
test target, not a software guarantee. If readings differ by a centimetre,
investigate the touch/pose/calibration before another automatic pick. Exit
`where` with Ctrl+C, then restart the controller, select Self-controlled and
Hold arm. This teaches the Z plane; it does not recalibrate camera XY mapping.

For the known 2 cm cylinder only, a controlled first trial can use:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
.venv/bin/python -u -B grab_text.py --hand --object-height 2 --text "small white cylinder" 2>&1 | tee pickup_test.log
```

Here `--object-height 2` supplies the cylinder's **measured 2 cm height**. Its
closing height becomes 1 cm above the saved table plane. This isolates motion
from the approximate RGB height estimate. For a short cap, measure its height
in the intended picking orientation and supply that value in centimetres. With
a measured height, the closing-height floor is 0.5 cm. Measured objects no taller
than 0.5 cm are refused before motion; they need a validated lower fingertip
clearance, not a guess from this script.

Omit `--object-height` for automatic height selection. Its existing policy is
half the RGB-estimated height, clamped to 1.5–6 cm; unknown height uses 1.5 cm.
That automatic policy is not suitable for an unmeasured very thin cap. Neither
policy supplies measured depth. `--close` remains a manual fingertip-height
override, not a jaw-width command.

Seek several consecutive successes in the same verified patch before expanding
positions or changing objects. Always-correct pickup of arbitrary objects is
not established: wrist alignment happens above the object, not continuously
during descent, and this implementation does not derive calibrated depth from
the two RGB cameras. Broader reliability needs measured 3D geometry, identity
tracking and physical validation across object sizes and poses.

## Send the result if it misses

Keep the controller running and, before parking or another attempt, run:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
.venv/bin/python -B collect_arm_debug.py
```

This performs read-only collection and creates a ZIP in Downloads. Transfer it
to your computer and attach it in the chat. It does not upload automatically.
