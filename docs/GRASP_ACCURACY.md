# Current step: honour the planned grip height, standing camera only

Branch: `feel-grab-reachy`. Motion baseline: friend's commit `a8ccb74`.
Running-controller identifier: **`friend-baseline-v1`** in `/status` capabilities.
This update changes the pickup client's request, not the motor controller.

## The one behaviour change in this update

The supplied failed run contained a 1.5 cm closing height. The client also sent
`feel_all_the_way: true`, which discarded that height and searched toward
-2.5 cm relative to the saved table plane. The normal client now omits that flag
and keeps its planned closing height. For this request the target becomes
**+1.5 cm instead of -2.5 cm**, avoiding an extra 4 cm of commanded descent.

The friend's existing guarded descent can stop earlier on contact. Otherwise it
ends at the planned height and continues through the existing close-and-lift
sequence. It does not require table contact to close. This is a limit in the
calibrated model's coordinates, not a guarantee of physical fingertip clearance.

The existing height policy remains: roughly half the estimated object height,
clamped to 1.5–6 cm, or the existing **1.5 cm fallback** when height is unknown.
`--close` remains an optional manual override. The terminal labels the height
as estimated, fallback or manual. This change does not add metric depth.

See [the recorded-run analysis](REPORTED_DESCENT_ANALYSIS.md) for the evidence.

## Why the last update was rolled back

The `feel-stop-v2` update (`7cefa06`) introduced a stricter encoder/FK contact
detector and refused to close when its conditions were not met. A real run then
reported `DESCENT_NO_CONTACT` while the operator observed the fingers touching
or pressing the table. The software did not confirm that physical contact and
kept the jaws open. The synthetic tests had not established hardware behaviour.

This update restores `controller/arm/reach.py`, `controller/arm/click.py` and
`controller/arm/arm.py` byte-for-byte to `a8ccb74`, including the friend's descent,
backoff, sag compensation, jaw closing and lift. The experimental detector and
its additional refusal are removed. Its file-based report wrapper is also
removed; the friend's terminal/controller reports remain.

Wrist-camera alignment was already enabled by default in the friend's branch;
it was not introduced by `7cefa06`. For this step it is now explicitly opt-in:
normal `grab_text.py` skips wrist detection, search and alignment movements.
`--hand` enables the existing wrist path; `--no-hand` remains accepted.
The standing-camera SAM path and the friend's 1 cm `--reach-out` default remain
as they were, so this update does not simultaneously retune aiming or motion.

## Run this step

With the arm stationary, exit the old pickup program and stop the old controller.
Keep Station running. From the existing Git checkout:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
git pull --ff-only
```

Controller terminal, for this standing-camera-only test:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm/controller"
python3 -m arm.click --camera /dev/video4
```

The wrist camera can stay physically connected; this command does not open its
video device. Existing calibration/settings are reused. In Station select
follower **Self-controlled**, then press **Hold arm** on `http://localhost:8890`.
`http://localhost:8890/status` should show `controller_build: friend-baseline-v1`.

In another terminal:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
.venv/bin/python -B grab_text.py --text "small white cylinder"
```

The first line should say **standing camera only; wrist alignment OFF**.
Look for **Planned closing height:** and **Descent limit: planned closing height**.
This is a normal pickup ending at that height or an earlier contact stop. No
`--close` override is added. Preserve the existing camera/table calibration; do not press
Reset for this update. If the arm again presses the table, stop the trial and
report what happened rather than repeatedly attempting the same motion.

If the running controller already reports `friend-baseline-v1`, it can stay
running for this update; restart the pickup program so it loads the new request
logic. If `feel-stop-v2` is still running, restart the controller as shown above.

## Verification and remaining work

The full suite passes **119 tests**, including the JSON serialization boundary
from the text client's request to the controller, retention of a numeric grip
height, closing and lifting without table contact, varying estimated heights,
and disabled wrist guidance. The supplied failed detector run was also replayed
locally, reproducing its no-contact result.

Tests run without Station, SAM inference or physical motion. A separate source
comparison confirms the arm motion/controller files still match the friend's
baseline. No motor gains, contact thresholds or calibration files are changed.

The next physical test must confirm the fingers close around the object before
table pressure. Model/table error, inaccurate size estimates, XY alignment and
the outward compensation can still cause a miss. This fixes the discarded
height request; it does not establish calibrated 3D accuracy or prove a physical
grasp. Add wrist guidance only in a later step.
