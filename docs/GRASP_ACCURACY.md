# Current step: planned grip height with standing and wrist guidance

Branch: `feel-grab-reachy`. Motion baseline: friend's commit `a8ccb74`.
Running-controller identifier: **`friend-baseline-v1`** in `/status` capabilities.
The height fix changes the pickup client's request, not the motor controller.
It applies with either camera mode. Use `--hand` for the requested two-camera
flow: standing-camera selection/localization, then optional wrist alignment,
then descent to the planned closing height and close/lift.

## The one behaviour change in this update

The supplied failed run contained a 1.5 cm closing height. The client also sent
`feel_all_the_way: true`, which discarded that height and searched toward
-2.5 cm relative to the saved table plane. The normal client now omits that flag
and keeps its planned closing height. For this request the target becomes
**+1.5 cm instead of -2.5 cm**, avoiding an extra 4 cm of commanded descent.

The friend's existing guarded descent can stop earlier on contact. Otherwise it
finishes the reachable planned path and continues through the existing close-and-lift
sequence. A partial path may end above the requested height. It does not require
table contact to close. The requested height is in the calibrated model's
coordinates, not a guarantee of physical fingertip clearance.

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

## Run the two-camera pickup

With the arm stationary, exit the old pickup program and stop the old controller.
Keep Station running. From the existing Git checkout:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
git switch feel-grab-reachy
git pull --ff-only
git log -1 --oneline
```

If Git reports local changes or a conflict, keep them; do not reset the checkout.
Controller terminal:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm/controller"
python3 -m arm.click --camera /dev/video4 --hand-camera /dev/video6
```

These device paths are the ones used on the friend's laptop. Both cameras are
opened; wrist alignment also needs the existing `controller/calib/hand.json`.
Existing calibration/settings are reused. In Station select
follower **Self-controlled**, then press **Hold arm** on `http://localhost:8890`.
`http://localhost:8890/status` should show `controller_build: friend-baseline-v1`.

In another terminal:

```bash
cd "$HOME/Projects/Robotics Hackathon/LosPollosCV_Arm"
.venv/bin/python -B grab_text.py --hand --text "small bottle"
```

Omit `--text "small bottle"` to keep the interactive `What now?` prompt.
The first line should say **standing camera + wrist alignment ON**.
Look for **Planned closing height:** and **Descent limit: planned closing height**.
This is a normal pickup targeting that height, with an earlier contact stop possible. No
`--close` override is added. Preserve the existing camera/table calibration; do not press
Reset for this update. If the arm again presses the table, stop the trial and
report what happened rather than repeatedly attempting the same motion.

Pulling files does not update a running Python process. Restart both programs
when switching from `feel-stop-v2`, including the pickup program that loaded
SAM. Keep Station running. For standing-only comparison, omit the controller's
`--hand-camera` argument and the pickup client's `--hand` flag.

## The bottle run that stopped above the table

The supplied bottle log says `height estimate does not set the closing point`
and ends with `DESCENT_NO_CONTACT`. Those messages belong to the superseded
contact-required version, not the corrected client/controller pair. The image
caption estimates a 4 cm bottle and a 2 cm closing height; the old request
discarded that closing height. The corrected request retains it in both modes.
That estimate is approximate, not a measured physical bottle height.

`floor_reached_without_contact` in the old code only meant the planned movement
ended without accepted contact. Its planner permits partial paths, so it could
report that even when the requested floor was unreachable. It does not prove
the fingers reached the table. The bottle run's report
`config/local/grasp_runs/20261116T015937.767262Z_a627a04c.json` is needed to compare
commanded and encoder/FK heights; it has not been supplied locally. Table/model
error and incomplete reach remain possible explanations, not established causes.

The 710 pixel wrist discrepancy was converted to 41.2 cm by the saved local
camera calibration and rejected by the existing 8 cm limit. It is not a depth
measurement. The arm kept the standing-camera aim plus its inherited 1 cm
outward compensation. To inspect why wrist guidance disagreed, use the image
`pick_views/hand_latest_0.jpg` from that run and the saved wrist calibration.
Do not increase the correction limit to accept this discrepancy.

## Verification and remaining work

The full suite passes **120 tests**, including the JSON serialization boundary
from the text client's request to the controller, retention of a numeric grip
height, closing and lifting without table contact, varying estimated heights,
and disabled wrist guidance. A new bottle regression checks the two-camera
path with both a rejected 710 pixel correction and an accepted correction:
the estimated 4 cm object retains a 2 cm numeric closing height through JSON
routing, and the real controller's grab sequence closes/lifts with no contact
reported by the mocked descent. The supplied earlier cylinder detector run
was also replayed locally, reproducing its no-contact result.

Tests run without Station, SAM inference or physical motion. A separate source
comparison confirms the arm motion/controller files still match the friend's
baseline. No motor gains, contact thresholds or calibration files are changed.

The next physical test must confirm the fingers close around the object before
table pressure. Model/table error, inaccurate size estimates, XY alignment and
the outward compensation can still cause a miss. This fixes the discarded
height request; it does not establish calibrated 3D accuracy or prove a physical
grasp. Wrist guidance still uses approximate local calibration and discrete
looks above the object; it does not continuously verify alignment during descent.
