# Current step: restore the friend's pickup, standing camera only

Branch: `feel-grab-reachy`. Motion baseline: friend's commit `a8ccb74`.
Running-controller identifier: **`friend-baseline-v1`** in `/status` capabilities.

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
This is a normal pickup using the friend's contact-based descent. No `--close`
override is added. Preserve the existing camera/table calibration; do not press
Reset for this update. If the arm again presses the table, stop the trial and
report what happened rather than repeatedly attempting the same motion.

## Verification and remaining work

Local tests cover the standing-only default, explicit wrist opt-in, unchanged
guarded-grab request fields, and exclusion of wrist calls from a normal pickup.
The existing suite is run without Station, SAM inference or physical motion.
A separate source comparison confirms the arm motion/controller files match
the friend's baseline. Physical success still needs the next robot run.

Restoring the baseline does not establish accurate automatic height or guarantee
that the gripper stops before pressing a surface. Its contact heuristic responds
to joint tracking error, not a force sensor. First confirm the previous pickup
behaviour is restored. Then improve the descent/closing point as one separate
change using measured observations; add wrist guidance only in a later step.
