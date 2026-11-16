# Current arm work: premature closing on small objects

Branch: `feel-grab-reachy`. Baseline: friend's commit `a8ccb74`.
Controller update: **`feel-stop-v2`** (reported in `/status` capabilities).

The reported failure is that the arm stops too high and closes above the object.
Code inspection found a reproducible way this can happen: the old guarded
descent classified increasing commanded-versus-measured height error as contact
after only two readings. It did not check whether the fingertips were still
descending. Changes in servo lag could therefore trigger closing in free space.
This is a demonstrated software failure mode, not yet the confirmed cause of
every miss on the physical robot.

## What changed

- Contact inference needs a sustained nearly stationary encoder/FK height while
  the reference keeps descending. It also needs a preceding free downward
  movement. Increasing lag while still descending is not enough.
- Every kinematic joint must have fresh, calibrated, torque-enabled feedback
  during descent. Updates from other motors cannot disguise a frozen reading.
- The guarded path starts from measured angles and preserves the existing
  initial servo reference. Its joint reference speed is bounded.
- On contact inference, the controller first stops advancing its reference,
  then backs off smoothly from the measured position by the existing 0.3 cm.
  This replaces jumping to a nominal earlier waypoint followed by sag correction.
- A search with no explicit closing height cannot close without contact evidence.
  A stall before verified free descent reports `DESCENT_BLOCKED`; reaching the
  search floor without evidence reports `DESCENT_NO_CONTACT`. Both leave jaws open.
- Each pickup saves a JSON report in `config/local/grasp_runs/`, including failure
  reports. The terminal prints the path; `/status` exposes the latest summary.

The contact detector uses a 0.6 cm excess-lag threshold, a minimum 0.24-second
observation window (longer for slow motion), no more than 0.10 cm height spread
within that window, at least 0.30 cm command advance and two confirming samples.
At least 0.25 cm of observed free descent must precede a contact inference.
These are engineering starting values tested on synthetic feedback, not tuned
or physically validated thresholds for this arm.

## Try the normal pickup

In your existing working checkout of this branch:

```bash
git pull --ff-only
```

Stop the old **arm controller** with Ctrl+C and restart it with your usual launch
command. Keep Station running. Restart `grab_text.py` too; pulling files alone
does not update running Python processes. Use follower **Self-controlled** and
**Hold arm**, as before.

From the repository root, using the same environment and calibration:

```bash
.venv/bin/python -B grab_text.py --text "small white cylinder"
```

This attempts a normal pickup. No `--close` or `--descent-only` flag is required.
`--close` explicitly supplies a table-relative closing height; `--no-feel`
disables guarded descent, so that mode does not exercise this fix.

The controller should identify itself as `feel-stop-v2` in
`http://localhost:8890/status`. The client prints `Descent stop:` and the report
path. If it misses, send that report and say whether it closed above, beside or
around the object. The report contains no camera images or credentials.

## What is verified locally

The full suite passes **122 tests**, with no Station connection or arm motion.
The new regressions exercise:

- A rising-lag trace that triggers the old detector while the simulated tip is
  still descending; the new detector does not call it contact.
- A sustained surface stop after free motion, and a brief pause that must not
  count as contact.
- The actual guarded-descent loop using synthetic encoder feedback, including
  surface backoff, a blocked start, and one stale joint while the others update.
- Keeping jaws open after an unsuccessful unknown-height search, and preserving
  a structured failure report through the HTTP controller route.

These tests do not run SAM inference, establish calibration accuracy or prove
physical pickup success. The simulated surface and encoder model are synthetic.

## Where we are and what remains

This update addresses premature stopping. It does not add true depth sensing,
change SAM segmentation, retune the gripper squeeze, or replace the friend's
calibration. Encoders alone cannot distinguish table contact from another
obstruction, servo weakness or mechanical binding. A physically inaccurate
kinematic model can also report an incorrect fingertip height.

Next, use the arm result to separate stopping errors from XY alignment errors.
The current wrist-camera correction remains an approximate hover check, and the
client still has the friend's default 1 cm `--reach-out` compensation. Its final
wrist correction is not always re-observed before pickup. Those behaviours need
separate measured validation; this change does not silently retune them.

Reliable raised-surface pickup and hand/plate placement still require measured
camera geometry and support height. Moving-hand delivery additionally requires
tracking and a release check. SAM masks alone do not provide metric depth.
