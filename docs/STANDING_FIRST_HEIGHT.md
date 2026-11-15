# Standing-camera pickup and height troubleshooting

The fixed standing camera selects the object and estimates its table footprint.
The wrist camera may point elsewhere before approach. A missing or ambiguous
wrist detection no longer prevents pickup. SAM 3 still segments both snapshots
and saves both previews. Both camera feeds must be connected, fresh and advancing;
changing a camera identity or losing a feed remains an error. A changed standing
scene blocks submission; a changed wrist background is recorded as supplementary
information. A wrist detection cannot rescue a missing/ambiguous standing target.

The controller's pickup is still one motion sequence. It does **not** run SAM
between approach and closing, scan automatically, or correct its trajectory from
the wrist image. Do not interpret `HOLDING_REPORTED` as visual lift verification.
For Reachy medicine requests, exact-label OCR is required on the selected standing
object before pickup and after placement. Wrist OCR, when possible, is additional
information; it cannot replace the standing check.

## What changed in the closing height

Previously, side pickup multiplied measured object height by the selected **image
row fraction**. Perspective and the silhouette ranking could change that fraction
and therefore the commanded height even when the physical object was unchanged.
That conversion has been removed from both `text_pick.py` and `pick_v5.py`.

The closing height now follows this order:

1. Explicit `--height` in centimetres, when supplied.
2. The item's measured `grasp_height_cm`.
3. Measured `height_cm * grasp_height_frac` (fraction defaults to 0.5).

For a measured 10 cm upright bottle with `grasp_height_frac: 0.4`, closing height
is 4 cm regardless of which image row SAM's silhouette suggests. The image jaw
line is an advisory 2D contact suggestion, not a calibrated projection of the
physical contact. Closing at/above the measured top is blocked, as is an approach
that does not clear the top. Do not compensate for a bad table plane by changing
the bottle's real dimensions.

`z` in the pickup request is **distance above the table along its normal**.
It is not absolute robot-base Z and is not the wrist-camera height. The controlled
reference is the point between the closed fingertips, defined by `TIP_OFFSET`
in `controller/arm/kinematics.py`. The table is `z = a*x + b*y + c` in metres.
The report's `height_geometry` gives the table contact and requested fingertip
XYZ in base-frame centimetres. `approach.height_source` records the chosen input.

## Steps on your friend's Linux laptop

From the existing clone, update the code and restart its controller/service:

```bash
git pull --ff-only
.venv/bin/python -B calibration_report.py --arm-url http://localhost:8890
```

This command only reads `/status`. Preserve local `controller/calib/`, poses and
`config/local/` measurements. The unchanged setup retains its existing errors.
The previously inspected upstream files had 4.34 cm maximum camera leave-one-out
error and 16.03 mm maximum table-touch residual. Those are stored-data diagnostics,
not measured current pickup errors. Run the command above against the actual arm.

If the table height is suspect, check it before redoing the camera board alignment:

1. With the arm stationary, physically measure the height of the point between
   the fingertips above the table. Compare several reachable XY positions and
   several heights, including the intended side-grasp orientation. Keep a record
   of repeated measurements; one good position does not validate the whole area.
2. Read the encoder/FK height using the example below. Replace `4.0` with the
   actual ruler measurement. This does not command the arm or alter calibration.

```bash
.venv/bin/python -B calibration_report.py --arm-url http://localhost:8890 \
  --measured-tip-height-cm 4.0
```

Inspect `fingertip_comparison.encoder_minus_measured_cm`. It compares encoder FK
with your measurement; neither SAM nor the wrist camera supplies that height.
The transform is sampled at frame reception, so hold the arm still. For a tilted
table measure perpendicular to the surface. A constant discrepancy suggests an
offset; a discrepancy varying with XY or orientation suggests a plane, joint/FK
or tool-geometry issue. These are diagnostic clues, not an automatic correction.
Reteach the table with `python -m arm.reach table` from `controller/` if its
measurements are inconsistent; follow its physical leader/mirroring procedure.

Then calibrate the fixed camera over the intended pickup area. With the controller
running without `--read-only`, use the helper from `controller/`:

```bash
../.venv/bin/python vision/calibrate_board.py \
  --page http://localhost:8890 --square-cm 2.15
```

Use your measured square size, not necessarily 2.15. The helper backs up the old
camera file and reloads the controller; a blanket Reset is not needed for this
file replacement. One board pose is a start, not independent validation. Use
additional positions covering the pickup area and unused check points as described
in [workspace and validation](WORKSPACE_AND_VALIDATION.md). Targets must lie inside
the fitting and independent-check regions. The camera's 1 cm error gate is only
a coarse ceiling: a 4 cm bottle in a 4.7 cm opening has only 3.5 mm nominal clearance
on each side when centered. A passing calibration score does not prove it will fit.

Create or edit `config/local/items.json` with the exact requested object name.
These example numbers are **not measurements of your bottle**:

```json
{
  "small bottle": {
    "height_cm": 10.0,
    "width_cm": 4.0,
    "grasp": "side",
    "grasp_height_cm": 4.0,
    "measurements_verified": false
  }
}
```

Replace the dimensions/closing height with your measurements, then set
`measurements_verified` to `true` after checking the intended physical grasp.
Alternatively omit `grasp_height_cm` and use `grasp_height_frac: 0.4`.
Both the measured width and the silhouette estimate must fit the usable opening.
Reachy delivery also needs `fetch_allowed: true` and a measured destination;
see [Reachy integration](REACHY_INTEGRATION.md).

Review a plan using the calibrated search area and independent measurements:

```bash
.venv/bin/python -B text_pick.py --arm-url http://localhost:8890 \
  --text "small bottle" --items config/local/items.json \
  --workspace config/local/workspace.json --validation config/local/validation.json
```

Review `execution_blocks`, `approach`, `height_geometry` and the standing preview.
The wrist may look away from the bottle. The follower must be self-controlled
and held for an intended physical trial; use a clear, reachable park pose rather
than forcing it to see the target first. Add `--go` only for the actual trial.
No robot has been operated or pickup accuracy measured during this code change.

To inspect a proposed target in base coordinates without motion (substitute the
plan's actual X, Y and height), use:

```bash
.venv/bin/python -B calibration_report.py --arm-url http://localhost:8890 \
  --target-cm 0 24 --height-cm 4
```

## Using an up/down wrist scan later

A scan is a useful next stage after the stand camera has located the object.
Metric height from multiple views needs measured camera intrinsics/distortion,
camera-to-gripper mounting geometry, trustworthy measured arm poses associated
with the frames, and corresponding points on the same object. SAM masks alone
do not establish those point correspondences. The repo's `two_camera_setup.json`
still has empty calibration fields, and live frames have receive times rather
than exposure times. It would be misleading to infer centimetres from those data.

The intended sequence is: stand selection → clear approach pose → stop and capture
several wrist views at bounded offsets → estimate height with uncertainty →
re-plan/verify → close and verify lift. Stopping for each capture can reduce motion
timing error, but requires flushing buffered frames and checking measured pose
stability. It does not eliminate calibration requirements. Wrist hand-eye
calibration itself needs varied poses/rotations, not only a vertical sweep.
See [OpenCV's camera and hand-eye calibration methods](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

Automatic scan motion and wrist-driven corrections are not enabled by this update.
The existing read-only `camera_capture.py` can save both images and encoder poses
at manually held positions for calibration work. No Ollama service is needed.
