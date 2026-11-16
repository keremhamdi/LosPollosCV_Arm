# Getting the jaws around small things

What `grab_text.py` (and `dispense_server.py`, which now picks the same way) does to
land the jaws on an object, what was going wrong before, and what to do on the robot.

**None of this has been run on the real arm yet.** It was worked out from the code
and checked on a simulated table (`test_aim.py`, `sim_scene.py`). The numbers below
are from that simulation. Do the three steps under "On the robot" before trusting it.

## What was going wrong

| Problem | Effect | Now |
| --- | --- | --- |
| The gripper camera's "object is under the jaws" spot was measured at one lean of the gripper, but the arm leans the gripper outwards the further it reaches. The camera leans with it; the fingertips still go straight down. | The arm stopped short, about 2 cm for every 10 degrees of extra lean. (This is what `--reach-out 1` was papering over.) | `handcal` measures where the spot moves at each lean; the correction uses it. Looks are done with the setup's own lean wherever the arm can. |
| The aim was the lowest fifth of the outline: the object's near *edge*, not where it stands. | About 1 cm off for short things (caps, cubes), several cm for flat things (blister packs). | Aim at the middle of the footprint; flat things at the middle of their outline (`aim.footprint`). |
| A small object is a few dozen pixels in the full picture and its outline is rough. | Position and height read from a ragged edge. | A second SAM 3 pass on an enlarged cut-out around it (`sharpen`). |
| Pixel to table used a flat mapping on a wide-angle picture. | Drifts away from the calibration marks. | The picture is straightened first, using the same marks, but only if that clearly predicts left-out marks better (`aim.TableMap`). |
| Feeling for the table always started at 7 cm. | The fast part of the way down could run the palm into anything taller. | Feeling starts 2 cm above the object's estimated top. |
| One attempt, whatever happened. | A miss was the end. | If the jaws close on nothing: open, park, look again, grab again (`--tries`, default 2). It says what the miss looked like. |
| `dispense_server.py` closed the jaws at a fixed 5 cm with no gripper camera and no feeling. | Anything shorter than 5 cm was missed. | It uses the `grab_text.py` path. |

In the simulation, with a gripper that leans up to 40 degrees: the corrected aim ends
within 0.1 to 0.6 cm of the object, against 2.6 to 6.7 cm at leaning spots with an old
gripper camera setup. A 3 cm bottle cap: aim 0.02 to 0.13 cm off, against about 1 cm.

## On the robot

1. **Restart the arm page** (`python3 -m arm.click ...`): `controller/arm/reach.py` changed.
2. **Run `handcal` once more** in `grab_text.py`, with a small object, at a spot where
   the gripper points straight down (fairly near the base). It now also leans the
   gripper in 5 degree steps over the object and notes where it shows; that adds
   under a minute. It prints how far the spot moved and what geometry expected.
   Until you do, the old setup keeps working exactly as before (including the 1 cm
   reach-out) and a one-line reminder is printed.
3. **Check without moving**: `grab_text.py --dry --text "small bottle"`, then look at
   `pick_views/latest.jpg`. The cross should sit at the middle of where the object
   stands, not on its front edge. The first line says whether a lens correction is
   in use and how well left-out calibration marks land.

Then grab a few small things at different distances from the base. If the arm is
still consistently short or long by the same amount everywhere, set `--reach-out`
(cm, negative pulls it in). If it is off only far from the base, redo `handcal` and
read its "lean" lines.

## Reading what it prints

- `fixed camera: 162 calibration marks, lens correction on: left-out marks land within 0.1 cm (flat mapping: 0.6 cm)`
  how good the pixel to table mapping is. "the marks are too loose to tell" means the
  marks disagree with each other by more than 0.5 cm; a checkerboard calibration fixes that.
- `about 2.1 cm tall, 3.0 cm wide` / `lying flat, 6.2 cm wide`: how it sized the object up.
- `gripper camera: leaning 20 deg here (0 in its setup), allowed for`: the lean correction at work.
- `gripper camera: off by 1.4 cm`, then smaller: it is centring. "getting worse" or
  "couldn't find it": it fell back to the fixed camera's aim.
- `Didn't get it: it stopped 2.1 cm up, on top of the object: a fingertip came down on it (aim a little off)`:
  where the arm stopped compared with the object's height tells what kind of miss it was.

## Options

`--tries N` grabs per request (1 = the old behaviour). `--reach-out CM` extra reach.
`--no-lens` map pixels exactly as the arm page does. `--no-zoom` skip the second look.
`--no-hand` skip the gripper camera. `--no-feel`, `--close CM` as before.
`dispense_server.py` takes `--tries`, `--no-hand`, `--no-feel`, `--reach-out`, `--close`.

## What is still not solved

- **Long things lying down** (pens, blister packs) are found and aimed at their
  middle, but the wrist does not turn to them. If their long side lies along the
  jaws' closing direction and is over 4.7 cm, the jaws cannot close around it.
- **One camera cannot tell everything.** A thing lying and pointing straight at the
  fixed camera looks like a tall thin thing standing. Above 5 cm of apparent height
  it is told apart by whether it gets wider towards its top; below that, and for very
  thin things, the gripper camera has to sort it out.
- **A wrong height estimate costs aim when the gripper leans**: about the height
  error times tan(lean), so 0.6 cm for 1 cm at 30 degrees. Nothing is lost where the
  gripper points straight down.
- **Beyond the leans `handcal` could measure**, the correction falls back to geometry
  that assumes the camera sits about 6 cm behind the fingertips (`CAMERA_BACK_CM`).
- The second look trusts SAM 3 to outline the same object better when it is enlarged.
  It is dropped unless the two outlines overlap well, but whether it helps on your
  objects needs checking on the robot.
