# `--precise`: a careful aim for small objects (off by default)

`grab_text.py` without `--precise` is unchanged: the friend's pickup, standing camera
only, planned closing height, 1 cm reach-out. Nothing in `controller/` is touched
or needed. `--precise` only changes **where the arm is sent**, **how low it is asked
to close for small things**, and **what happens after a miss**, through the arm
page's existing requests. It never sends `feel_all_the_way`.

**Not run on the real arm yet.** It was worked out from the code and checked on a
simulated table (`test_aim.py`, `sim_scene.py`). Numbers below are from that
simulation. Try it one step at a time, as listed under "Trying it".

## What it changes, and why

| In the default pickup | Effect | With `--precise` |
| --- | --- | --- |
| The aim is the lowest fifth of the outline: the object's near *edge*. | About 1 cm off for short things (caps, cubes), several cm for flat things (blister packs). On top of that comes the fixed 1 cm reach-out. | Aim at the middle of where the object stands; flat things at the middle of their outline (`aim.footprint`). No reach-out unless you pass `--reach-out`. |
| The planned closing height is never under 1.5 cm. | A 2 cm object is held by its top half centimetre at best; a flat one is closed over. | Same plan (half the object's height, at most 6 cm, 1.5 cm if unknown), but small things may go lower, down to `--min-close` (default 0.8 cm), and flat things are closed on at that height. `--min-close 1.5` gives exactly the default plan. |
| Pixel to table is a flat mapping on a wide-angle picture. | Drifts away from the calibration marks. | The picture is straightened first, from the same marks, and only if that clearly predicts left-out marks better. Otherwise the numbers are the arm page's own (`aim.TableMap`). |
| A small object is a few dozen pixels and its outline is rough. | Position and height are read off a ragged edge. | A second SAM 3 pass on an enlarged cut-out around it (`sharpen`); dropped unless it clearly shows the same object. |
| One attempt. A miss says only "nothing in the gripper". | Stopping errors and aiming errors look the same. | After a miss it says which it was, from where the arm stopped compared with the object's height: *stopped in the air above it*, *on top of it* (a fingertip landed on it), *beside it*, or *closed above the table without touching anything*. Then it opens, parks, looks again and grabs again (`--tries`, default 2). |
| The contact stop ends the descent too high and the jaws close above the object. | Miss. | If the first try stopped clearly in the air (more than 1.5 cm above the object's top), the second try goes to the same planned height without the contact stop. If the first try closed at the planned height on nothing, the second goes half as high (never under 0.5 cm). A `--close` you give is never changed. |
| `--hand`: "object under the jaws" was measured at one lean of the gripper, but the arm leans the gripper out the further it reaches. The camera leans with it; the fingertips still go straight down. | The arm stops short, about 2 cm per 10 degrees of extra lean. This is very likely what the 1 cm reach-out was compensating. | `handcal` (run with `--precise`) also measures where that spot shows at each lean, in 5 degree steps; the correction uses it. |
| `dispense_server.py` closes at a fixed 5 cm, without the contact stop. | Anything under 5 cm is missed. | `dispense_server.py --precise` picks as above. |

Simulation: a 3 cm bottle cap is aimed at within 0.02 to 0.13 cm (default aim: about
1 cm off before the reach-out). With `--hand` and a gripper leaning up to 40 degrees,
the corrected aim ends within 0.1 to 0.6 cm of the object (three different camera
mounts tried), against 2.6 to 6.7 cm at leaning spots without the lean correction.

## Trying it, one change at a time

Keep the controller and calibration exactly as they are.

1. **Look, don't move.** `grab_text.py --precise --dry --text "small white cylinder"`.
   Open `pick_views/latest.jpg`: the cross should sit at the middle of where the
   object stands, not on its front edge. The printout says how it sized the object
   up, the planned closing height, and whether a lens correction is in use.
2. **Aim only.** `--precise --tries 1 --no-zoom --no-lens --min-close 1.5`. The only
   differences from the default are the aim point and no reach-out. If it now lands
   short by the same amount everywhere, add `--reach-out 1` and note that.
3. **Lower closing for small things.** Drop `--min-close 1.5`. With the saved table
   plane off by a centimetre, 0.8 cm can mean fingertips on the table: watch the
   first descent and stop if it presses.
4. **Add the rest.** `--precise` alone: second look, lens correction if the marks
   support it, retry with the miss diagnosis.
5. **Later, the gripper camera.** `--precise --hand`, after running `handcal` once
   in `--precise` mode at a spot where the gripper points straight down. An old
   `hand.json` still works and then gets the 1 cm reach-out as before.

After a miss, the line `Didn't get it: ...` is the thing to write down: it separates
stopping errors from aiming errors.

## Options (all only with `--precise`)

`--tries N` grabs per request. `--min-close CM` lowest planned closing height.
`--reach-out CM` extra reach (negative pulls in). `--no-lens`, `--no-zoom` switch
those two steps off. `--hand`, `--no-feel`, `--close CM`, `--dry` mean what they
mean without `--precise`. `dispense_server.py`: `--precise`, `--hand`, `--tries`,
`--min-close`, `--no-feel`, `--reach-out`; its `--close` is then only used if given.

## Not solved

- The arm's own contact stop is untouched. `--precise` only reacts to what it
  reports; it cannot make the first descent stop at the right place, and all its
  heights are in the saved table plane's coordinates.
- Long things lying down are aimed at their middle, but the wrist does not turn to
  them: lying along the jaws' closing direction and over 4.7 cm, they cannot be gripped.
- One camera cannot tell everything: a thing lying and pointing straight at the fixed
  camera looks like a tall thin thing standing. Above 5 cm of apparent height it is
  told apart by whether it gets wider towards its top; below that it is not.
- With `--hand`, a wrong height estimate costs aim where the gripper leans: about
  the height error times tan(lean), 0.6 cm for 1 cm at 30 degrees.
- Whether SAM 3 really outlines the same object better when enlarged needs checking
  on your objects.

## Files

`aim.py` (pixel to table, footprint; no arm or model needed), `precise_grab.py`
(the careful grab; uses `grab_text.py` for detection, pictures and the gripper
camera's detection), `sim_scene.py` + `test_aim.py` (simulated table, 37 tests).
`grab_text.py` and `dispense_server.py` only gain the `--precise` switch.
