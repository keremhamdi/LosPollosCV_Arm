# Branch `small-object-grab`: a careful aim for small objects

On this branch `grab_text.py` and `dispense_server.py` pick the careful way by
default. `--basic` gives the pickup exactly as it is on `feel-grab-reachy`.
Nothing in `controller/` is changed: the arm's motion, contact stop and
calibration files are the friend's. Only **where the arm is sent**, **how low it
closes on small things** and **what happens after a miss** differ.

**Not run on the real arm yet.** It was worked out from the code and your run
notes, and checked on a simulated table (`test_aim.py`, `sim_scene.py`).

## On the robot laptop

```bash
git fetch origin
```

```bash
git switch small-object-grab
```

Controller as before (it can keep running if it already reports `friend-baseline-v1`):

```bash
cd controller && python3 -m arm.click --camera /dev/video4 --hand-camera /dev/video6
```

1. **Look, don't move.** From the repository root:

   ```bash
   .venv/bin/python -B grab_text.py --dry --text "small bottle"
   ```

   Open `pick_views/latest.jpg`: the cross should sit at the middle of where the
   object stands, not on its front edge. The printout gives its estimated size and
   the planned closing height.
2. **Grab, standing camera only.** Same command without `--dry`. Watch the first
   descent: small things are closed on as low as 0.8 cm above the saved table
   plane. If the fingers press the table, stop and use `--min-close 1.5`.
3. **Measure the aim once** (one minute, replaces the guessed 1 cm reach-out).
   Start `grab_text.py` without `--text`, then type `aimcheck small bottle`. The
   open jaws stop just above the object; type `out 1`, `in 0.5`, `left 0.5`,
   `right 0.5` until they are centred over it, then Enter. Repeat at one or two
   other distances from the base. `aimcheck clear` forgets it.
4. **Both cameras.** Add `--hand`. The existing `hand.json` works as it is.
   Running `handcal` once more also measures how the gripper's lean moves the
   picture, which matters far from the base.

If anything behaves worse than before, `--basic` is the previous pickup.

## What it does differently

| Basic pickup | Effect | Careful pickup |
| --- | --- | --- |
| Aims at the lowest fifth of the outline: the object's near *edge*, then adds 1 cm of reach. | About 1 cm off for short things (caps, cubes), several cm for flat things (blister packs). | Aims at the middle of where the object stands; flat things at the middle of their outline (`aim.footprint`). |
| Fixed 1 cm reach-out, a guess. | Right or wrong depending on where the object is. | Uses what `aimcheck` measured. Until then: no reach-out, and if a fingertip comes down on top of the object, the next try uses 1 cm (and keeps it for the session if that works). `--reach-out` overrides. |
| Planned closing height never under 1.5 cm. | A 2 cm object is held by its top half centimetre at best; a flat one is closed over. | Same plan, but small things may go down to `--min-close` (0.8 cm) and flat things are closed on there. Never `feel_all_the_way`. |
| The contact stop ends the descent by backing off and pulling the arm up to the exact planned height. | Seen on the arm: it comes down right, then rises a little just before closing and misses a small object. | Things under 6 cm (or of unknown height) go down without the contact stop: the arm's own small-object descent, no pull-up, ending a little low. `--feel-small` keeps the contact stop. |
| Flat pixel-to-table mapping on a wide-angle picture. | Drifts away from the calibration marks. | The picture is straightened first, from the same marks, only if that clearly predicts left-out marks better. |
| A small object is a few dozen pixels with a rough outline. | Position and height read off a ragged edge. | A second SAM 3 pass on an enlarged cut-out; dropped unless it clearly shows the same object. |
| One attempt; a miss says only "nothing in the gripper". | Stopping and aiming errors look the same. | Says which it was (stopped in the air, on top of it, beside it, closed too high), then opens, parks, looks again and grabs again (`--tries`, default 2). |
| The contact stop ends the descent in the air. | Jaws close above the object. | The second try goes to the same height without the contact stop. If it closed on nothing without touching anything, the second try goes half as high (not under 0.5 cm). |
| `--hand`: takes whatever matches the name, even 710 px from where the object must be, then throws the whole correction away. | No wrist correction on that grab. | Only a match within 8 cm of where the object must be counts; otherwise it looks for a small object there. |
| `--hand` is opt-in, and stops after a move it assumes was enough. | The last correction is never checked. | The gripper camera is used whenever it is set up (`hand.json` exists; `--no-hand` turns it off). It keeps moving and looking until a look itself shows the object within 0.15 cm of under the jaws (up to 6 looks). A blob of quite another size than the fixed camera measured (a glint, a plate) is not taken for the object, and the fallback looks for a small object of any colour. |
| `--hand`: "object under the jaws" was measured at one lean of the gripper. | Arm stops short where the gripper leans, about 2 cm per 10 degrees. | After a new `handcal`, the correction allows for the lean. |
| A flat cap or a tablet is planned at 0.8 cm above the *saved* table plane. | Seen on the arm: it stops just short of the table and closes over the object. | Things lying flat or estimated under 1.2 cm: go down until the fingertips touch the table itself (the arm's contact stop), close there. `--no-touch` keeps to the planned height. The arm lifts 0.3 cm off the table before closing; that is in the controller. |
| The same object standing or lying needs a different closing height, typed by hand. | A new command for every pose. | Read from how it looks: standing things go to a height planned from their height, lying or very low things go to the table by touch. If a thing under 3 cm is missed, the next try goes down the other way, and what worked for that kind of thing (name + standing/lying) is used first for the rest of the run. |
| `put` / `bring` let go at a fixed 5 cm. | A small thing is dropped from several cm; a tall thing gripped higher than that is pressed into the table. | Lets go where the jaws closed when picking it up, plus 0.5 cm: its base is back on the table. `--put-height` overrides. |
| `dispense_server.py` closes at a fixed 5 cm. | Anything under 5 cm is missed. | Picks as above. |

Simulation: a 3 cm bottle cap is aimed at within 0.02 to 0.13 cm (basic aim: about
1 cm off before the reach-out). With `--hand` and a gripper leaning up to 40 degrees
the corrected aim ends within 0.1 to 0.6 cm (three camera mounts tried), against
2.6 to 6.7 cm at leaning spots without the lean correction.

## If the jaws still close a little too high: `--lower`

`--lower 0.5` makes the arm go 0.5 cm further straight down after its normal descent,
slowly and without pulling back up, and close there (0 to 2 cm). This is the **one
addition to `controller/`** on this branch: `reach.py` and `click.py` take a `lower`
field in the grab request and do nothing different without it. The arm page must be
restarted from this branch; it then reports `friend-baseline-v1+lower`. Start with
0.3 to 0.5 and watch for the fingertips pressing the table.

## Options

`--basic` the previous pickup. `--tries N` grabs per request. `--min-close CM`
lowest planned closing height. `--reach-out CM` fixed extra reach (negative pulls
in). `--no-lens`, `--no-zoom` switch those steps off. `--hand`, `--no-feel`,
`--close CM`, `--dry` as before. `dispense_server.py` takes `--basic`, `--hand`,
`--tries`, `--min-close`, `--no-feel`, `--reach-out`, `--close`.

## Not solved

- The arm's own contact stop is untouched. The careful pickup only reacts to what
  it reports, and every height is in the saved table plane's coordinates.
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
(the careful pickup; uses `grab_text.py` for detection and pictures),
`sim_scene.py` + `test_aim.py` (simulated table). `grab_text.py` and
`dispense_server.py` only gain the switch between the two pickups.
