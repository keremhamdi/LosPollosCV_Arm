"""The careful way to grab. On this branch it is what grab_text.py and dispense_server.py do;
`--basic` gives the pickup as it was, and then nothing here runs.

Nothing in controller/ is changed or needed: this only decides where to send the arm and what to
do after a miss, through the arm page's existing requests.

What it does differently (docs/PRECISE_AIM.md has the why, and how to try it one step at a time):
  - aims at the middle of where the object stands, not at the near edge of its outline, and
    straightens the wide-angle picture first (aim.py)
  - outlines small things a second time on an enlarged cut-out, so the outline is sharp
  - goes down as the default pickup does (to the planned closing height, the arm's contact stop
    as a backup), but plans that height lower for small and flat things (--min-close)
  - if the jaws close on nothing, says what the miss looked like (stopped in the air, on top of
    the object, too high or beside it), looks again and tries again: without the contact stop
    after a stop in the air, lower after closing above the table on nothing
  - "aimcheck <object>" measures once how far off the aim really is (instead of a guessed
    reach-out); until then a grab whose aim missed is retried with the other guess (0 or 1 cm)
  - with --hand: the gripper camera only accepts a match near where the object must be, and its
    correction allows for how much the gripper leans (after "handcal" is run once more)

Not yet run on the real arm: checked on a simulated table only (test_aim.py).
"""
import argparse
import io

import numpy as np
import requests
from PIL import Image

import aim
import grab_text as G
import target_selector_v5 as ts

OPEN_VIEWS = True        # show the first picture in a viewer (a server sets this to False)
LOW_CLOSE_CM = 0.8       # the lowest closing height planned for small and flat things (--min-close)
FLOOR_CLOSE_CM = 0.5     # the arm page refuses anything lower


def show_view(*args, **kwargs):
    """grab_text's picture of what was found; a server keeps it from popping up a viewer."""
    if not OPEN_VIEWS:
        G._view_opened = True
    return G.show_view(*args, **kwargs)


def options(**settings):
    """Settings for the functions here, as the command line would give them (for other programs)."""
    base = dict(arm_url="http://localhost:8890", close=None, above=14.0, side=False, no_hand=True,
                reach_out=None, tries=2, no_lens=False, no_zoom=False, no_feel=False, dry=False,
                min_close=LOW_CLOSE_CM, put_height=5.0, bring=None)
    base.update(settings)
    return argparse.Namespace(**base)


_maps = {}


def table_map(a, img_size):
    """The fixed camera's pixel -> table mapping (aim.TableMap), from the arm page's calibration marks.
    Worked out once per calibration; None if the page has fewer than 4 marks."""
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    pairs, size = st.get("pairs") or [], st.get("size") or img_size[0]
    if len(pairs) < 4:
        return None
    lens = not getattr(a, "no_lens", False)
    key = (len(pairs), size, tuple(pairs[0]), tuple(pairs[-1]), tuple(img_size), lens)
    if key not in _maps:
        _maps.clear()
        try:
            _maps[key] = aim.TableMap(pairs, size, (img_size[1], img_size[0]), lens=lens)
            print(f"  fixed camera: {len(pairs)} calibration marks, {_maps[key].lens_note}")
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"  fixed camera: can't use the calibration marks ({e})")
            _maps[key] = None
    return _maps[key]


def footprint_of(cand, tmap, img_w):
    """Where the candidate stands on the table (aim.footprint), using its sharpened outline if it has one."""
    fine = cand.get("fine")
    if fine:
        fp = aim.footprint(fine["mask"], tmap, img_w, offset=fine["offset"], zoom=fine["zoom"])
        if fp is not None:
            return fp
    return aim.footprint(np.asarray(cand["mask"]), tmap, img_w)


def in_reach(a, cand, tmap, img_w, margin=0.08):
    """Where the candidate stands (its footprint) and whether the arm can reach it. Things in the
    background or off the calibrated part of the table map far away or nowhere, and are left out."""
    if tmap is None:
        return None, False
    mask = np.asarray(cand["mask"])
    u, v = G.base_pixel(mask)
    k = tmap.size / img_w
    if not tmap.covers((u * k, v * k), margin * tmap.size):
        return None, False
    fp = footprint_of(cand, tmap, img_w)
    if fp is None or not np.all(np.isfinite(fp["center"])):
        return None, False
    d = float(np.hypot(fp["center"][0], fp["center"][1] - G.BASE_Y_CM))
    return fp, G.REACH_CM[0] <= d <= G.REACH_CM[1]


FALLBACK_CLOSE_CM = 1.5   # where the jaws close when the object's height couldn't be worked out
FLAT_LOOK_CM = 1.0        # a flat thing is taken to be about this high (for where to look from)


def grab_heights(a, h, flat=False):
    """Where the jaws close and where they come from, for an object about h cm tall (None = unknown).
    --close given: always that. Otherwise the default pickup's plan (half the object's height, at
    most 6 cm; 1.5 cm if the height is unknown), except that small things may be closed on lower
    than 1.5 cm, down to --min-close, and flat things are closed on at --min-close."""
    if a.close is not None:
        return a.close, a.above
    low = max(FLOOR_CLOSE_CM, float(getattr(a, "min_close", LOW_CLOSE_CM)))
    if not h:
        return (low if flat else FALLBACK_CLOSE_CM), a.above
    close = min(6.0, max(min(low, 1.5), 0.5 * h))
    return round(close, 1), max(a.above, round(h + 5, 1))


ZOOM_BELOW = 0.25         # objects narrower than this share of the picture get a second, enlarged look
ZOOM_MAX = 4.0
SAME_OUTLINE = 0.5        # the second look must overlap the first outline at least this much (IoU)


def sharpen(img, cand, a=None):
    """A small object's outline again, sharper: SAM 3 on an enlarged cut-out around it.

    In the full picture a small thing is a few dozen pixels and its outline is rough at the edges,
    which is exactly where its position and height are read from. Returns the candidate with
    "fine" = {"mask" (enlarged), "offset", "zoom"} added, or unchanged if the second look doesn't
    clearly show the same object."""
    import cv2
    if a is not None and getattr(a, "no_zoom", False):
        return cand
    try:
        mask = np.asarray(cand["mask"]).astype(bool)
        ys, xs = np.nonzero(mask)
        if len(xs) < 4:
            return cand
        W, H = img.size
        box = cand.get("box")
        x0, y0, x1, y1 = box if box is not None else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        bw, bh = x1 - x0, y1 - y0
        if max(bw, bh) > ZOOM_BELOW * W:
            return cand
        pad = max(40.0, 0.8 * max(bw, bh))
        bx0, by0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        bx1, by1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
        cw, ch = bx1 - bx0, by1 - by0
        z = min(ZOOM_MAX, 1008.0 / max(cw, ch))
        if z < 1.5:
            return cand
        zw, zh = int(round(cw * z)), int(round(ch * z))
        big = img.crop((bx0, by0, bx1, by1)).resize((zw, zh), Image.BICUBIC)
        prompts = sorted(cand.get("prompts") or ([cand["prompt"]] if cand.get("prompt") else []))
        if not prompts:
            return cand
        old = ts.SAM_SCORE_MIN
        ts.SAM_SCORE_MIN = min(old, 0.25)
        try:
            found = ts.sam3_candidates(big, prompts)
        finally:
            ts.SAM_SCORE_MIN = old
        was = cv2.resize(mask[by0:by1, bx0:bx1].astype(np.uint8), (zw, zh), interpolation=cv2.INTER_NEAREST).astype(bool)
        best, best_iou = None, 0.0
        for c in found:
            m = np.asarray(c["mask"]).astype(bool)
            if m.shape != was.shape:
                continue
            iou = np.count_nonzero(m & was) / max(1, np.count_nonzero(m | was))
            if iou > best_iou:
                best, best_iou = m, iou
        # the same thing, whole: it overlaps the first outline well and is about as big
        if best is None or best_iou < SAME_OUTLINE or not 0.6 <= best.sum() / max(1, was.sum()) <= 1.6:
            return cand
        out = dict(cand)
        out["fine"] = {"mask": best, "offset": (bx0, by0), "zoom": zw / cw}
        small = cv2.resize(best.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_AREA)
        full = np.zeros_like(mask)
        full[by0:by1, bx0:bx1] = small > 0
        out["mask"] = full
        return out
    except Exception as e:   # a second look must never stop a grab
        print(f"  (closer look skipped: {e.__class__.__name__})")
        return cand


SAME_OBJECT_CM = 10.0     # trying again: a match within this of the last aim is taken to be the same object


def frame(a):
    """The fixed camera's current picture (from the arm page)."""
    return Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")


def make_target(img, tmap, cand, text, what, score, fp=None, hand_text=None):
    """Everything the grab needs to know about one found object (None if its outline is unusable)."""
    fp = fp or footprint_of(cand, tmap, img.width)
    if fp is None or not np.all(np.isfinite(fp["center"])):
        return None
    x, y = fp["center"]
    k = img.width / tmap.size
    bu, bv = G.base_pixel(np.asarray(cand["mask"]))
    u, v = tmap.pixel_of((x, y), start=(bu / k, bv / k))       # where that table point shows, for the picture
    return {"text": text, "hand_text": hand_text or text, "what": what, "score": float(score),
            "x": float(x), "y": float(y), "u": u * k, "v": v * k, "height": fp["height"],
            "width": fp["width"], "flat": fp["flat"], "how": fp["how"], "mask": cand["mask"]}


def describe(t):
    """A few words on how the object was sized up, for the picture and the printout."""
    if t["height"]:
        return f"about {t['height']:.1f} cm tall, {t['width']:.1f} cm wide"
    if t["flat"]:
        return f"lying flat, {t['width']:.1f} cm wide"
    return f"{t['width']:.1f} cm wide, height unknown"


def find_by_look(a, img, tmap, text, near=None):
    """The object that looks like `text`, among those the arm can reach.
    Returns (target or None, whether anything looking like it was seen at all)."""
    res = G.detect_fixed(img, text)
    cands = res["candidates"]
    if not cands:
        return None, False
    reach = []
    for c in cands:
        fp, ok = in_reach(a, c, tmap, img.width)
        c["out_of_reach"] = not ok
        where = f"x={fp['center'][0]:.1f}, y={fp['center'][1]:.1f}" if fp else "?"
        print(f"  candidate {c['target_score']:.2f} at {where} cm: " + ("in reach" if ok else "out of reach, ignored"))
        if ok:
            reach.append((c, fp))
    if not reach:
        show_view(img, cands, None, None, None, f"'{text}': found, but nothing within reach")
        print(f"Found '{text}', but nothing within the arm's reach. Not grabbing.")
        return None, True
    if near is not None:   # trying again: stay with the same object if it is still about where it was
        same = [cf for cf in reach if np.hypot(cf[1]["center"][0] - near[0], cf[1]["center"][1] - near[1]) <= SAME_OBJECT_CM]
        reach = same or reach
    best, fp = max(reach, key=lambda cf: cf[0]["target_score"])   # the highest score among what the arm can reach
    sharp = sharpen(img, best, a)
    if sharp is not best:
        cands = [sharp if c is best else c for c in cands]
        fp = footprint_of(sharp, tmap, img.width) or fp
    t = make_target(img, tmap, sharp, text, f"'{text}'", best["target_score"], fp)
    if t is None:
        return None, True
    show_view(img, cands, sharp, t["u"], t["v"], f"'{text}': {describe(t)}")
    return t, True


WORD_OBJECTS = ["medicine", "blister pack", "bottle", "box", "object"]   # what writing is usually on


def find_by_word(a, img, tmap, text, min_conf=0.0, prompts=None):
    """The object that has `text` written on it (read with at least min_conf confidence), or None."""
    try:
        hit = G.find_word(img, text)
    except ImportError:
        print("To read writing, install the reader once: uv pip install easyocr")
        return None
    if hit is None or hit[3] < min_conf:
        show_view(img, [], None, None, None, f"'{text}': nothing found")
        print(f"Couldn't find '{text}' by look or by writing. Not grabbing.")
        return None
    wu, wv, read, conf = hit
    print(f"Read '{read}' (confidence {conf:.2f}) at pixel {wu:.0f},{wv:.0f}")
    obj = G.object_at(img, wu, wv, prompts or WORD_OBJECTS)
    if obj is None:
        show_view(img, [], None, None, None, f"'{text}': writing found (circle), no object around it", word_at=(wu, wv))
        print("Found the writing, but no object around it. Not grabbing.")
        return None
    fp, ok = in_reach(a, obj, tmap, img.width)
    if not ok:
        show_view(img, [], obj, None, None, f"'{text}' written on the green one: out of reach", score_key="sam_score", word_at=(wu, wv))
        print(f"Found '{text}' written on something, but the arm can't reach it. Not grabbing.")
        return None
    obj = sharpen(img, obj, a)
    # the gripper camera can't read: from above it looks for the kind of thing the writing was on
    t = make_target(img, tmap, obj, text, f"the thing with '{text}' on it", obj["sam_score"],
                    hand_text=obj.get("prompt") or text)
    if t is None:
        return None
    t["read"], t["read_confidence"] = read, float(conf)
    show_view(img, [], obj, t["u"], t["v"], f"'{text}' written on the green one: {describe(t)}",
              score_key="sam_score", word_at=(wu, wv))
    return t


def locate(a, text, near=None):
    """Find `text` in the fixed camera: by look; if nothing looks like it, by writing.
    Returns what make_target gives, or None (having said why)."""
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    if st.get("pending"):
        print("The page is waiting for a calibration click: finish or Undo that first.")
        return None
    if st.get("read_only"):
        print("The controller runs with --read-only: restart it without that to move the arm.")
        return None
    img = frame(a)
    tmap = table_map(a, img.size)
    if tmap is None:
        print("The fixed camera isn't calibrated yet: the arm page needs 4 or more marks.")
        return None
    t, seen = find_by_look(a, img, tmap, text, near)
    if t is not None or seen:
        return t
    print(f"Nothing looks like '{text}'. Looking for it written on something...")
    return find_by_word(a, img, tmap, text)


def clear_view(a, t, above):
    """After a miss: open the jaws and take the arm out of the fixed camera's view of the object,
    so it can be looked for again. Parks if there is a park pose, else pulls back towards the base."""
    try:
        requests.post(a.arm_url + "/api/release", json={}, timeout=30)
        if requests.post(a.arm_url + "/api/park", json={}, timeout=180).json().get("ok"):
            return
        r = np.array([t["x"], t["y"] - G.BASE_Y_CM])
        d = float(np.linalg.norm(r))
        back = max(G.REACH_CM[0] + 4.0, d - 14.0)
        if d > 1e-6 and back < d:
            goto(a, r[0] / d * back, G.BASE_Y_CM + r[1] / d * back, above)
    except (RuntimeError, ValueError, requests.RequestException) as e:
        print(f"  (couldn't move the arm out of the view: {e})")


def next_descent(res, t):
    """How to go about the next try, from where the arm stopped on this one. Returns
    {"stop": use the arm's contact stop on the way down, "close": closing height or None for the plan,
     "reach": guessed extra reach in cm, or None to keep it}.
      it stopped clearly in the air (nothing can have been there): the same height, without the
          contact stop, so it can't stop early again
      it closed at the planned height on nothing, having touched nothing: half as high
      a fingertip came down on top of the object: the aim was off by a centimetre or more. With
          nothing measured about the aim, try the other guess: 1 cm more reach, or none
      otherwise (the jaws closed beside it): the same again, at the new aim"""
    rep = res.get("report") or {}
    met, closed = rep.get("met_at_cm"), res.get("close")
    top = t["height"] if t["height"] else (FLAT_LOOK_CM if t["flat"] else None)
    if rep.get("contact") and met is not None and top is not None and met > top + IN_THE_AIR_CM:
        return {"stop": False, "close": None, "reach": None}
    if rep.get("contact") is False and closed is not None:
        return {"stop": True, "close": max(FLOOR_CLOSE_CM, round(0.5 * closed, 1)), "reach": None}
    other = None
    if res.get("reach") is not None and not rep.get("slipped") and landed_on_it(rep, t):
        other = 0.0 if abs(res["reach"]) > 1e-6 else SHORT_GUESS_CM
    return {"stop": True, "close": None, "reach": other}


def grab_with_retries(a, t, find_again, tries=None):
    """Grab `t`. If the jaws close on nothing: open, take the arm out of the view, find it again with
    find_again(near) and grab again, up to `tries` grabs (--tries). Returns the last grab's result
    (see grab_target) with "attempts" added."""
    tries = max(1, int(tries if tries is not None else getattr(a, "tries", 2)))
    descent, res = None, None
    for n in range(1, tries + 1):
        res = grab_target(a, t, descent)
        res["attempts"] = n
        if res["holding"] and res["reach"] is not None and res["reach"] != _session["reach"]:
            _session["reach"] = res["reach"]     # this guess got it: start with it from now on
            print(f"  Got it with {res['reach']:g} cm extra reach, so the next grabs start with that. "
                  'To measure it instead of guessing, type "aimcheck <object>" once.')
        if a.dry or res["holding"] or not res["ok"] or n == tries:
            break
        print(f"Didn't get it: {res['why']}. Looking again (try {n + 1} of {tries})...")
        clear_view(a, t, res["above"])
        descent = next_descent(res, t)
        again = find_again((t["x"], t["y"]))
        if again is None:
            print("Can't find it any more after that miss. Stopping.")
            res["why"] += "; then it could not be found again"
            break
        t = again
    return res


def find_and_grab(a, text, tries=None):
    """Find `text` and grab it, looking and trying again after a miss (--tries).
    Returns the last grab's result (see grab_target), or None if it was never found."""
    t = locate(a, text)
    if t is None:
        return None
    return grab_with_retries(a, t, lambda near: locate(a, text, near), tries)


def find_hand(a, text, near, close_enough=None, tries=2):
    """The object's middle in the gripper camera's picture: ((u, v), picture, outline) or (None, picture, None).

    Like grab_text.find_in_hand, but a match only counts if it can be the object: close_enough(centre)
    says whether a blob is near enough to where the object must show. (A real run took something
    710 px from the expected spot for the object; the whole correction was then thrown away. Here
    such a match is passed over and the object is looked for where it has to be.)"""
    import time
    if close_enough is None:
        return G.find_in_hand(a, text, near=near, tries=tries)
    ts.VLM_BACKEND = "none"
    near = np.asarray(near, float)

    def centre(c):
        ys, xs = np.nonzero(np.asarray(c["mask"]))
        return np.array([xs.mean(), ys.mean()])

    def nearest(cands):
        seen = [(float(np.linalg.norm(centre(c) - near)), c) for c in cands]
        seen = [(d, c) for d, c in seen if close_enough(centre(c))]
        return min(seen, key=lambda dc: dc[0])[1] if seen else None
    img, passed_over = None, 0
    for attempt in range(tries):
        time.sleep(0.8)                               # let the camera catch up after moving
        img = G.hand_frame(a)
        old = ts.SAM_SCORE_MIN
        ts.SAM_SCORE_MIN = 0.2 if attempt else old    # later tries: accept weaker matches
        try:
            res = ts.decide(img, text, None)
        finally:
            ts.SAM_SCORE_MIN = old
        cands = [c for c in res["candidates"] if np.asarray(c["mask"]).sum() >= 60]   # not specks
        if not cands:
            continue
        top = max(c["target_score"] for c in cands)
        good = [c for c in cands if c["target_score"] >= 0.5 * top]
        best = nearest(good) or nearest(cands)
        if best is not None:
            c = centre(best)
            return (float(c[0]), float(c[1])), img, best
        passed_over = len(cands)
    if passed_over and "far_match" not in _told:
        _told.add("far_match")
        print(f"  gripper camera: saw '{text}' only far from where the object must be; not taking that for it")
    if img is not None:
        # From straight above things look different (a standing cylinder is just a disc), so the name
        # may not match. Last try: any small object, the one nearest where it should be.
        res = ts.decide(img, G.HAND_FALLBACK[0], G.HAND_FALLBACK[1])
        n_px = img.width * img.height
        small = [c for c in res["candidates"] if 30 < np.asarray(c["mask"]).sum() < 0.03 * n_px]
        best = nearest(small)
        if best is not None:
            c = centre(best)
            print(f"  gripper camera: '{text}' not recognised from above; using the small object where it should be")
            return (float(c[0]), float(c[1])), img, best
    return None, img, None


class CantReach(RuntimeError):
    """The arm page says this point can't be reached (with this lean)."""


def goto(a, x, y, h, tilt=None):
    """Move the fingertips above table x, y at height h (cm). tilt: how far the gripper leans from
    straight down, degrees (default: the least that reaches). Returns the lean used, or None if the
    page doesn't say."""
    body = {"x": x, "y": y, "height": h}
    if tilt is not None:
        body["tilt"] = tilt
    r = requests.post(a.arm_url + "/api/goto", json=body, timeout=120).json()
    if not r.get("ok"):
        err = r.get("error") or "move failed"
        raise (CantReach if err.startswith("Can't reach") else RuntimeError)(err)
    return r.get("tilt")


def look_from(a, x, y, look, like=None):
    """Go and look down from `look` cm above x, y. With `like` (the lean the setup was done with) it
    leans exactly that much if the arm can, so the picture is the setup's picture; else the least
    lean that reaches. Returns the lean used (degrees), or None if unknown."""
    if like is not None:
        try:
            goto(a, x, y, look, tilt=like)
            return like
        except CantReach:
            pass
    return goto(a, x, y, look)


LEAN_STEP_DEG = 5        # the setup looks at the object with the gripper leaning this much more each time ...
LEAN_STEPS = 8           # ... up to this many times (and as often with less lean, if it leaned to begin with)


CAMERA_BACK_CM = 6.0     # geometry's guess of how far behind the fingertips the gripper camera sits


def _lean_cm(cal, tilt):
    """An object right under the fingertips, seen with the gripper leaning `tilt` degrees, shows where
    in the setup's picture an object this many cm nearer the base would have shown. By geometry:
    the fingertips are the pivot, the object is look_above below them, the camera a little behind."""
    t0, t = np.radians(float(cal["tilt"])), np.radians(float(tilt))
    D, b = float(cal.get("look_above", G.LOOK_ABOVE)), CAMERA_BACK_CM
    T = D * np.sin(t) / (D * np.cos(t) + b)
    return (T * (D * np.cos(t0) + b) - D * np.sin(t0)) / (np.cos(t0) + T * np.sin(t0))


def _lean_geometry(cal, J, x, y, tilt, frm):
    """Pixels the spot under the jaws moves when the lean goes from `frm` to `tilt` degrees, by geometry."""
    return J @ (G.arm_frame(x, y)[:, 0] * (_lean_cm(cal, tilt) - _lean_cm(cal, frm)))


def lean_shift(cal, J, x, y, tilt):
    """How far (pixels) the spot under the jaws moves in the gripper camera's picture because the
    gripper leans differently here than in the setup.

    The arm keeps its gripper straight down near the base and leans it outwards further away. The
    camera leans with it, but the fingertips still go straight down. So with more lean, an object
    right under the fingertips shows where, in the setup, an object nearer the base would have shown
    (about 2 cm for every 10 degrees). Not allowing for that leaves the arm short.
    Uses what the setup measured at each lean; beyond those leans (or without them), that geometry."""
    t0 = cal.get("tilt")
    if t0 is None or tilt is None:
        return np.zeros(2)
    t0, tilt = float(t0), float(tilt)
    if abs(tilt - t0) < 1e-6:
        return np.zeros(2)
    lean = cal.get("lean") or {}
    degs, px = np.array(lean.get("deg", []), float), np.array(lean.get("px", []), float).reshape(-1, 2)
    if len(degs) < 2:
        return _lean_geometry(cal, J, x, y, tilt, t0)
    inside = float(np.clip(tilt, degs.min(), degs.max()))
    seen = np.array([np.interp(inside, degs, px[:, 0]), np.interp(inside, degs, px[:, 1])]) - np.array(cal["target_px"], float)
    return seen + (_lean_geometry(cal, J, x, y, tilt, inside) if inside != tilt else 0.0)


def hand_calibrate(a, text):
    """One-time: where an object right under the jaws appears in the gripper camera, how the
    picture shifts when the arm moves (pixels per cm), and how it shifts when the gripper leans.
    Saved to controller/calib/hand.json."""
    import json
    print("Gripper camera setup. Put the object on the table (in the arm's reach).")
    input("Follower on MIRRORING: put the open jaws right around the object, low, centred. Then Enter. ")
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    if not st.get("tip"):
        print("The page shows no tip position. Is the controller running?"); return
    x, y = st["tip"][0], st["tip"][1]
    print(f"Object is at x={x}, y={y} cm.")
    try:
        obj_h = float(input("How tall is this object, in cm (a ruler is fine)? ").replace(",", "."))
    except ValueError:
        print("Need a number, e.g. 3.5"); return
    if obj_h < 0.5:
        print("That's very flat; using 0.5 cm. (It only sets how high the camera looks from.)")
        obj_h = 0.5
    look = obj_h + G.LOOK_ABOVE
    input("Now follower on (Self-controlled), press Hold arm on the page, then Enter here. The arm will move. ")
    pts, tilt = {}, None
    # moving the arm back (-y) moves the object UP in the picture, away from the jaws, so it stays visible
    for name, dx, dy in (("centre", 0, 0), ("+x", 2, 0), ("-y", 0, -2)):
        try:   # all three looks with the same lean: a change of lean would look like a shift of the object
            got = goto(a, x + dx, y + dy, look, tilt=tilt)
        except CantReach:
            print(f"The arm can't keep the same lean for the look at {name}. Do the setup a little nearer the base.")
            goto(a, x, y, look + 4)
            return
        if name == "centre":
            tilt = got
        # first look: the match nearest the spot between the jaws; then: the same object as in the first look
        p, img, best = G.find_in_hand(a, text, near=pts.get("centre", G.HAND_JAW_PX))
        if p is None:
            print(f"The gripper camera can't see '{text}' from there. Try a clearer object or description."); return
        pts[name] = np.array(p)
        G.save_hand_view(img, best, p, pts["centre"], f"cal_{name}")
        print(f"  {name}: object at pixel {p[0]:.0f},{p[1]:.0f}")
    J = np.column_stack([(pts["+x"] - pts["centre"]) / 2, (pts["-y"] - pts["centre"]) / -2])   # pixels per cm
    nx, ny = np.linalg.norm(J[:, 0]), np.linalg.norm(J[:, 1])
    cos = abs(float(J[:, 0] @ J[:, 1])) / max(1e-9, nx * ny)
    if min(nx, ny) < 4 or max(nx, ny) > 60 or max(nx, ny) / min(nx, ny) > 3 or cos > 0.85:
        print(f"Setup looks wrong ({nx:.0f} and {ny:.0f} px per cm, {np.degrees(np.arccos(min(1, cos))):.0f} deg apart; "
              f"at least 30 deg apart and within 3x of each other): it probably "
              f"picked a different object in one look. See pick_views/hand_latest_cal_*.jpg. Not saved; try again.")
        goto(a, x, y, look + 4)
        return
    data = {"target_px": pts["centre"].tolist(), "px_per_cm": J.tolist(), "look_above": G.LOOK_ABOVE,
            "at": [x, y]}   # the gripper camera turns with the base: remember which way the arm pointed
    if tilt is None:
        print("  (this arm page doesn't say how far the gripper leans: update controller/arm/click.py "
              "to have the lean allowed for)")
    else:
        data["tilt"] = float(tilt)   # ... and how far the gripper leaned
        lean = measure_lean(a, text, x, y, look, float(tilt), data, pts["centre"])
        if lean:
            data["lean"] = lean
    open(G.HAND_FILE, "w").write(json.dumps(data, indent=1))
    print(f"Saved {G.HAND_FILE}: the object should appear at pixel {pts['centre'][0]:.0f},{pts['centre'][1]:.0f}; "
          f"about {np.linalg.norm(J[:, 0]):.0f} px per cm.")
    goto(a, x, y, look + 4)


def measure_lean(a, text, x, y, look, tilt, cal, centre):
    """Setup, last step: lean the gripper more (and less) over the same spot, a few degrees at a time,
    and note where the object shows each time. Returns {"deg": [...], "px": [[u, v], ...]} (sorted by
    lean, the setup's own look included), or None if the arm can't lean differently here."""
    J = np.array(cal["px_per_cm"])
    seen = {float(tilt): np.array(centre, float)}
    for way in (1, -1):
        at, p_at = float(tilt), np.array(centre, float)
        for _ in range(LEAN_STEPS):
            t = at + way * LEAN_STEP_DEG
            if not 0 <= t <= 80:
                break
            try:
                goto(a, x, y, look, tilt=t)
            except CantReach:
                break
            expect = _lean_geometry(cal, J, x, y, t, at)
            limit = max(60.0, 2.0 * float(np.linalg.norm(expect)))     # the same object: about where geometry says
            p, img, best = find_hand(a, text, p_at + expect,
                                     lambda c, at=p_at + expect, limit=limit: np.linalg.norm(c - at) <= limit)
            if p is None:
                break                                        # out of the picture: far enough
            step = np.array(p) - p_at
            ne, ns = np.linalg.norm(expect), np.linalg.norm(step)
            if not 0.3 * ne <= ns <= 3.0 * ne:
                print(f"  lean {t:g} deg: the object moved {ns:.0f} px where about {ne:.0f} px was expected "
                      "(another object picked?): stopping there")
                break
            if float(step @ expect) < 0 and "lean_way" not in _told:
                _told.add("lean_way")
                print("  lean: the object moved the opposite way to what geometry says. Using what the camera saw; "
                      "if grabs far from the base get worse, delete \"lean\" from hand.json and report it.")
            G.save_hand_view(img, best, p, centre, "cal_lean")
            seen[t] = np.array(p, float)
            at, p_at = t, np.array(p, float)
    if len(seen) < 2:
        print("  lean: the arm can't lean differently at this spot, so the lean is worked out from geometry "
              "(fine near here; for far objects a setup nearer the base measures it)")
        return None
    degs = sorted(seen)
    print(f"  lean: measured from {degs[0]:g} to {degs[-1]:g} deg; the spot under the jaws moves "
          f"{np.linalg.norm(seen[degs[-1]] - seen[degs[0]]):.0f} px over that")
    return {"deg": degs, "px": [seen[d].tolist() for d in degs]}


_told = set()


def hand_align(a, text, x, y, obj_h=None):
    """Hover above (x, y), look with the gripper camera, shift until the object is where the jaws close.
    Returns (x, y, info): the corrected aim (or the same if the gripper camera can't help), and
    info = {"used": the aim comes from the gripper camera, "lean_aware": its setup knows the lean}."""
    import json
    import os
    info = {"used": False, "lean_aware": False}
    if not os.path.exists(G.HAND_FILE):
        print("  gripper camera: not set up yet (run handcal), keeping the fixed camera's aim")
        return x, y, info
    cal = json.load(open(G.HAND_FILE))
    target0, J0 = np.array(cal["target_px"], float), np.array(cal["px_per_cm"], float)
    c0, c1 = np.linalg.norm(J0[:, 0]), np.linalg.norm(J0[:, 1])
    if abs(float(J0[:, 0] @ J0[:, 1])) / max(1e-9, c0 * c1) > 0.85:
        print("  gripper camera: its saved setup is broken (the two test moves point the same way). Run handcal again.")
        return x, y, info
    cal_tilt = cal.get("tilt")
    info["lean_aware"] = cal_tilt is not None
    if cal_tilt is None and "handcal" not in _told:
        _told.add("handcal")
        print("  gripper camera: this setup doesn't know how the gripper leans. Run handcal once more "
              "for a better aim away from where it was set up.")
    look = (obj_h if obj_h else 3.0) + cal.get("look_above", G.LOOK_ABOVE)   # same distance above its top as in setup
    x0, y0 = x, y
    try:   # open jaws: closed ones block the gripper camera's view
        requests.post(a.arm_url + "/api/release", json={}, timeout=30)
    except requests.RequestException:
        pass

    def look_at(px, py, tries):
        """One look from above px, py: where the object is in the picture, where it should be, and
        how the picture moves per cm there."""
        tilt = look_from(a, px, py, look, cal_tilt)
        # setup was done with the arm pointing one way; turn the table directions to match now
        J = J0 @ G.arm_frame(*cal["at"]) @ G.arm_frame(px, py).T if "at" in cal else J0
        shift = lean_shift(cal, J, px, py, tilt)
        target = target0 + shift
        if np.linalg.norm(shift) > 1 and "lean" not in _told:
            _told.add("lean")
            print(f"  gripper camera: leaning {tilt:g} deg here ({cal_tilt:g} in its setup), allowed for")
        def close_enough(c):    # the fixed camera put the arm within a few cm: further off, it's something else
            return np.hypot(*np.linalg.solve(J, target - c)) <= G.MAX_FIX_CM
        p, img, best = find_hand(a, text, target, close_enough, tries)
        return p, img, best, target, J

    # 1. find it: look straight above the fixed camera's aim; if it isn't in view, look around it
    found, img, target = None, None, target0
    for k, (dx, dy) in enumerate(G.SEARCH_CM):
        lx, ly = x + dx, y + dy
        try:
            p, img, best, target, J = look_at(lx, ly, 2 if k == 0 else 1)
        except CantReach:
            if k == 0:
                raise
            continue
        if p is not None:
            found = (lx, ly, p, img, best, target, J)
            break
        if k == 0:
            print("  gripper camera: not in view yet, looking around...")
    if found is None:
        G.save_hand_view(img, None, None, target, 0)
        print("  gripper camera: couldn't find it, keeping the fixed camera's aim "
              "(last picture: pick_views/hand_latest_0.jpg)")
        return x0, y0, info
    # 2. perfect it: shift until it sits where the jaws close (up to 3 looks)
    x, y, p, img, best, target, J = found
    last = None
    for step in range(3):
        if step:
            p, img, best, target, J = look_at(x, y, 2)
            if p is None:
                print("  gripper camera: lost it, using the last good aim")
                info["used"] = True
                return x, y, info
        d = np.linalg.solve(J, target - np.array(p))       # how far to move (cm) to bring it under the jaws
        print(f"  gripper camera: off by {np.hypot(*d):.1f} cm ({np.hypot(*(target - np.array(p))):.0f} px)")
        G.save_hand_view(img, best, p, target, step)
        if last is not None and np.hypot(*d) > last + 0.3:
            print("  gripper camera: getting worse, not trusting it, keeping the fixed camera's aim")
            return x0, y0, info
        last = np.hypot(*d)
        nx, ny = x + d[0], y + d[1]
        if np.hypot(nx - x0, ny - y0) > G.MAX_FIX_CM:
            print(f"  gripper camera wants more than {G.MAX_FIX_CM} cm from the fixed camera's aim: not trusting it")
            return x0, y0, info
        x, y = nx, ny
        if np.hypot(*d) < 0.3:
            break
    info["used"] = True
    print(f"  gripper camera: aim moved by {np.hypot(x - x0, y - y0):.1f} cm in total")
    return x, y, info


OLD_SETUP_REACH_OUT = 1.0   # cm further out, only with a gripper camera setup that doesn't know the lean
SHORT_GUESS_CM = 1.0        # "the arm stops about this short": tried on the second grab if the first one's aim missed
AIM_FILE = "controller/calib/aim.json"   # what "aimcheck" measured: how far off the fixed camera's aim really is
AIM_BLEND_CM = 8.0          # a check counts fully near where it was made, less further away
AIM_CHECK_ABOVE_CM = 1.5    # aimcheck holds the open jaws this far above the object's top
_session = {"reach": 0.0}   # the reach guess that last got something, for the next grabs of this run


def aim_checks():
    import json
    import os
    if not os.path.exists(AIM_FILE):
        return []
    try:
        return [c for c in json.load(open(AIM_FILE)).get("checks", [])
                if all(np.isfinite(v) for v in (*c["at"], c["out"], c["left"]))]
    except (ValueError, KeyError, TypeError, OSError):
        return []


def aim_offset(x, y):
    """How far the arm has to go beyond the fixed camera's aim at x, y to really be over the object:
    (cm further out from the base, cm to the left seen from the base), from the saved aim checks.
    None if no check was ever saved."""
    checks = aim_checks()
    if not checks:
        return None
    w = np.array([1.0 / ((c["at"][0] - x) ** 2 + (c["at"][1] - y) ** 2 + AIM_BLEND_CM ** 2) for c in checks])
    w /= w.sum()
    return (float(sum(wk * c["out"] for wk, c in zip(w, checks))),
            float(sum(wk * c["left"] for wk, c in zip(w, checks))))


def aim_check(a, text):
    """Measure, once, how far off the aim really is: the arm holds its open jaws just above the
    object, you say which way they should move until they are centred over it, and that is saved
    and added to every aim from then on (instead of a guessed reach-out). "aimcheck clear" forgets it."""
    import json
    import os
    if text.strip().lower() == "clear":
        if os.path.exists(AIM_FILE):
            os.remove(AIM_FILE)
        print("Aim checks forgotten.")
        return
    t = locate(a, text)
    if t is None:
        return
    top = t["height"] if t["height"] else (FLAT_LOOK_CM if t["flat"] else 3.0)
    hover, above = max(2.0, top + AIM_CHECK_ABOVE_CM), max(a.above, round(top + 5, 1))
    F = G.arm_frame(t["x"], t["y"])                    # columns: away from the base, and to its left
    had = aim_offset(t["x"], t["y"]) or (0.0, 0.0)
    move = np.array(had, float)
    print(f"Aim check on {t['what']} at x={t['x']:.1f}, y={t['y']:.1f} cm ({describe(t)}). The open jaws will stop "
          f"{hover:.1f} cm above the table, over where it thinks the object is.")
    input("Follower on (Self-controlled), Hold arm pressed. Enter to move (Ctrl+C to cancel)... ")
    requests.post(a.arm_url + "/api/release", json={}, timeout=30)

    def go_to(h):
        x, y = np.array([t["x"], t["y"]]) + F @ move
        goto(a, float(x), float(y), h)
    go_to(above)
    go_to(hover)
    print("""Look straight down on the jaws. Are they centred over the object? Type where they should go, in cm:
   out 1      further from the arm's base        in 0.5    back towards it
   left 0.5   left, seen from the base           right 1   right, seen from the base
   Enter      they are centred: save             q         cancel, save nothing""")
    ways = {"out": (1, 0), "in": (-1, 0), "left": (0, 1), "right": (0, -1)}
    while True:
        line = input("aimcheck> ").strip().lower()
        if line in ("q", "quit"):
            go_to(above)
            print("Cancelled, nothing saved.")
            return
        if not line:
            break
        try:
            way, cm = line.split()
            step = np.array(ways[way], float) * float(cm.replace(",", "."))
        except (ValueError, KeyError):
            print('Like "out 1", "left 0.5", or just Enter when centred.')
            continue
        if np.linalg.norm(step) > 3 or np.linalg.norm(move + step) > 5:
            print("That is a lot (over 3 cm at once, or 5 cm in all): the calibration needs redoing rather than this.")
            continue
        move = move + step
        try:
            go_to(hover)
        except RuntimeError as e:
            move = move - step
            print(f"  can't go there: {e}")
            continue
        print(f"  now {move[0]:+.1f} cm out, {move[1]:+.1f} cm left of the camera's aim")
    checks = [c for c in aim_checks() if np.hypot(c["at"][0] - t["x"], c["at"][1] - t["y"]) > 5.0]   # replaces a check made here
    checks.append({"at": [round(t["x"], 1), round(t["y"], 1)], "out": round(float(move[0]), 2), "left": round(float(move[1]), 2)})
    os.makedirs(os.path.dirname(AIM_FILE), exist_ok=True)
    with open(AIM_FILE, "w") as f:
        json.dump({"checks": checks}, f, indent=1)
    go_to(above)
    print(f"Saved {AIM_FILE}: here the arm goes {move[0]:+.1f} cm out and {move[1]:+.1f} cm left of the camera's aim "
          f"({len(checks)} check{'s' if len(checks) != 1 else ''} in all). One or two more at other distances from the base help.")


def landed_on_it(rep, t):
    """Did the descent stop on top of a small object (a fingertip came down on it)?"""
    met = rep.get("met_at_cm")
    low = t["height"] if t["height"] else (FLAT_LOOK_CM if t["flat"] else None)
    return bool(rep.get("contact") and met is not None and low is not None and low < 4.0
                and max(1.2, 0.5 * low) < met <= low + IN_THE_AIR_CM)


def why_missed(rep, t, close=None):
    """In a few words, what most likely went wrong when the jaws closed on nothing."""
    h, met = t["height"], rep.get("met_at_cm")
    if rep.get("slipped"):
        return "it had it, but lost it on the way up"
    if rep.get("contact") and met is not None:
        low = h if h else (FLAT_LOOK_CM if t["flat"] else None)
        if low is not None and met > low + IN_THE_AIR_CM:
            return f"it stopped {met:.1f} cm up, above the object, as if it touched something there"
        if landed_on_it(rep, t):
            return f"it stopped {met:.1f} cm up, on top of the object: a fingertip came down on it (aim a little off)"
        return "the jaws closed beside it (aim a little off)"
    if rep.get("contact") is False:
        at = rep.get("closed_at_cm", close)
        return (f"it closed {at} cm above the table without touching anything: too high for this object, or beside it"
                if at is not None else "it closed without touching anything: too high for this object, or beside it")
    return "the jaws closed on nothing"


IN_THE_AIR_CM = 1.5      # stopping this far above the object's top: nothing was there, a false touch


def grab_target(a, t, descent=None):
    """Grab the object `t` (from make_target): with --hand centre the jaws with the gripper camera,
    then go down to the planned closing height (the arm's contact stop can end it earlier), close, lift.
    `descent`: what next_descent gives after a miss, to go down differently.
    Returns {"ok": the arm did the grab, "holding": it has something, "why": what went wrong if not,
             "report": what the arm page measured, "error": the page's refusal if any,
             "above": lift height, "close": the closing height asked for, "reach": guessed extra reach or None}."""
    h = t["height"]
    close, above = grab_heights(a, h, t["flat"])
    descent = descent or {"stop": True, "close": None}
    if descent.get("close") is not None and a.close is None:
        close = descent["close"]
    look_h = h if h else (FLAT_LOOK_CM if t["flat"] else None)
    out = {"ok": True, "holding": False, "why": "", "report": {}, "error": None, "above": above, "close": close,
           "reach": None}    # "reach": the guessed extra reach used, when nothing better was known
    print(f"Found {t['what']} (score {t['score']:.2f}): {describe(t)}, standing at table x={t['x']:.1f}, y={t['y']:.1f} cm")
    stop = not a.side and not a.no_feel and descent.get("stop", True)
    print(f"Planned closing height: {close:g} cm above the calibrated table; approach {above:g} cm; "
          + ("the arm's contact stop can end the descent earlier." if stop else "no contact stop on this descent."))
    if a.dry:
        return out
    body = {"height": close, "above": above, "method": "side" if a.side else "top"}
    if stop:
        body["guard"] = True     # the arm's own guarded descent, ending at `height` (never a search below the table)
    gx, gy = t["x"], t["y"]
    if not a.side:
        hand = {"used": False, "lean_aware": False}
        if not a.no_hand:
            try:
                gx, gy, hand = hand_align(a, t["hand_text"], gx, gy, look_h)
            except (RuntimeError, ValueError, KeyError, requests.RequestException, np.linalg.LinAlgError) as e:
                print(f"  gripper camera skipped: {e}")
        # How far beyond that aim to go (cm out from the base, cm to its left), in this order of trust:
        # what you asked for; the gripper camera's own aim; what "aimcheck" measured; a guess that is
        # switched between 0 and 1 cm when a grab's aim misses.
        saved = aim_offset(gx, gy)
        if a.reach_out is not None:
            shift, why = (a.reach_out, 0.0), "--reach-out"
        elif hand["used"]:
            shift, why = (0.0 if hand["lean_aware"] else OLD_SETUP_REACH_OUT, 0.0), "gripper camera setup without lean"
        elif saved is not None:
            shift, why = saved, "aim check"
        else:
            guess = descent.get("reach")
            out["reach"] = _session["reach"] if guess is None else guess
            shift, why = (out["reach"], 0.0), "in case the arm stops short"
        if abs(shift[0]) > 1e-6 or abs(shift[1]) > 1e-6:
            gx, gy = np.array([gx, gy]) + G.arm_frame(gx, gy) @ np.array(shift, float)
            print(f"  aiming {shift[0]:+.1f} cm out" + (f", {shift[1]:+.1f} cm left" if abs(shift[1]) > 1e-6 else "")
                  + f" ({why}): x={gx:.1f}, y={gy:.1f}")
    body["x"], body["y"] = round(float(gx), 2), round(float(gy), 2)
    if h:
        body["object_height"] = round(h, 1)   # the arm starts feeling above it, and handles small and big differently
    r = requests.post(a.arm_url + "/api/grab", json=body, timeout=240).json()
    rep = r.get("report") or {}
    out.update(ok=bool(r.get("ok")), holding=bool(r.get("holding")), report=rep, error=r.get("error"))
    if rep.get("contact"):
        print(f"  contact stop at model height {rep['met_at_cm']} cm; after backoff {rep.get('closed_at_cm', '?')} cm")
    elif "contact" in rep:
        print(f"  descent ended at model height {rep.get('closed_at_cm', '?')} cm (planned {close:g} cm), "
              "without an early contact stop")
    if rep.get("jaws_gap_cm") is not None:
        print(f"  jaws closed to {rep['jaws_gap_cm']} cm")
    if rep.get("slipped"):
        print("  it had it, but lost it on the way up")
    if out["ok"] and not out["holding"]:
        out["why"] = why_missed(rep, t, close)
    log = requests.get(a.arm_url + "/status", timeout=10).json().get("log", "")
    print(("OK: " if r.get("ok") else "Failed: ") + (r.get("error") or log))
    return out


