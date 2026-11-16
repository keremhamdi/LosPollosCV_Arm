"""Type what to grab, the arm grabs it. Then drop it, put it somewhere, bring it, or rest.

  .venv/bin/python grab_text.py                    # then type, e.g.:
      small bottle        -> finds it (by its look, else by the word written on it) and grabs it
      Parol               -> works for writing too: grabs the thing with "Parol" on it
      drop                -> open the gripper where it is
      put -12 25          -> put it down at table x=-12, y=25 cm
      bring               -> put it down at the hand-over spot (--bring X Y, or "bring -20 15")
      rest                -> park
      handcal small bottle -> one-time gripper camera setup, using that object
      trim                -> straighten a crooked gripper (turn a joint until it points straight down)
      q                   -> quit
  .venv/bin/python grab_text.py --text "small bottle"     # one grab, then exit
  add --dry to only show what it found and where, without moving

How it gets the jaws around small things (details: docs/GRAB_ACCURACY.md):
  - it aims at the middle of where the object stands, not at the near edge of its outline,
    and straightens the wide-angle picture first (aim.py)
  - small things are outlined a second time on an enlarged cut-out, so the outline is sharp
  - the gripper camera then centres the jaws; this also allows for how much the gripper leans
    (run "handcal" once more after updating: it now measures that lean)
  - going down it feels for the table or the object's top, starting above the object
  - if the jaws close on nothing, it looks again and tries once more (--tries)

Needs the arm page running WITHOUT --read-only, follower (Self-controlled), Hold arm pressed.
The page's buttons keep working at the same time. Reading writing needs once: uv pip install easyocr
"""
import argparse
import difflib
import io

import numpy as np
import requests
from PIL import Image

import aim
import target_selector_v5 as ts


def base_pixel(mask):
    """The lowest fifth of the object's outline in the picture: the edge it stands on, nearest the
    camera. (Where to aim is footprint_of(); this is kept for the pictures and older callers.)"""
    ys, xs = np.nonzero(mask)
    low = ys >= ys.max() - 0.2 * (ys.max() - ys.min() + 1)
    return float(xs[low].mean()), float(ys[low].mean())


_reader = None


def find_word(img, word, scale=2.5, angles=(0, 30, -30, 60, -60, 90, -90, 15, -15)):
    """Where `word` is written in the picture: (centre u, v, what was read, confidence) or None.

    Labels lie at any angle on the table and are small in the picture, so: take the table part
    of the picture, enlarge it, and read it at several tilts. Tolerates small reading mistakes
    (e.g. "Paro1" for "Parol")."""
    import cv2
    import easyocr
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    reader = _reader
    arr = np.asarray(img)
    top = int(0.3 * arr.shape[0])                       # the table, not the room behind it
    big = cv2.resize(arr[top:], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = big.shape[:2]
    w_l = word.lower()
    best = None
    for ang in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        rot = cv2.warpAffine(big, M, (nw, nh), borderValue=(0, 0, 0))
        back = cv2.invertAffineTransform(M)
        for box, text, conf in reader.readtext(rot):
            t = "".join(ch for ch in text.lower() if ch.isalnum())
            if not t:
                continue
            sim = 1.0 if w_l in t else max(difflib.SequenceMatcher(None, w_l, t[i:i + len(w_l)]).ratio()
                                           for i in range(max(1, len(t) - len(w_l) + 1)))
            if sim < 0.75 or (best is not None and sim + conf <= best[0]):
                continue
            cx, cy = sum(p[0] for p in box) / 4, sum(p[1] for p in box) / 4
            bx, by = back @ np.array([cx, cy, 1.0])        # rotated -> enlarged picture
            best = (sim + conf, bx / scale, by / scale + top, text, conf, ang)
    if best is None:
        return None
    print(f"(read at a tilt of {best[5]} degrees)")
    return best[1:5]


def object_at(img, u, v, prompts):
    """SAM 3 outlines of anything matching the prompts; returns the one the point (u, v) lies on."""
    cands = ts.sam3_candidates(img, prompts)
    on = [c for c in cands if c["mask"][int(v), int(u)]]
    if not on:   # writing at the very edge of the outline: take the smallest box around the point
        on = [c for c in cands if c["box"][0] <= u <= c["box"][2] and c["box"][1] <= v <= c["box"][3]]
    return min(on, key=lambda c: c["mask"].sum()) if on else None


_view_opened = False
OPEN_VIEWS = True        # show the first picture in a viewer (a server sets this to False)


def show_view(img, cands, chosen, u, v, title, score_key="target_score", word_at=None):
    """Save what was found as a picture and show it: the chosen object's mask filled green,
    other candidates outlined red, a cross where the arm is sent. pick_views/latest.jpg is
    replaced every time (it opens once; most image viewers then update by themselves),
    and each request is also kept as its own file."""
    import cv2
    import os
    import subprocess
    import time
    global _view_opened
    vis = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    for c in cands:
        if c is chosen:
            continue
        m = np.asarray(c["mask"]).astype(np.uint8)
        col = (150, 150, 150) if c.get("out_of_reach") else (60, 60, 230)   # grey: out of reach
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)
        x0, y0 = int(c["box"][0]), int(c["box"][1])
        label = f"{c[score_key]:.2f}" + (" too far" if c.get("out_of_reach") else "")
        cv2.putText(vis, label, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if chosen is not None:
        m = np.asarray(chosen["mask"]).astype(bool)
        vis[m] = (0.45 * vis[m] + 0.55 * np.array([60, 210, 60])).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (40, 255, 40), 2)
        x0, y0 = int(chosen["box"][0]), int(chosen["box"][1])
        cv2.putText(vis, f"PICK {chosen[score_key]:.2f}", (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
    if u is not None:
        cv2.drawMarker(vis, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 0), cv2.MARKER_CROSS, 28, 1)
    if word_at is not None:
        cv2.circle(vis, (int(word_at[0]), int(word_at[1])), 10, (0, 200, 255), 2)
    for color, th in (((255, 255, 255), 4), ((0, 0, 0), 2)):
        cv2.putText(vis, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, th)
    os.makedirs("pick_views", exist_ok=True)
    cv2.imwrite(os.path.join("pick_views", time.strftime("%H%M%S") + ".jpg"), vis)
    latest = os.path.join("pick_views", "latest.jpg")
    cv2.imwrite(latest + ".tmp.jpg", vis)
    os.replace(latest + ".tmp.jpg", latest)
    print(f"Picture of what it found: {latest}")
    if OPEN_VIEWS and not _view_opened:
        _view_opened = True
        try:
            subprocess.Popen(["xdg-open", latest], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


REACH_CM = (8.0, 37.5)   # the arm reaches objects this far from its base, roughly
BASE_Y_CM = 4.2          # the base's turning axis in table coordinates


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
    u, v = base_pixel(mask)
    k = tmap.size / img_w
    if not tmap.covers((u * k, v * k), margin * tmap.size):
        return None, False
    fp = footprint_of(cand, tmap, img_w)
    if fp is None or not np.all(np.isfinite(fp["center"])):
        return None, False
    d = float(np.hypot(fp["center"][0], fp["center"][1] - BASE_Y_CM))
    return fp, REACH_CM[0] <= d <= REACH_CM[1]


FALLBACK_CLOSE_CM = 1.5   # where the jaws close when the object's height couldn't be worked out
FLAT_LOOK_CM = 1.0        # a flat thing is taken to be about this high (for where to look from)


def grab_heights(a, h):
    """Where the jaws close and where they come from, for an object about h cm tall (None = unknown).
    --close given: always that. Otherwise about half the object's height. (With the feel-for-it
    way down, the closing height is only used if nothing is felt.)"""
    if a.close is not None:
        return a.close, a.above
    if not h:
        return FALLBACK_CLOSE_CM, a.above                # height unknown: low enough for small things
    close = min(6.0, max(1.5, 0.5 * h))                 # halfway up, never lower than 1.5 cm
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


def detect_fixed(img, text):
    """SAM 3 on the fixed camera picture. If nothing shows up, look again more closely:
    zoom into the table (left and right half, 2x) and accept weaker matches, so small or far
    objects that are only a few pixels wide still get found. Masks always come back full size."""
    import cv2
    ts.VLM_BACKEND = "none"
    res = ts.decide(img, text, None)
    if res["candidates"]:
        return res
    print("  nothing at first glance, zooming in on the table...")
    W, H = img.size
    top = int(H * 0.35)                                   # the table part of the picture
    half = int(W * 0.575)
    found = []
    old = ts.SAM_SCORE_MIN
    ts.SAM_SCORE_MIN = 0.25
    try:
        for x0 in (0, W - half):
            box = (x0, top, x0 + half, H)
            crop = img.crop(box)
            big = crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC)
            r = ts.decide(big, text, None)
            for c in r["candidates"]:
                m = cv2.resize(np.asarray(c["mask"]).astype(np.uint8), crop.size, interpolation=cv2.INTER_NEAREST)
                full = np.zeros((H, W), bool)
                full[top:H, x0:x0 + half] = m.astype(bool)
                if not full.any():
                    continue
                c["mask"] = full
                b = c["box"]
                c["box"] = [b[0] / 2 + x0, b[1] / 2 + top, b[2] / 2 + x0, b[3] / 2 + top]
                found.append(c)
    finally:
        ts.SAM_SCORE_MIN = old
    kept = []                                             # the two halves overlap: drop repeats
    for c in sorted(found, key=lambda c: -c["target_score"]):
        if all(np.count_nonzero(c["mask"] & k["mask"]) < 0.5 * np.count_nonzero(c["mask"]) for k in kept):
            kept.append(c)
    return {"status": "ZOOMED", "candidates": kept, "chosen": None}


TRIM_FILE = "controller/calib/trim.json"
TRIM_SPOT = (0.0, 25.0, 10.0)   # x, y, height (cm) where the gripper should point straight down


def trim(a):
    """Straighten the gripper: the arm points it straight down, you turn joints a few degrees at a time
    until it really is straight, and those degrees are saved as a fix for every move after."""
    import json
    import os
    cur = json.load(open(TRIM_FILE)) if os.path.exists(TRIM_FILE) else {}
    print("Follower on (Self-controlled), Hold arm pressed. The gripper goes straight down above x=0, y=25.")
    input("Enter to move (Ctrl+C to cancel)... ")
    goto(a, *TRIM_SPOT)
    print("""Look at it from the base side and from the side. Type a joint and degrees to turn it, e.g.
   5 +3     turn joint 5 by +3 degrees      5 -3   back again
Try joints 7, 4, 5, 6 with +3: keep the one that makes it lean less, undo the others.
   save     when it points straight down      q   cancel (turns everything back)""")
    turned = {}
    while True:
        line = input("trim> ").strip().lower()
        if line in ("q", "quit"):
            for j, d in turned.items():
                if abs(d) > 1e-6:
                    requests.post(a.arm_url + "/api/nudge", json={"joint": j, "deg": -d}, timeout=30)
            print("Cancelled, nothing saved.")
            return
        if line == "save":
            break
        try:
            j, d = line.split()
            j, d = int(j), float(d)
        except ValueError:
            print('Like "5 +3", or "save", or "q".')
            continue
        r = requests.post(a.arm_url + "/api/nudge", json={"joint": j, "deg": d}, timeout=30).json()
        if not r.get("ok"):
            print("  " + (r.get("error") or "didn't turn"))
            continue
        turned[j] = turned.get(j, 0.0) + d
        print("  turned so far: " + ", ".join(f"joint {k} {v:+g}" for k, v in sorted(turned.items()) if v))
    for j, d in turned.items():
        if abs(d) > 1e-6:
            cur[str(j)] = round(cur.get(str(j), 0.0) - d, 2)     # the arm really sits that much further
    if True:
        # you said it's straight now: remember joint 7 (wrist roll) here. It's normally left wherever
        # the leader put it in mirroring; from now on every move puts it back here
        st = requests.get(a.arm_url + "/status", timeout=10).json()
        raw = (st.get("joints_raw") or {}).get("7")
        if raw is None:
            print("Couldn't read joint 7's position: restart the controller with the new click.py.")
            return
        cur["roll_raw"] = int(raw)
    os.makedirs(os.path.dirname(TRIM_FILE), exist_ok=True)
    open(TRIM_FILE, "w").write(json.dumps(cur, indent=1))
    print(f"Saved {TRIM_FILE}: {cur}. Checking: going straight down again with the fix...")
    goto(a, TRIM_SPOT[0], TRIM_SPOT[1], TRIM_SPOT[2] + 3)
    goto(a, *TRIM_SPOT)
    print("It should point straight down now. If not, run trim again.")
    if any(k.isdigit() for k in cur):
        print("Then redo handcal (and the board calibration if grabs are off): recorded with the crooked gripper.")
    else:
        print("Only the wrist roll's straight position was saved (no angle fixes). Redo handcal.")


SAME_OBJECT_CM = 10.0     # trying again: a match within this of the last aim is taken to be the same object


def options(**settings):
    """Settings for the functions here, as the command line would give them (for other programs)."""
    base = dict(arm_url="http://localhost:8890", close=None, above=14.0, side=False, no_hand=False,
                reach_out=None, tries=2, no_lens=False, no_zoom=False, no_feel=False, dry=False,
                put_height=5.0, bring=None)
    base.update(settings)
    return argparse.Namespace(**base)


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
    bu, bv = base_pixel(np.asarray(cand["mask"]))
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
    res = detect_fixed(img, text)
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
        hit = find_word(img, text)
    except ImportError:
        print("To read writing, install the reader once: uv pip install easyocr")
        return None
    if hit is None or hit[3] < min_conf:
        show_view(img, [], None, None, None, f"'{text}': nothing found")
        print(f"Couldn't find '{text}' by look or by writing. Not grabbing.")
        return None
    wu, wv, read, conf = hit
    print(f"Read '{read}' (confidence {conf:.2f}) at pixel {wu:.0f},{wv:.0f}")
    obj = object_at(img, wu, wv, prompts or WORD_OBJECTS)
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
        r = np.array([t["x"], t["y"] - BASE_Y_CM])
        d = float(np.linalg.norm(r))
        back = max(REACH_CM[0] + 4.0, d - 14.0)
        if d > 1e-6 and back < d:
            goto(a, r[0] / d * back, BASE_Y_CM + r[1] / d * back, above)
    except (RuntimeError, ValueError, requests.RequestException) as e:
        print(f"  (couldn't move the arm out of the view: {e})")


def find_and_grab(a, text, tries=None):
    """Find `text` and grab it. If the jaws close on nothing, look again and try again (--tries).
    Returns the last grab's result (see grab_target), or None if it was never found."""
    tries = max(1, int(tries if tries is not None else getattr(a, "tries", 2)))
    near, res = None, None
    for n in range(1, tries + 1):
        t = locate(a, text, near)
        if t is None:
            if n > 1:
                print("Can't find it any more after that miss. Stopping.")
            return res
        res = grab_target(a, t)
        res["attempts"] = n
        if a.dry or res["holding"] or not res["ok"] or n == tries:
            break
        print(f"Didn't get it: {res['why']}. Looking again (try {n + 1} of {tries})...")
        clear_view(a, t, res["above"])
        near = (t["x"], t["y"])
    return res


def action(a, name, body=None):
    """Press one of the page's buttons from here. Prints what the page says."""
    r = requests.post(a.arm_url + "/api/" + name, json=body or {}, timeout=180).json()
    log = requests.get(a.arm_url + "/status", timeout=10).json().get("log", "")
    print(("OK: " if r.get("ok") else "Failed: ") + (r.get("error") or log))


def put(a, x, y):
    if a.dry:
        print(f"(dry) would put it down at x={x}, y={y}")
        return
    action(a, "place", {"x": x, "y": y, "height": a.put_height, "above": a.above})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="grab this once and exit (no typing)")
    ap.add_argument("--word", help='grab the thing with this written on it, once, and exit')
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--close", type=float, help="always close the jaws this high (cm); default: go down until the "
                                                "gripper feels the table or the object's top, and close there")
    ap.add_argument("--put-height", type=float, default=5.0, help="let go this high above the table for put/bring, cm")
    ap.add_argument("--above", type=float, default=14.0, help="approach and lift height, cm")
    ap.add_argument("--side", action="store_true", help="grab from the side instead of the top")
    ap.add_argument("--bring", nargs=2, type=float, metavar=("X", "Y"), help="hand-over spot for 'bring', cm")
    ap.add_argument("--no-hand", action="store_true", help="don't use the gripper camera to correct the aim")
    ap.add_argument("--reach-out", type=float, default=None, metavar="CM",
                    help="aim this many cm further from the base than measured (negative: nearer). Default: 0, "
                         "or 1 with a gripper camera setup from before the lean correction")
    ap.add_argument("--tries", type=int, default=2,
                    help="if the jaws close on nothing, look again and try again, up to this many grabs (default 2)")
    ap.add_argument("--no-lens", action="store_true",
                    help="don't straighten the fixed camera's picture: map pixels exactly as the arm page does")
    ap.add_argument("--no-zoom", action="store_true",
                    help="don't take a second, enlarged look at small objects")
    ap.add_argument("--no-feel", action="store_true",
                    help="don't feel for the table on the way down (the old way: trust the table's height)")
    ap.add_argument("--dry", action="store_true", help="find things and say where, never move")
    a = ap.parse_args()

    if a.text or a.word:
        find_and_grab(a, a.text or a.word)
        return
    print("Type what to grab, or a command:" + __doc__.split("then type, e.g.:")[1].split("  .venv")[0].rstrip())
    while True:
        try:
            line = input("\nWhat now? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        words = line.lower().split()
        if not words:
            continue
        try:
            if words[0] in ("q", "quit", "exit"):
                return
            elif words[0] in ("drop", "release", "open"):
                print("(dry) would open the gripper") if a.dry else action(a, "release")
            elif words[0] == "trim":
                trim(a)
            elif words[0] == "handcal":
                hand_calibrate(a, " ".join(line.split()[1:]) or "small white cylinder")
            elif words[0] in ("rest", "park"):
                print("(dry) would park") if a.dry else action(a, "park")
            elif words[0] in ("put", "place") and len(words) == 3:
                put(a, float(words[1]), float(words[2]))
            elif words[0] == "bring":
                if len(words) == 3:
                    a.bring = [float(words[1]), float(words[2])]
                if not a.bring:
                    print('Where should I bring it? Type e.g. "bring -20 15" once (table x y in cm).')
                    continue
                put(a, *a.bring)
            else:
                find_and_grab(a, line)
        except ValueError:
            print('Numbers please, e.g. "put -12 25".')
        except requests.RequestException as e:
            print(f"Can't reach the arm page ({e.__class__.__name__}). Is the controller running?")


# ---------- gripper camera: correct the aim just before going down ----------

HAND_FILE = "controller/calib/hand.json"
LOOK_ABOVE = 13.0        # cm above the object's TOP where the arm stops to look: same view for any size
MAX_FIX_CM = 8.0         # never shift more than this on the gripper camera's word


def hand_frame(a):
    """The gripper camera's current picture (from the arm page)."""
    import base64
    obs = requests.get(a.arm_url + "/observation", timeout=10).json()
    f = (obs.get("frames") or {}).get("hand") or {}
    if not f.get("jpeg_base64"):
        raise RuntimeError("no gripper camera picture: start the controller with --hand-camera /dev/video6")
    return Image.open(io.BytesIO(base64.b64decode(f["jpeg_base64"]))).convert("RGB")


def find_in_hand(a, text, near=None, tries=3):
    """Pixel (u, v) of the object's middle in the gripper camera, or None.
    Takes up to 3 pictures. With `near` (where it should be), picks the matching blob closest to that,
    so a second similar object in view doesn't pull it away."""
    import time
    ts.VLM_BACKEND = "none"
    img = None
    for attempt in range(tries):
        time.sleep(0.8)                               # let the camera catch up after moving
        img = hand_frame(a)
        old = ts.SAM_SCORE_MIN
        ts.SAM_SCORE_MIN = 0.2 if attempt else old    # 2nd/3rd try: accept weaker matches
        try:
            res = ts.decide(img, text, None)
        finally:
            ts.SAM_SCORE_MIN = old
        cands = [c for c in res["candidates"] if np.asarray(c["mask"]).sum() >= 60]   # not specks
        if not cands:
            continue
        top = max(c["target_score"] for c in cands)
        good = [c for c in cands if c["target_score"] >= 0.5 * top]
        def centre(c):
            ys, xs = np.nonzero(np.asarray(c["mask"]))
            return np.array([xs.mean(), ys.mean()])
        if near is not None:
            best = min(good, key=lambda c: np.linalg.norm(centre(c) - np.asarray(near)))
        else:
            best = max(good, key=lambda c: c["target_score"])
        c = centre(best)
        return (float(c[0]), float(c[1])), img, best
    if near is not None and img is not None:
        # From straight above things look different (a standing cylinder is just a disc), so the
        # description may not match. Last try: any small object, the one nearest where it should be.
        res = ts.decide(img, HAND_FALLBACK[0], HAND_FALLBACK[1])
        n_px = img.width * img.height
        small = []
        for c in res["candidates"]:
            m = np.asarray(c["mask"]).astype(bool)
            if 30 < m.sum() < 0.03 * n_px:
                ys, xs = np.nonzero(m)
                small.append((np.hypot(xs.mean() - near[0], ys.mean() - near[1]), xs.mean(), ys.mean(), c))
        if small:
            dist, u, v, best = min(small, key=lambda t: t[0])
            if dist < 0.25 * img.width:
                print(f"  gripper camera: '{text}' not recognised from above; using the small object nearest the jaws")
                return (float(u), float(v)), img, best
    return None, img, None


HAND_JAW_PX = (640.0, 430.0)   # roughly where an object right under the open jaws shows in the gripper camera
HAND_FALLBACK = ("small white object", "small object")   # what to look for from above if the name fails


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
    D, b = float(cal.get("look_above", LOOK_ABOVE)), CAMERA_BACK_CM
    T = D * np.sin(t) / (D * np.cos(t) + b)
    return (T * (D * np.cos(t0) + b) - D * np.sin(t0)) / (np.cos(t0) + T * np.sin(t0))


def _lean_geometry(cal, J, x, y, tilt, frm):
    """Pixels the spot under the jaws moves when the lean goes from `frm` to `tilt` degrees, by geometry."""
    return J @ (arm_frame(x, y)[:, 0] * (_lean_cm(cal, tilt) - _lean_cm(cal, frm)))


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
    look = obj_h + LOOK_ABOVE
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
        p, img, best = find_in_hand(a, text, near=pts.get("centre", HAND_JAW_PX))
        if p is None:
            print(f"The gripper camera can't see '{text}' from there. Try a clearer object or description."); return
        pts[name] = np.array(p)
        save_hand_view(img, best, p, pts["centre"], f"cal_{name}")
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
    data = {"target_px": pts["centre"].tolist(), "px_per_cm": J.tolist(), "look_above": LOOK_ABOVE,
            "at": [x, y]}   # the gripper camera turns with the base: remember which way the arm pointed
    if tilt is None:
        print("  (this arm page doesn't say how far the gripper leans: update controller/arm/click.py "
              "to have the lean allowed for)")
    else:
        data["tilt"] = float(tilt)   # ... and how far the gripper leaned
        lean = measure_lean(a, text, x, y, look, float(tilt), data, pts["centre"])
        if lean:
            data["lean"] = lean
    open(HAND_FILE, "w").write(json.dumps(data, indent=1))
    print(f"Saved {HAND_FILE}: the object should appear at pixel {pts['centre'][0]:.0f},{pts['centre'][1]:.0f}; "
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
            p, img, best = find_in_hand(a, text, near=p_at + expect, tries=2)
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
            save_hand_view(img, best, p, centre, "cal_lean")
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


def hand_correct(a, text, x, y, obj_h=None):
    """Hover above (x, y), look with the gripper camera, shift until the object is where the jaws close.
    Returns (x, y, info): the corrected aim (or the same if the gripper camera can't help), and
    info = {"used": the aim comes from the gripper camera, "lean_aware": its setup knows the lean}."""
    import json
    import os
    info = {"used": False, "lean_aware": False}
    if a.no_hand or not os.path.exists(HAND_FILE):
        return x, y, info
    cal = json.load(open(HAND_FILE))
    target0, J0 = np.array(cal["target_px"], float), np.array(cal["px_per_cm"], float)
    c0, c1 = np.linalg.norm(J0[:, 0]), np.linalg.norm(J0[:, 1])
    if abs(float(J0[:, 0] @ J0[:, 1])) / max(1e-9, c0 * c1) > 0.85:
        print("  gripper camera: its saved setup is broken (the two test moves point the same way). Run handcal again.")
        return x, y, info
    cal_tilt = cal.get("tilt")
    info["lean_aware"] = cal_tilt is not None
    if cal_tilt is None and "handcal" not in _told:
        _told.add("handcal")
        print("  gripper camera: this setup is from before the lean correction. Run handcal once more "
              "for a better aim away from where it was set up.")
    look = (obj_h if obj_h else 3.0) + cal.get("look_above", LOOK_ABOVE)   # same distance above its top as in setup
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
        J = J0 @ arm_frame(*cal["at"]) @ arm_frame(px, py).T if "at" in cal else J0
        shift = lean_shift(cal, J, px, py, tilt)
        target = target0 + shift
        if np.linalg.norm(shift) > 1 and "lean" not in _told:
            _told.add("lean")
            print(f"  gripper camera: leaning {tilt:g} deg here ({cal_tilt:g} in its setup), allowed for")
        p, img, best = find_in_hand(a, text, near=target, tries=tries)
        return p, img, best, target, J

    # 1. find it: look straight above the fixed camera's aim; if it isn't in view, look around it
    found, img, target = None, None, target0
    for k, (dx, dy) in enumerate(SEARCH_CM):
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
        save_hand_view(img, None, None, target, 0)
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
        save_hand_view(img, best, p, target, step)
        if last is not None and np.hypot(*d) > last + 0.3:
            print("  gripper camera: getting worse, not trusting it, keeping the fixed camera's aim")
            return x0, y0, info
        last = np.hypot(*d)
        nx, ny = x + d[0], y + d[1]
        if np.hypot(nx - x0, ny - y0) > MAX_FIX_CM:
            print(f"  gripper camera wants more than {MAX_FIX_CM} cm from the fixed camera's aim: not trusting it")
            return x0, y0, info
        x, y = nx, ny
        if np.hypot(*d) < 0.3:
            break
    info["used"] = True
    print(f"  gripper camera: aim moved by {np.hypot(x - x0, y - y0):.1f} cm in total")
    return x, y, info


def arm_frame(x, y):
    """Columns: unit vectors pointing away from the base (radial) and sideways (tangential) at x, y."""
    r = np.array([x, y - BASE_Y_CM], float)
    r /= np.linalg.norm(r)
    return np.column_stack([r, [-r[1], r[0]]])


SEARCH_CM = [(0, 0), (0, -3), (0, 3), (-3, 0), (3, 0)]   # where to look if it isn't in view (cm from the aim)


def save_hand_view(img, best, p, target, step):
    import cv2
    import os
    if img is None:
        return
    vis = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR).copy()
    if best is not None:
        m = np.asarray(best["mask"]).astype(bool)
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([60, 210, 60])).astype(np.uint8)
    cv2.drawMarker(vis, (int(target[0]), int(target[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)       # where it should be
    if p is not None:
        cv2.drawMarker(vis, (int(p[0]), int(p[1])), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 25, 3)  # where it is
    else:
        cv2.putText(vis, "NOT FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    os.makedirs("pick_views", exist_ok=True)
    cv2.imwrite(os.path.join("pick_views", f"hand_latest_{step}.jpg"), vis)


GRIPPER_OPEN_CM = 5.1    # jaw gap when fully open (100 %)
OLD_SETUP_REACH_OUT = 1.0   # cm further out, only with a gripper camera setup that doesn't know the lean


def why_missed(rep, t):
    """In a few words, what most likely went wrong when the jaws closed on nothing."""
    h, met = t["height"], rep.get("met_at_cm")
    if rep.get("slipped"):
        return "it had it, but lost it on the way up"
    if rep.get("contact") and met is not None:
        low = h if h else (FLAT_LOOK_CM if t["flat"] else None)
        if low is not None and met > low + 2.0:
            return f"it stopped {met:.1f} cm up, above the object, as if it touched something there"
        if low is not None and low < 4.0 and met > max(1.2, 0.5 * low):
            return f"it stopped {met:.1f} cm up, on top of the object: a fingertip came down on it (aim a little off)"
        return "the jaws closed beside it (aim a little off)"
    if rep.get("contact") is False:
        return "it went all the way down without feeling the table (teach the table again if this keeps happening)"
    return "the jaws closed on nothing"


def grab_target(a, t):
    """Grab the object `t` (from make_target): centre the jaws with the gripper camera, go down, close, lift.
    Returns {"ok": the arm did the grab, "holding": it has something, "why": what went wrong if not,
             "report": what the arm page measured, "error": the page's refusal if any, "above": lift height}."""
    h = t["height"]
    close, above = grab_heights(a, h)
    look_h = h if h else (FLAT_LOOK_CM if t["flat"] else None)
    out = {"ok": True, "holding": False, "why": "", "report": {}, "error": None, "above": above}
    print(f"Found {t['what']} (score {t['score']:.2f}): {describe(t)}, standing at table x={t['x']:.1f}, y={t['y']:.1f} cm")
    feel = not a.side and not a.no_feel
    print(("Going down by feel" if feel else f"Closing the jaws {close} cm above the table") + f", coming from {above} cm")
    if a.dry:
        return out
    body = {"height": close, "above": above, "method": "side" if a.side else "top"}
    if feel:
        # Go down slowly at the end and stop at the first thing the gripper meets: the table under the
        # fingertips (small things) or the object's top under the palm (tall things). So neither the
        # table's exact height nor the object's estimated height has to be right.
        body["guard"] = True
        if a.close is None:
            body["feel_all_the_way"] = True
    gx, gy = t["x"], t["y"]
    if not a.side:
        hand = {"used": False, "lean_aware": False}
        try:
            gx, gy, hand = hand_correct(a, t["hand_text"], gx, gy, look_h)
        except (RuntimeError, ValueError, KeyError, requests.RequestException, np.linalg.LinAlgError) as e:
            print(f"  gripper camera skipped: {e}")
        reach_out = a.reach_out
        if reach_out is None:   # only an old gripper camera setup leaves the arm short (it doesn't know the lean)
            reach_out = OLD_SETUP_REACH_OUT if hand["used"] and not hand["lean_aware"] else 0.0
        if reach_out:
            # Aim this far further out, straight away from the base (--reach-out, cm).
            r = np.array([gx, gy - BASE_Y_CM])
            n = np.linalg.norm(r)
            if n > 1e-6:
                r = r / n
                gx, gy = gx + reach_out * r[0], gy + reach_out * r[1]
                print(f"  aiming {reach_out:g} cm further out: x={gx:.1f}, y={gy:.1f}")
    body["x"], body["y"] = round(float(gx), 2), round(float(gy), 2)
    if h:
        body["object_height"] = round(h, 1)   # the arm starts feeling above it, and handles small and big differently
    r = requests.post(a.arm_url + "/api/grab", json=body, timeout=240).json()
    rep = r.get("report") or {}
    out.update(ok=bool(r.get("ok")), holding=bool(r.get("holding")), report=rep, error=r.get("error"))
    if rep.get("contact"):
        print(f"  felt something {rep['met_at_cm']} cm above where the table was thought to be, closed there")
    elif "contact" in rep:
        print("  went all the way down without feeling anything")
    if rep.get("jaws_gap_cm") is not None:
        print(f"  jaws closed to {rep['jaws_gap_cm']} cm")
    if rep.get("slipped"):
        print("  it had it, but lost it on the way up")
    if out["ok"] and not out["holding"]:
        out["why"] = why_missed(rep, t)
    log = requests.get(a.arm_url + "/status", timeout=10).json().get("log", "")
    print(("OK: " if r.get("ok") else "Failed: ") + (r.get("error") or log))
    return out


if __name__ == "__main__":
    main()
