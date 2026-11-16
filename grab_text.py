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
  standing-camera guidance is the default; add --hand to enable wrist-camera corrections

Needs the arm page running WITHOUT --read-only, follower (Self-controlled), Hold arm pressed.
The page's buttons keep working at the same time. Reading writing needs once: uv pip install easyocr
"""
import argparse
import difflib
import io

import numpy as np
import requests
from PIL import Image

import target_selector_v5 as ts


def base_pixel(mask):
    """Middle of where the object stands: the lowest fifth of its outline in the picture."""
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
    if not _view_opened:
        _view_opened = True
        try:
            subprocess.Popen(["xdg-open", latest], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


REACH_CM = (8.0, 37.5)   # the arm reaches objects this far from its base, roughly
BASE_Y_CM = 4.2          # the base's turning axis in table coordinates


def on_calibrated_table(a, u, v, img_w, margin=0.08):
    """True if pixel (u, v) is on the part of the table the calibration marks cover (plus a margin).
    Outside it the mapping to table cm is a wild guess: a thing in the background can land 'in reach'."""
    import cv2
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    pairs, size = st.get("pairs") or [], st.get("size") or img_w
    if len(pairs) < 4:
        return True
    k = img_w / size
    hull = cv2.convexHull(np.array([[p[0] * k, p[1] * k] for p in pairs], np.float32))
    return cv2.pointPolygonTest(hull, (float(u), float(v)), True) >= -margin * img_w


def in_reach(a, cand):
    """Table position of the candidate's base (via the page's calibration) and whether the arm can
    reach it. Things in the background or off the table map far away and are left out."""
    mask = np.asarray(cand["mask"])
    u, v = base_pixel(mask)
    if not on_calibrated_table(a, u, v, mask.shape[1]):
        return None, False
    try:
        xy = requests.post(a.arm_url + "/api/click", json={"u": u, "v": v}, timeout=10).json()
        d = float(np.hypot(xy["x"], xy["y"] - BASE_Y_CM))
    except (KeyError, ValueError, TypeError, requests.RequestException):
        return None, False
    return (xy["x"], xy["y"]), REACH_CM[0] <= d <= REACH_CM[1]


_camera = {"key": None, "geo": None}


def estimate_height(a, mask, img_size):
    """How tall the object is (cm), from where the camera sits (worked out from the calibration)
    and the top and bottom of its outline. None if it can't be worked out."""
    import height
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    pairs, size = st.get("pairs") or [], st.get("size") or img_size[0]
    if len(pairs) < 8:
        return None
    key = (len(pairs), size, tuple(pairs[0]), tuple(pairs[-1]))
    if _camera["key"] != key:
        scale = img_size[0] / size
        scaled = [[u * scale, v * scale, x, y] for u, v, x, y in pairs]
        try:
            _camera["geo"] = height.camera_from_pairs(scaled, img_size)
        except (ValueError, np.linalg.LinAlgError):
            _camera["geo"] = None
        _camera["key"] = key
    if _camera["geo"] is None:
        print("  (size estimate: couldn't work out where the fixed camera sits from the calibration)")
        return None
    K, R, _, C, up = _camera["geo"]

    def table_of_pixel(u, v):
        xy = requests.post(a.arm_url + "/api/click", json={"u": u, "v": v}, timeout=10).json()
        return xy["x"], xy["y"]
    try:
        h, width, center = height.object_height(K, R, C, up, np.asarray(mask).astype(bool), table_of_pixel)
    except (KeyError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
        print(f"  (size estimate failed: {e!r})")
        return None
    ok_w = np.isfinite(width) and 0.2 < width < 15
    if not (np.isfinite(h) and 0.5 < h < 30 and ok_w):
        print(f"  (size estimate didn't make sense: {h:.1f} cm tall, {width:.1f} cm wide; "
              + ("keeping only the width)" if ok_w else "ignoring it)"))
        return (None, width, center) if ok_w else None
    return h, width, center


FALLBACK_CLOSE_CM = 1.5   # where the jaws close when the object's height couldn't be worked out


def grab_heights(a, mask, img_size):
    """Where the jaws close and where they come from, for this object.
    --close given: always that. Otherwise about half the object's estimated height."""
    est = estimate_height(a, mask, img_size)
    if a.close is not None:
        return a.close, a.above, est
    if est is None or est[0] is None:
        return FALLBACK_CLOSE_CM, a.above, est           # height unknown: low enough for small things
    h = est[0]
    close = min(6.0, max(1.5, 0.5 * h))                 # halfway up, never lower than 1.5 cm
    return round(close, 1), max(a.above, round(h + 5, 1)), est


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


def find_and_grab(a, text):
    """Find `text` in the fixed camera (by look; if nothing looks like it, by writing) and grab it."""
    a.last_text = text
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    if st.get("pending"):
        print("The page is waiting for a calibration click: finish or Undo that first.")
        return
    if st.get("read_only"):
        print("The controller runs with --read-only: restart it without that to move the arm.")
        return
    img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
    res = detect_fixed(img, text)
    if res["candidates"]:
        near = []
        for c in res["candidates"]:
            xy, ok = in_reach(a, c)
            c["out_of_reach"] = not ok
            where = f"x={xy[0]:.0f}, y={xy[1]:.0f}" if xy else "?"
            print(f"  candidate {c['target_score']:.2f} at {where} cm: " + ("in reach" if ok else "out of reach, ignored"))
            if ok:
                near.append(c)
        if not near:
            show_view(img, res["candidates"], None, None, None, f"'{text}': found, but nothing within reach")
            print(f"Found '{text}', but nothing within the arm's reach. Not grabbing.")
            return
        best = max(near, key=lambda c: c["target_score"])   # the highest score among what the arm can reach
        u, v = base_pixel(np.asarray(best["mask"]))
        close, above, est = grab_heights(a, best["mask"], img.size)
        tall = f"about {est[0]:.1f} cm tall, " if est and est[0] else ""
        show_view(img, res["candidates"], best, u, v, f"'{text}': {tall}closing at {close} cm")
        return go(a, img, u, v, f"'{text}'", best["target_score"], close, above, est)
    print(f"Nothing looks like '{text}'. Looking for it written on something...")
    try:
        hit = find_word(img, text)
    except ImportError:
        print("To read writing, install the reader once: uv pip install easyocr")
        return
    if hit is None:
        show_view(img, [], None, None, None, f"'{text}': nothing found")
        print(f"Couldn't find '{text}' by look or by writing. Not grabbing.")
        return
    wu, wv, read, conf = hit
    print(f"Read '{read}' (confidence {conf:.2f}) at pixel {wu:.0f},{wv:.0f}")
    obj = object_at(img, wu, wv, ["medicine", "blister pack", "bottle", "box", "object"])
    if obj is None:
        show_view(img, [], None, None, None, f"'{text}': writing found (circle), no object around it", word_at=(wu, wv))
        print("Found the writing, but no object around it. Not grabbing.")
        return
    u, v = base_pixel(obj["mask"])
    close, above, est = grab_heights(a, obj["mask"], img.size)
    show_view(img, [], obj, u, v, f"'{text}' written on the green one, closing at {close} cm",
              score_key="sam_score", word_at=(wu, wv))
    return go(a, img, u, v, f"the thing with '{text}' on it", obj["sam_score"], close, above, est)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="grab this once and exit (no typing)")
    ap.add_argument("--word", help='grab the thing with this written on it, once, and exit')
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--close", type=float, help="always close the jaws this high (cm); default: half the object's estimated height")
    ap.add_argument("--put-height", type=float, default=5.0, help="let go this high above the table for put/bring, cm")
    ap.add_argument("--above", type=float, default=14.0, help="approach and lift height, cm")
    ap.add_argument("--side", action="store_true", help="grab from the side instead of the top")
    ap.add_argument("--bring", nargs=2, type=float, metavar=("X", "Y"), help="hand-over spot for 'bring', cm")
    wrist = ap.add_mutually_exclusive_group()
    wrist.add_argument("--hand", dest="no_hand", action="store_false",
                       help="explicitly enable the existing wrist-camera alignment (off by default)")
    wrist.add_argument("--no-hand", dest="no_hand", action="store_true",
                       help="use standing-camera guidance only (the default)")
    ap.set_defaults(no_hand=True)
    ap.add_argument("--reach-out", type=float, default=1.0, metavar="CM",
                    help="aim this many cm further from the base than measured, because the arm tends to "
                         "stop short (default 1; 0 = off)")
    ap.add_argument("--no-feel", action="store_true",
                    help="don't feel for the table on the way down (the old way: trust the table's height)")
    ap.add_argument("--dry", action="store_true", help="find things and say where, never move")
    a = ap.parse_args(argv)
    print('Pickup guidance: ' + ('standing camera only; wrist alignment OFF' if a.no_hand else
                                'standing camera + wrist alignment ON'))

    if a.text or a.word:
        return find_and_grab(a, a.text or a.word)
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


def goto(a, x, y, h):
    r = requests.post(a.arm_url + "/api/goto", json={"x": x, "y": y, "height": h}, timeout=120).json()
    if not r.get("ok"):
        raise RuntimeError(r.get("error") or "move failed")


def hand_calibrate(a, text):
    """One-time: where an object right under the jaws appears in the gripper camera, and how the
    picture shifts when the arm moves (pixels per cm). Saved to controller/calib/hand.json."""
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
    pts = {}
    # moving the arm back (-y) moves the object UP in the picture, away from the jaws, so it stays visible
    for name, dx, dy in (("centre", 0, 0), ("+x", 2, 0), ("-y", 0, -2)):
        goto(a, x + dx, y + dy, look)
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
    open(HAND_FILE, "w").write(json.dumps(data, indent=1))
    print(f"Saved {HAND_FILE}: the object should appear at pixel {pts['centre'][0]:.0f},{pts['centre'][1]:.0f}; "
          f"about {np.linalg.norm(J[:, 0]):.0f} px per cm.")
    goto(a, x, y, look + 4)


def hand_correct(a, text, x, y, obj_h=None):
    """Hover above (x, y), look with the gripper camera, shift until the object is where the jaws close.
    Returns the corrected (x, y), or the same if the gripper camera can't help."""
    import json
    import os
    if a.no_hand or not os.path.exists(HAND_FILE):
        return x, y
    cal = json.load(open(HAND_FILE))
    target, J = np.array(cal["target_px"]), np.array(cal["px_per_cm"])
    c0, c1 = np.linalg.norm(J[:, 0]), np.linalg.norm(J[:, 1])
    if abs(float(J[:, 0] @ J[:, 1])) / max(1e-9, c0 * c1) > 0.85:
        print("  gripper camera: its saved setup is broken (the two test moves point the same way). Run handcal again.")
        return x, y
    if "at" in cal:   # setup was done with the arm pointing one way; turn the table directions to match now
        J = J @ arm_frame(*cal["at"]) @ arm_frame(x, y).T
    look = (obj_h if obj_h else 3.0) + cal.get("look_above", LOOK_ABOVE)   # same distance above its top as in setup
    x0, y0 = x, y
    try:   # open jaws: closed ones block the gripper camera's view
        requests.post(a.arm_url + "/api/release", json={}, timeout=30)
    except requests.RequestException:
        pass
    # 1. find it: look straight above the fixed camera's aim; if it isn't in view, look around it
    found = None
    for k, (dx, dy) in enumerate(SEARCH_CM):
        lx, ly = x + dx, y + dy
        goto(a, lx, ly, look)
        p, img, best = find_in_hand(a, text, near=target, tries=2 if k == 0 else 1)
        if p is not None:
            found = (lx, ly, p, img, best)
            break
        if k == 0:
            print("  gripper camera: not in view yet, looking around...")
    if found is None:
        save_hand_view(img, None, None, target, 0)
        print("  gripper camera: couldn't find it, keeping the fixed camera's aim "
              "(last picture: pick_views/hand_latest_0.jpg)")
        return x0, y0
    # 2. perfect it: shift until it sits where the jaws close (up to 3 looks)
    x, y, p, img, best = found
    for step in range(3):
        if step:
            goto(a, x, y, look)
            p, img, best = find_in_hand(a, text, near=target, tries=2)
            if p is None:
                print("  gripper camera: lost it, using the last good aim")
                return x, y
        d = np.linalg.solve(J, target - np.array(p))       # how far to move (cm) to bring it under the jaws
        print(f"  gripper camera: off by {np.hypot(*d):.1f} cm ({np.hypot(*(target - np.array(p))):.0f} px)")
        save_hand_view(img, best, p, target, step)
        if step and np.hypot(*d) > last + 0.3:
            print("  gripper camera: getting worse, not trusting it, keeping the fixed camera's aim")
            return x0, y0
        last = np.hypot(*d)
        nx, ny = x + d[0], y + d[1]
        if np.hypot(nx - x0, ny - y0) > MAX_FIX_CM:
            print(f"  gripper camera wants more than {MAX_FIX_CM} cm from the fixed camera's aim: not trusting it")
            return x0, y0
        x, y = nx, ny
        if np.hypot(*d) < 0.3:
            break
    print(f"  gripper camera: aim moved by {np.hypot(x - x0, y - y0):.1f} cm in total")
    return x, y


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
SQUEEZE_CM = 1.0         # close the jaws this much narrower than the object: holds without crushing


def go(a, img, u, v, what, score, close=None, above=None, est=None):
    where = requests.post(a.arm_url + "/api/click", json={"u": u, "v": v, "size": img.width}, timeout=10).json()
    print(f"Found {what} (score {score:.2f}) at pixel {u:.0f},{v:.0f} "
          f"-> table x={where.get('x')}, y={where.get('y')} cm")
    close = close if close is not None else (a.close if a.close is not None else 3.0)
    above = above if above is not None else a.above
    h, width, center = est if est else (None, None, None)
    kind = ("small: ends a little low going down" if h < 6 else "big: holds the exact height going down") if h else "size unknown: treated as small"
    print((f"About {h:.1f} cm tall ({kind}): " if h else f"({kind}) ") + f"closing the jaws {close} cm above the table, coming from {above} cm")
    if a.dry:
        return
    body = {"u": u, "v": v, "size": img.width, "height": close, "above": above,
            "method": "side" if a.side else "top"}
    if not a.side and not a.no_feel:
        # Go down slowly at the end and stop at the first thing the gripper meets: the table under the
        # fingertips (small things) or the object's top under the palm (tall things). So neither the
        # table's exact height nor the object's estimated height has to be right any more.
        body["guard"] = True
        if a.close is None:
            body["feel_all_the_way"] = True
    if not a.side and where.get("x") is not None:
        gx, gy = float(where["x"]), float(where["y"])
        if not a.no_hand:
            try:
                gx, gy = hand_correct(a, a.last_text, gx, gy, h)
            except (RuntimeError, ValueError, requests.RequestException, np.linalg.LinAlgError) as e:
                print(f"  gripper camera skipped: {e}")
        if a.reach_out:
            # The arm tends to stop a little short, towards its base. Aim this far further out,
            # straight away from the base (--reach-out, cm).
            r = np.array([gx, gy - BASE_Y_CM])
            n = np.linalg.norm(r)
            if n > 1e-6:
                r = r / n
                gx, gy = gx + a.reach_out * r[0], gy + a.reach_out * r[1]
                print(f"  aiming {a.reach_out:g} cm further out: x={gx:.1f}, y={gy:.1f}")
        body.pop("u"); body.pop("v")
        body["x"], body["y"] = round(gx, 2), round(gy, 2)
    if h:
        body["object_height"] = round(h, 1)   # lets the arm handle small and big objects differently
    if width:   # the jaws now stop by themselves on whatever they meet, so no width guess is needed
        print(f"About {width:.1f} cm wide.")
    r = requests.post(a.arm_url + "/api/grab", json=body, timeout=240).json()
    rep = r.get("report") or {}
    if rep.get("contact"):
        print(f"  felt something {rep['met_at_cm']} cm above where the table was thought to be, closed there")
    elif "contact" in rep:
        print("  went all the way down without feeling anything")
    if rep.get("jaws_gap_cm") is not None:
        print(f"  jaws closed to {rep['jaws_gap_cm']} cm")
    if rep.get("slipped"):
        print("  it had it, but lost it on the way up")
    log = requests.get(a.arm_url + "/status", timeout=10).json().get("log", "")
    print(("OK: " if r.get("ok") else "Failed: ") + (r.get("error") or log))


if __name__ == "__main__":
    main()
