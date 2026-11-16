"""Type what to grab, the arm grabs it. Then drop it, put it somewhere, bring it, or rest.

  .venv/bin/python grab_text.py                    # then type, e.g.:
      small bottle        -> finds it (by its look, else by the word written on it) and grabs it
      Parol               -> works for writing too: grabs the thing with "Parol" on it
      drop                -> open the gripper where it is
      put -12 25          -> put it down at table x=-12, y=25 cm
      bring               -> put it down at the hand-over spot (--bring X Y, or "bring -20 15")
      rest                -> park
      q                   -> quit
  .venv/bin/python grab_text.py --text "small bottle"     # one grab, then exit
  add --dry to only show what it found and where, without moving

Needs the arm page running WITHOUT --read-only, follower (Self-controlled), Hold arm pressed.
The page's buttons keep working at the same time. Reading writing needs once: uv pip install easyocr
"""
import argparse
import math
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


def find_and_grab(a, text):
    """Find `text` in the fixed camera (by look; if nothing looks like it, by writing) and grab it."""
    st = requests.get(a.arm_url + "/status", timeout=10).json()
    if st.get("pending"):
        print("The page is waiting for a calibration click: finish or Undo that first.")
        return
    if st.get("read_only"):
        print("The controller runs with --read-only: restart it without that to move the arm.")
        return
    img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
    ts.VLM_BACKEND = "none"
    res = ts.decide(img, text, None)
    if res["candidates"]:
        scores = ", ".join(f"{c['target_score']:.2f}" for c in res["candidates"])
        best = max(res["candidates"], key=lambda c: c["target_score"])   # always the highest score
        print(f"Looks like '{text}': [{scores}], taking the highest.")
        u, v = base_pixel(np.asarray(best["mask"]))
        return go(a, img, u, v, f"'{text}'", best["target_score"])
    print(f"Nothing looks like '{text}'. Looking for it written on something...")
    try:
        hit = find_word(img, text)
    except ImportError:
        print("To read writing, install the reader once: uv pip install easyocr")
        return
    if hit is None:
        print(f"Couldn't find '{text}' by look or by writing. Not grabbing.")
        return
    wu, wv, read, conf = hit
    print(f"Read '{read}' (confidence {conf:.2f}) at pixel {wu:.0f},{wv:.0f}")
    obj = object_at(img, wu, wv, ["medicine", "blister pack", "bottle", "box", "object"])
    if obj is None:
        print("Found the writing, but no object around it. Not grabbing.")
        return
    u, v = base_pixel(obj["mask"])
    return go(a, img, u, v, f"the thing with '{text}' on it", obj["sam_score"])


def action(a, name, body=None):
    """Press one of the page's buttons from here. Prints what the page says."""
    r = requests.post(a.arm_url + "/api/" + name, json=body or {}, timeout=180).json()
    log = requests.get(a.arm_url + "/status", timeout=10).json().get("log", "")
    print(("OK: " if r.get("ok") else "Failed: ") + (r.get("error") or log))


def put(a, x, y):
    if a.dry:
        print(f"(dry) would put it down at x={x}, y={y}")
        return
    action(a, "place", {"x": x, "y": y, "height": a.close, "above": a.above})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="grab this once and exit (no typing)")
    ap.add_argument("--word", help='grab the thing with this written on it, once, and exit')
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--close", type=float, default=5.0, help="cm above the table where the jaws close / let go")
    ap.add_argument("--above", type=float, default=14.0, help="approach and lift height, cm")
    ap.add_argument("--no-feel", action="store_true",
                    help="don't feel for the table on the way down; always close at --close instead")
    ap.add_argument("--reach-out", type=float, default=1.0, metavar="CM",
                    help="aim this many cm further from the base than measured, because the arm tends to "
                         "stop short (default 1; 0 = off)")
    ap.add_argument("--side", action="store_true", help="grab from the side instead of the top")
    ap.add_argument("--bring", nargs=2, type=float, metavar=("X", "Y"), help="hand-over spot for 'bring', cm")
    ap.add_argument("--dry", action="store_true", help="find things and say where, never move")
    a = ap.parse_args()

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


BASE_Y_CM = 4.2   # the base's turning axis in table coordinates


def go(a, img, u, v, what, score):
    where = requests.post(a.arm_url + "/api/click", json={"u": u, "v": v, "size": img.width}, timeout=10).json()
    print(f"Found {what} (score {score:.2f}) at pixel {u:.0f},{v:.0f} "
          f"-> table x={where.get('x')}, y={where.get('y')} cm")
    if a.dry:
        return
    body = {"u": u, "v": v, "size": img.width, "height": a.close, "above": a.above,
            "method": "side" if a.side else "top"}
    if not a.side and not a.no_feel:
        # Dynamic height: instead of always closing at --close, the arm goes down slowly at the end and
        # stops at the FIRST thing the gripper meets - the table under the fingertips (small objects:
        # grabbed right at their base) or the object's top under the palm (tall ones: grabbed around the
        # body). So each object gets its own closing height, with nothing to measure or guess.
        body["guard"] = True
        body["feel_all_the_way"] = True
    if a.reach_out:
        # the arm tends to stop a little short, towards its base: aim this far further out
        x, y = where.get("x"), where.get("y")
        if x is not None:
            r_ = math.hypot(x, y - BASE_Y_CM)
            if r_ > 1e-6:
                body.pop("u"); body.pop("v")
                body["x"] = round(x + a.reach_out * x / r_, 2)
                body["y"] = round(y + a.reach_out * (y - BASE_Y_CM) / r_, 2)
                print(f"  aiming {a.reach_out:g} cm further out: x={body['x']}, y={body['y']}")
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
