"""Medicine Buddy -> arm: the most basic dispense. One request = locate, pick, put in the bowl, park.

  .venv/bin/python dispense_server.py                        # finds the bowl in the camera itself
  .venv/bin/python dispense_server.py --bowl -20 15          # or: bowl always at table x=-20, y=15 cm
  (Buddy's config: MEDICINE_BUDDY_ARM_URL=http://<this laptop's IP>:9090)

Needs the arm page running WITHOUT --read-only (python3 -m arm.click ... in controller/),
follower (Self-controlled), Hold arm pressed. Reading labels needs once: uv pip install easyocr

POST /v1/dispense   body: {"schema": "medicine_buddy.arm.v1", "message_type": "dispense",
                           "request_id": "...", "medicine": {"name": "Aspirin", ...},
                           "ocr_min_confidence": 0.65, "dry_run": false, ...}
     -> the dispense_result (response body of the same POST; it waits until the arm is done)
POST /v1/move_object  an everyday thing (Buddy's move_object tool):
                           {"message_type": "move_object", "object": {"label": "water bottle"},
                            "destination": "bowl", ...}   -> move_object_result
                      destination "bowl" (default) or "handover" (needs --handover X Y)
POST /v1/check_taken  always answers taken: null, so Buddy asks the person
GET  /health        is the server up, and which requests it takes
GET  /v1/last       the last request and result (for Family/debug)

  .venv/bin/python dispense_server.py --listen-only      # no arm, no camera: only show what
                                                         # Reachy sends (to test the connection)

Basic on purpose: one bowl position you give, no slots, place_verified is always false
(not checked yet), "actions" and "slot_hint" are accepted and ignored.

On this branch it picks the careful way (precise_grab.py: aim at the middle of where the object
stands, close at a height planned from the object instead of a fixed 5 cm, look again and retry
after a miss; docs/PRECISE_AIM.md). Add --hand to centre the jaws with the gripper camera.
  .venv/bin/python dispense_server.py --basic        # the pick as it was: fixed 5 cm, one grab
"""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import numpy as np
    import requests
    from PIL import Image

    import grab_text as G
    import precise_grab as P
    import target_selector_v5 as ts
except ImportError:   # fine for --listen-only, which needs no camera or arm packages
    np = requests = Image = G = P = ts = None

SCHEMA = "medicine_buddy.arm.v1"
LOCK = threading.Lock()          # one dispense at a time
LAST = {"request": None, "result": None}


def page(a, name, body=None):
    """Press one of the arm page's buttons; returns its answer (raises with the page's error)."""
    r = requests.post(a.arm_url + "/api/" + name, json=body or {}, timeout=180).json()
    if not r.get("ok"):
        raise RuntimeError(r.get("error") or f"{name} failed")
    return r


def find_bowl(a, img):
    """Where the bowl is on the table (x, y cm), found by SAM 3 in the same picture. None if not seen."""
    ts.VLM_BACKEND = "none"
    res = ts.decide(img, a.bowl_text, None)
    if not res["candidates"]:
        return None
    best = max(res["candidates"], key=lambda c: c["target_score"])
    ys, xs = np.nonzero(np.asarray(best["mask"]))
    u, v = float(xs.mean()), float(ys.mean())          # middle of the bowl as the camera sees it
    xy = page(a, "click", {"u": u, "v": v, "size": img.width})
    return xy["x"], xy["y"], best["target_score"]


def load_meds(path):
    """meds.txt: one line per medicine, "Aspirin = small bottle" (Buddy's name = what it looks like)."""
    meds = {}
    if path:
        for line in open(path):
            if "=" in line and not line.strip().startswith("#"):
                name, look = line.split("=", 1)
                meds[name.strip().lower()] = look.strip()
    return meds


def locate(a, name, min_conf):
    """Find the medicine in the fixed camera. Returns (u, v, label, score, how).
    A name in --meds: look for what it looks like. Any other name: look for something that looks
    like the name itself; without --meds (or with --read-labels) first try reading it on a label."""
    img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
    look = a.meds.get(name.lower())
    if look:
        ts.VLM_BACKEND = "none"
        res = ts.decide(img, look, None)
        if not res["candidates"]:
            return None
        best = max(res["candidates"], key=lambda c: c["target_score"])
        u, v = G.base_pixel(np.asarray(best["mask"]))
        return u, v, name.upper(), float(best["target_score"]), f"looks_like:{look}"
    hit = None
    if a.read_labels or not a.meds:   # reading labels is slow: skip it when a --meds list is used
        try:
            hit = G.find_word(img, name)
        except ImportError:
            hit = None
    if hit is not None and hit[3] >= min_conf:
        wu, wv, read, conf = hit
        obj = G.object_at(img, wu, wv, ["medicine", "medicine box", "blister pack", "bottle", "box", "object"])
        if obj is not None:
            u, v = G.base_pixel(obj["mask"])
            return u, v, read.upper(), float(conf), "label_read"
    ts.VLM_BACKEND = "none"
    res = ts.decide(img, name, None)
    if res["candidates"]:
        best = max(res["candidates"], key=lambda c: c["target_score"])
        u, v = G.base_pixel(np.asarray(best["mask"]))
        return u, v, name.upper(), float(best["target_score"]), "by_look"
    return None


LABELLED_THINGS = ["medicine", "medicine box", "blister pack", "bottle", "box", "object"]


def grab_settings(a):
    """precise_grab's settings (how it aims, goes down and tries again) from this server's options."""
    return P.options(arm_url=a.arm_url, close=getattr(a, "precise_close", None), above=a.above,
                     tries=getattr(a, "tries", 2), no_hand=not getattr(a, "hand", False),
                     no_feel=getattr(a, "no_feel", False), reach_out=getattr(a, "reach_out", None),
                     min_close=getattr(a, "min_close", P.LOW_CLOSE_CM), feel_small=getattr(a, "feel_small", False),
                     no_touch=getattr(a, "no_touch", False))


def locate_precise(a, kind, name, min_conf, near=None):
    """--precise: the same search as locate() / locate_object(), but giving what precise_grab grabs:
    (target, label, score, how), target = where the object stands on the table and its size.
    Only things the arm can reach are returned."""
    ga = grab_settings(a)
    img = P.frame(ga)
    tmap = P.table_map(ga, img.size)
    if tmap is None:
        raise RuntimeError("the fixed camera is not calibrated (the arm page needs 4 or more marks)")
    look = a.meds.get(name.lower()) if kind == "dispense" else None
    if look:
        t, _ = P.find_by_look(ga, img, tmap, look, near)
        return (t, name.upper(), t["score"], f"looks_like:{look}") if t else None
    if kind == "dispense" and (a.read_labels or not a.meds):   # reading labels is slow: skipped with a --meds list
        t = P.find_by_word(ga, img, tmap, name, min_conf, LABELLED_THINGS)
        if t is not None:
            return t, t["read"].upper(), t["read_confidence"], "label_read"
    t, _ = P.find_by_look(ga, img, tmap, name, near)
    return (t, name.upper() if kind == "dispense" else name, t["score"], "by_look") if t else None


NOT_FOR_THE_ARM = {"knife", "knives", "scissors", "blade", "razor", "syringe", "needle", "hot", "boiling",
                   "weapon", "gun", "person", "people"}


def locate_object(a, label):
    """Find an everyday thing in the fixed camera by its look. Returns (u, v, label, score, how)."""
    img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
    ts.VLM_BACKEND = "none"
    res = ts.decide(img, label, None)
    if not res["candidates"]:
        return None
    best = max(res["candidates"], key=lambda c: c["target_score"])
    u, v = G.base_pixel(np.asarray(best["mask"]))
    return u, v, label, float(best["target_score"]), "by_look"


def check_taken(req):
    """Whether the medicine was swallowed can't be seen from here: Buddy asks the person."""
    return {"schema": SCHEMA, "message_type": "check_taken_result", "request_id": req.get("request_id"),
            "ok": True, "taken": None, "via": "unavailable", "detail": "not checked by camera", "error": None}


def dispense(a, req, kind="dispense"):
    """One request from Buddy: a medicine (kind "dispense") or an everyday thing ("move_object")."""
    out = {"schema": SCHEMA, "message_type": kind + "_result", "request_id": req.get("request_id"),
           "ok": False, "final_state": "failed", "slot_id": None, "label": None, "score": None,
           "place_verified": False, "detect_detail": None, "error": None}
    if req.get("schema") != SCHEMA or req.get("message_type") != kind:
        out.update(final_state="aborted", error="unexpected schema or message_type")
        return out
    if kind == "dispense":
        name = ((req.get("medicine") or {}).get("name") or "").strip()
        place = "bowl"
    else:
        obj = req.get("object")
        name = " ".join(str((obj.get("label") if isinstance(obj, dict) else obj) or "").split())
        place = str(req.get("destination") or "bowl").lower().replace("the ", "").strip()
        if NOT_FOR_THE_ARM & set(name.lower().split()):
            out.update(final_state="aborted", error="I don't move sharp or hot things, or people")
            return out
        if place not in ("bowl", "handover") or (place == "handover" and not a.handover):
            out.update(final_state="aborted", error=f"I can't put things at the {place}; I can use the bowl"
                                                    + (" or the handover spot" if a.handover else ""))
            return out
    if not name:
        out.update(final_state="aborted", error="medicine.name is missing" if kind == "dispense" else "object.label is missing")
        return out
    if a.listen_only:
        out.update(final_state="aborted", label=name, detect_detail="listen-only: request received, nothing looked for or moved",
                   error="This is a connection test; the arm is not connected")
        return out
    if not LOCK.acquire(blocking=False):
        out.update(final_state="aborted", error="arm busy with another dispense")
        return out
    try:
        st = requests.get(a.arm_url + "/status", timeout=10).json()
        if st.get("read_only"):
            raise RuntimeError("arm controller is in read-only mode")
        precise = bool(getattr(a, "precise", False))
        min_conf = float(req.get("ocr_min_confidence", 0.65))
        if precise:
            found = locate_precise(a, kind, name, min_conf)
        else:
            found = (locate(a, name, min_conf) if kind == "dispense" else locate_object(a, name))
        if place == "handover":
            bowl = (a.handover[0], a.handover[1])
        elif a.bowl:
            bowl = (a.bowl[0], a.bowl[1])
        else:   # find the bowl now, before the arm moves into the picture
            img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
            b = find_bowl(a, img)
            if b is None:
                out.update(detect_detail="bowl_not_found", error=f"could not find the {a.bowl_text}")
                return out
            bowl = (b[0], b[1])
        if found is None:
            out.update(detect_detail="not_found", error=f"could not find {name}")
            return out
        if precise:
            target, label, score, how = found
            xy = {"x": round(target["x"], 1), "y": round(target["y"], 1)}
        else:
            u, v, label, score, how = found
            xy = page(a, "click", {"u": u, "v": v, "size": 1280})
        out.update(label=label, score=round(score, 3),
                   detect_detail=f"{how} at x={xy.get('x')}, y={xy.get('y')} cm")
        if req.get("dry_run"):
            out.update(ok=True, final_state="aborted", detect_detail=out["detect_detail"] + " (dry_run: no motion)")
            return out
        why = ""
        if precise:   # go down by feel; after a miss look again and grab again (--tries)
            def again(near):
                refound = locate_precise(a, kind, name, min_conf, near)
                return refound[0] if refound else None
            got = P.grab_with_retries(grab_settings(a), target, again)
            if not got["ok"]:
                raise RuntimeError(got["error"] or "the arm refused the grab")
            why = got["why"]
            if got["attempts"] > 1:
                out["detect_detail"] += f"; {got['attempts']} grabs"
        else:
            got = page(a, "grab", {"u": u, "v": v, "size": 1280, "height": a.close, "above": a.above, "method": "top"})
        if not got.get("holding"):
            page(a, "park")
            out.update(detect_detail=out["detect_detail"] + "; gripper closed on nothing",
                       error="pick failed: " + (why or "nothing in the gripper"))
            return out
        page(a, "place", {"x": bowl[0], "y": bowl[1], "height": a.bowl_height, "above": a.above})
        page(a, "park")
        out.update(ok=True, final_state="placed", detect_detail=out["detect_detail"] + f"; put in {'bowl' if place == 'bowl' else 'handover spot'} (not verified)")
        return out
    except Exception as e:   # anything the arm or camera refused: report it, never pretend
        out.update(error=f"{type(e).__name__}: {e}")
        return out
    finally:
        LOCK.release()


def make_handler(a):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, data):
            body = json.dumps(data, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/v1/last", "/debug/last"):
                return self._send(200, LAST)
            if self.path == "/health":
                return self._send(200, {"ok": True, "schema": SCHEMA, "listen_only": a.listen_only,
                                        "routes": ["/v1/dispense", "/v1/move_object", "/v1/check_taken"]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            kind = {"/v1/dispense": "dispense", "/v1/move_object": "move_object",
                    "/v1/check_taken": "check_taken"}.get(self.path)
            if kind is None:
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, {"error": "body is not JSON"})
            if not isinstance(req, dict):
                return self._send(400, {"error": "body is not a JSON object"})
            if kind == "check_taken":
                return self._send(200, check_taken(req))
            LAST["request"], LAST["result"] = req, None
            print(f"{kind}: {req.get('medicine') or req.get('object')} dry_run={req.get('dry_run')}", flush=True)
            res = dispense(a, req, kind)
            LAST["result"] = res
            print(f"  -> {res['final_state']} ok={res['ok']} {res['detect_detail'] or ''} {res['error'] or ''}", flush=True)
            self._send(200, res)

        def log_message(self, *args):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds", help='file with lines like "Aspirin = small bottle"')
    ap.add_argument("--read-labels", action="store_true",
                    help="with --meds: also try reading names on labels (slow)")
    ap.add_argument("--bowl-text", default="bowl", help='what the bowl looks like to SAM 3, e.g. "white bowl"')
    ap.add_argument("--bowl", nargs=2, type=float, metavar=("X", "Y"),
                    help="where the bowl stands, table x y in cm (read it off the arm page)")
    ap.add_argument("--handover", nargs=2, type=float, metavar=("X", "Y"),
                    help='hand-over spot, table x y in cm, for move_object with destination "handover"')
    ap.add_argument("--listen-only", action="store_true",
                    help="no arm, no camera: receive Reachy's requests, show them, answer 'not connected'")
    ap.add_argument("--bowl-height", type=float, default=8.0, help="let go this high above the table, cm")
    ap.add_argument("--close", type=float, default=None,
                    help="close the jaws this high above the table, cm (default: planned from the object's "
                         "height; with --basic: 5)")
    ap.add_argument("--basic", action="store_true",
                    help="pick as before this branch: closing at --close (5 cm), one grab. The options below then do nothing")
    ap.add_argument("--precise", action="store_true", help="(on by default on this branch; kept so old commands work)")
    ap.add_argument("--hand", action="store_true", help="centre the jaws with the gripper camera (default if it is set up)")
    ap.add_argument("--no-hand", action="store_true", help="never use the gripper camera")
    ap.add_argument("--tries", type=int, default=2, help="grabs per request (default 2)")
    ap.add_argument("--no-feel", action="store_true", help="no contact stop on the way down")
    ap.add_argument("--min-close", type=float, default=0.8, metavar="CM",
                    help="small and flat things may be closed on this low above the table (default 0.8)")
    ap.add_argument("--feel-small", action="store_true", help="use the arm's contact stop for small objects too")
    ap.add_argument("--no-touch", action="store_true", help="don't go down to touch the table for flat and very low things")
    ap.add_argument("--reach-out", type=float, default=None, metavar="CM",
                    help="aim this many cm further from the base than measured (default: aim check, else the retry's guess)")
    ap.add_argument("--above", type=float, default=14.0, help="approach / carry height, cm")
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from Buddy's laptop")
    ap.add_argument("--port", type=int, default=9090)
    a = ap.parse_args()
    if ts is None and not a.listen_only:
        ap.error("camera/arm packages are missing: pip install -r requirements.txt (or use --listen-only)")
    a.meds = load_meds(a.meds)
    a.precise = not a.basic
    if a.precise and G is not None and not a.no_hand:
        import os
        a.hand = a.hand or os.path.exists(G.HAND_FILE)    # the gripper camera centres the jaws whenever it is set up
    a.precise_close = a.close                 # the closing height is planned from the object unless --close is given
    if a.close is None:
        a.close = 5.0
    if a.precise and P is not None:
        P.OPEN_VIEWS = False                  # a server: save the pictures of what it found, don't pop them up
    if a.meds:
        print("Medicines:", ", ".join(f"{k} -> {v}" for k, v in a.meds.items()))
    where = f"bowl at {a.bowl[0]}, {a.bowl[1]}" if a.bowl else f"finds the '{a.bowl_text}' in the camera"
    if a.precise:
        where += "; careful picking" + (" with the gripper camera" if a.hand else "")
    else:
        where += "; BASIC picking"
    if a.listen_only:
        where = "LISTEN ONLY: nothing is looked for or moved"
    print(f"Medicine Buddy arm server on http://{a.host}:{a.port}  /v1/dispense  /v1/move_object  ({where})")
    ThreadingHTTPServer((a.host, a.port), make_handler(a)).serve_forever()


if __name__ == "__main__":
    main()
