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
GET  /v1/last       the last request and result (for Family/debug)

Basic on purpose: one bowl position you give, no slots, place_verified is always false
(not checked yet), "actions" and "slot_hint" are accepted and ignored.
"""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
from PIL import Image

import grab_text as G
import target_selector_v5 as ts

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


def match_med(meds, name):
    """What does this medicine look like? Buddy's name doesn't have to be written exactly as in meds.txt:
    "Aspirin bottle", "aspirin 500mg" and "ASPIRIN" all find the "Aspirin" line. The longest matching
    name wins, so "Aspirin Plus" beats "Aspirin" when both are listed."""
    if not meds:
        return None
    n = " " + " ".join(name.lower().replace("-", " ").split()) + " "
    exact = meds.get(name.lower().strip())
    if exact:
        return exact
    hits = [(len(k), v) for k, v in meds.items() if (" " + k + " ") in n or k in name.lower().replace(" ", "")]
    return max(hits)[1] if hits else None


def locate(a, name, min_conf):
    """Find the medicine in the fixed camera. Returns (u, v, label, score, how).
    A name in --meds: look for what it looks like. Any other name: look for something that looks
    like the name itself; without --meds (or with --read-labels) first try reading it on a label."""
    img = Image.open(io.BytesIO(requests.get(a.arm_url + "/frame.jpg", timeout=10).content)).convert("RGB")
    look = match_med(a.meds, name)
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


def dispense(a, req):
    out = {"schema": SCHEMA, "message_type": "dispense_result", "request_id": req.get("request_id"),
           "ok": False, "final_state": "failed", "slot_id": None, "label": None, "score": None,
           "place_verified": False, "detect_detail": None, "error": None}
    if req.get("schema") != SCHEMA or req.get("message_type") != "dispense":
        out.update(final_state="aborted", error="unexpected schema or message_type")
        return out
    name = ((req.get("medicine") or {}).get("name") or "").strip()
    if not name:
        out.update(final_state="aborted", error="medicine.name is missing")
        return out
    if not LOCK.acquire(blocking=False):
        out.update(final_state="aborted", error="arm busy with another dispense")
        return out
    try:
        st = requests.get(a.arm_url + "/status", timeout=10).json()
        if st.get("read_only"):
            raise RuntimeError("arm controller is in read-only mode")
        found = locate(a, name, float(req.get("ocr_min_confidence", 0.65)))
        if a.bowl:
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
        u, v, label, score, how = found
        xy = page(a, "click", {"u": u, "v": v, "size": 1280})
        out.update(label=label, score=round(score, 3),
                   detect_detail=f"{how} at x={xy.get('x')}, y={xy.get('y')} cm")
        if req.get("dry_run"):
            out.update(ok=True, final_state="aborted", detect_detail=out["detect_detail"] + " (dry_run: no motion)")
            return out
        got = page(a, "grab", {"u": u, "v": v, "size": 1280, "height": a.close, "above": a.above, "method": "top"})
        if not got.get("holding"):
            page(a, "park")
            out.update(detect_detail=out["detect_detail"] + "; gripper closed on nothing",
                       error="pick failed: nothing in the gripper")
            return out
        page(a, "place", {"x": bowl[0], "y": bowl[1], "height": a.bowl_height, "above": a.above})
        page(a, "park")
        out.update(ok=True, final_state="placed", detect_detail=out["detect_detail"] + "; put in bowl (not verified)")
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
            if self.path == "/v1/last":
                return self._send(200, LAST)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/dispense":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, {"error": "body is not JSON"})
            LAST["request"], LAST["result"] = req, None
            print(f"dispense: {req.get('medicine')} dry_run={req.get('dry_run')}", flush=True)
            res = dispense(a, req)
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
    ap.add_argument("--bowl-height", type=float, default=8.0, help="let go this high above the table, cm")
    ap.add_argument("--close", type=float, default=5.0, help="close the jaws this high above the table, cm")
    ap.add_argument("--above", type=float, default=14.0, help="approach / carry height, cm")
    ap.add_argument("--arm-url", default="http://localhost:8890")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from Buddy's laptop")
    ap.add_argument("--port", type=int, default=9090)
    a = ap.parse_args()
    a.meds = load_meds(a.meds)
    if a.meds:
        print("Medicines:", ", ".join(f"{k} -> {v}" for k, v in a.meds.items()))
    where = f"bowl at {a.bowl[0]}, {a.bowl[1]}" if a.bowl else f"finds the '{a.bowl_text}' in the camera"
    print(f"Medicine Buddy arm server on http://{a.host}:{a.port}/v1/dispense  ({where})")
    ThreadingHTTPServer((a.host, a.port), make_handler(a)).serve_forever()


if __name__ == "__main__":
    main()
