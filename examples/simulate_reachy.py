"""Simulated Reachy: the real Medicine Buddy server talking to this arm service.

Speech, cameras, SAM and motors are not involved; the arm stays in preview.

    python examples/simulate_reachy.py /path/to/LosPollosForTheElders
"""
import sys, os, json, tempfile, threading, urllib.request
from pathlib import Path
from unittest.mock import Mock
from http.server import ThreadingHTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(sys.argv[1]) / "apps" / "medicine_buddy"))
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): os.environ.pop(k, None)
from arm_service import Service, make_handler
tmp = Path(tempfile.mkdtemp())
items = {"paracetamol box": {"medicine_names": ["Paracetamol"], "form": "box"}, "water bottle": {"category": "bottle"}}
pipe = Mock(); pipe.run.return_value = {"status": "PREVIEW", "selection": {"status": "GO"}, "motion_attempted": False, "execution_blocks": []}
arm = Service(items, pipeline=pipe, journal=str(tmp/"arm.sqlite3"))          # motion NOT enabled
arm_srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(arm)); threading.Thread(target=arm_srv.serve_forever, daemon=True).start()
os.environ["MEDICINE_BUDDY_ARM_URL"] = f"http://127.0.0.1:{arm_srv.server_port}"
from medicine_buddy.api import start_server
from medicine_buddy.config import load_app_config
svc, srv, _ = start_server("127.0.0.1", 0, tmp/"buddy.db", load_app_config(dry_run=False), voice_guard=True)   # as their main.py
B = f"http://127.0.0.1:{srv.server_port}"
def buddy(path, body):
    r = urllib.request.Request(B+path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try: return json.load(urllib.request.urlopen(r, timeout=30))
    except urllib.error.HTTPError as e: return {"HTTP": e.code, **json.loads(e.read() or b"{}")}
def show(title, out):
    print(f"\n### {title}\n  Buddy says: {out.get('speak_hint') or out}")
def arm_saw():
    last = arm.last()
    print("  arm received:", json.dumps(last["request"]))
    r = last["response"]; print("  arm answered:", {k: r.get(k) for k in ("message_type","ok","final_state","error","error_code") if k in r}, "| target:", (r.get("resolution") or {}).get("target"))
show("tool move_object (not confirmed)", buddy("/arm/move_object", {"object_label": "water bottle", "confirmed": False}))
print("  arm received anything yet?", bool(arm.last()))
buddy("/voice/heard", {"text": "yes please"})
show("person says yes -> tool move_object confirmed", buddy("/arm/move_object", {"object_label": "water bottle", "confirmed": True})); arm_saw()
show("schedule Paracetamol", buddy("/schedules", {"name": "Paracetamol", "time": "08:00", "confirmed": True}))
buddy("/voice/heard", {"text": "yes add paracetamol at eight"})
show("schedule Paracetamol (confirmed)", buddy("/schedules", {"name": "Paracetamol", "time": "08:00", "confirmed": True}))
show("tool dispense_medicine (not confirmed)", buddy("/doses/run", {"medicine_name": "Paracetamol", "confirmed": False}))
buddy("/voice/heard", {"text": "yes"})
out = buddy("/doses/run", {"medicine_name": "Paracetamol", "confirmed": True}); show("person says yes -> dispense confirmed", out); arm_saw()
print("  dose result keys:", {k: out.get(k) for k in ("final_state","error","blocked","demo_mode") if k in out})
