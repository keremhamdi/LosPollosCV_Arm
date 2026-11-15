"""Medicine Buddy HTTP adapter. SAM-only; no Ollama or speech runtime required.

The trusted Buddy voice guard owns spoken consent. This service defaults to
preview, serializes jobs, and persists request IDs to prevent repeated motion.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

from reachy_target import resolve_target

SCHEMA = "medicine_buddy.arm.v1"
# Destinations Buddy may name -> measured destination entries in --destinations.
DESTINATIONS = {"bowl": "bowl", "handover": "handover", "hand": "handover", "me": "handover"}
ROUTES = {"/v1/dispense": "dispense", "/v1/fetch": "fetch", "/v1/move_object": "move_object", "/v1/locate": "locate", "/v1/check_taken": "check_taken"}


def validate_command(body, kind):
    if not isinstance(body, dict) or body.get("schema") != SCHEMA or body.get("message_type") != kind:
        raise ValueError("Wrong schema or message_type for this endpoint")
    rid = body.get("request_id")
    if not isinstance(rid, str) or str(uuid.UUID(rid)) != rid.lower():
        raise ValueError("request_id must be a hyphenated UUID")
    if type(body.get("dry_run")) is not bool or body.get("one_shot_detect") is not True:
        raise ValueError("dry_run must be a JSON boolean and one_shot_detect must be true")
    if kind in ("dispense", "check_taken"):
        med = body.get("medicine")
        if not isinstance(med, dict) or type(med.get("id")) is not int or med["id"] < 0:
            raise ValueError("medicine.id must be a nonnegative integer")
        name = med.get("name")
        if med.get("person") is not None and not isinstance(med["person"], str):
            raise ValueError("medicine.person must be a string or null")
    elif kind == "move_object":
        # Current Medicine Buddy main: {"object": {"label", "kind_hint"}, "destination", "actions"}.
        obj = body.get("object")
        if not isinstance(obj, dict):
            raise ValueError("move_object needs object: {label, kind_hint}")
        name = obj.get("label")
        if obj.get("kind_hint") is not None and not isinstance(obj["kind_hint"], str):
            raise ValueError("object.kind_hint must be a string or null")
        if body.get("actions", ["locate", "pick", "place"]) != ["locate", "pick", "place"]:
            raise ValueError("Unsupported move_object actions")
        destination = body.get("destination", "bowl")
        if not isinstance(destination, str) or len(destination) > 100:
            raise ValueError("destination must be a string of at most 100 characters")
    else:
        name = body.get("object")
    if not isinstance(name, str) or not name.strip() or len(name) > 500:
        raise ValueError("Provide a nonempty object/medicine name, at most 500 characters")
    if kind == "dispense":
        if body.get("actions") != ["locate", "pick_to_bowl", "verify_place"]:
            raise ValueError("Unsupported dispense actions")
        score = body.get("ocr_min_confidence")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("ocr_min_confidence must be a finite number in [0,1]")
        hint = body.get("slot_hint")
        if hint is not None and (type(hint) is not int or hint < 0):
            raise ValueError("slot_hint must be a nonnegative integer or null")
    if kind == "fetch" and body.get("deliver_to", "handover") != "handover":
        raise ValueError("fetch only supports the configured handover destination")
    return name


def result_for(body, **updates):
    out = {"schema": SCHEMA, "message_type": body["message_type"] + "_result",
           "request_id": body["request_id"], "ok": False, "final_state": "aborted",
           "slot_id": None, "label": None, "score": None, "place_verified": False,
           "detect_detail": "", "error": None, "motion_attempted": False}
    out.update(updates)
    return out


class Service:
    def __init__(self, items, *, pipeline=None, fake=False, enable_motion=False,
                 journal="config/local/reachy_requests.sqlite3"):
        self.items, self.pipeline = items, pipeline
        self.fake, self.enable_motion = fake, enable_motion
        self.lock, self.journal = threading.Lock(), Path(journal)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, digest TEXT NOT NULL, command TEXT NOT NULL, result TEXT, created TEXT NOT NULL, needs_review INTEGER NOT NULL DEFAULT 0)")
            if "needs_review" not in {row[1] for row in db.execute("PRAGMA table_info(requests)")}:
                db.execute("ALTER TABLE requests ADD COLUMN needs_review INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.journal, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def health(self):
        return {"ok": True, "schema": SCHEMA, "fake": self.fake, "enable_motion": self.enable_motion,
                "busy": self.lock.locked(), "vision_backend": "sam3", "vlm": "none",
                "consent_source": "trusted Medicine Buddy voice guard", "routes": list(ROUTES)}

    @contextmanager
    def exclusive_job(self):
        # Linux/macOS: held across reservation, work and result persistence. It
        # also prevents local recovery from retiring a job that is still running.
        with self.journal.with_suffix(".job.lock").open("a") as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

    def last(self):
        with self.db() as db:
            row = db.execute("SELECT command,result,created FROM requests ORDER BY rowid DESC LIMIT 1").fetchone()
        return {"request": json.loads(row[0]), "response": json.loads(row[1]) if row[1] else None, "created": row[2]} if row else {}

    def acknowledge_unknown(self, request_id):
        """Local operator recovery after checking the physical outcome. Never replay."""
        with self.exclusive_job(), self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT command,result,needs_review FROM requests WHERE id=?", (request_id.lower(),)).fetchone()
            if not row or (row[1] is not None and not row[2]):
                raise ValueError("No unresolved request with that ID")
            result = result_for(json.loads(row[0]), error="OUTCOME_UNKNOWN_ACKNOWLEDGED", motion_attempted=None,
                                detect_detail="Operator acknowledged the unknown outcome; this request will never replay",
                                previous_result=json.loads(row[1]) if row[1] else None)
            db.execute("UPDATE requests SET result=?,needs_review=0 WHERE id=?", (json.dumps(result), request_id.lower()))

    def handle(self, path, body):
        validate_command(body, ROUTES[path])
        try:
            with self.exclusive_job():
                return self._handle(path, body)
        except BlockingIOError:
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
            with self.db() as db:
                row = db.execute("SELECT digest,result FROM requests WHERE id=?", (body["request_id"].lower(),)).fetchone()
            if row:
                if row[0] != digest:
                    return result_for(body, error="REQUEST_ID_CONFLICT")
                return json.loads(row[1]) if row[1] else result_for(body, error="IN_PROGRESS_OR_UNKNOWN", motion_attempted=None)
            return result_for(body, error="BUSY_OR_UNKNOWN", detect_detail="Another job owns this arm service journal")

    def _handle(self, path, body):
        kind = ROUTES[path]
        name = validate_command(body, kind)
        canonical = json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        rid = body["request_id"].lower()
        # Durable reservation precedes model, network and motor work. A crashed
        # in-flight request is never replayed automatically after a restart.
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,result FROM requests WHERE id=?", (rid,)).fetchone()
            if row:
                if row[0] != digest:
                    return result_for(body, error="REQUEST_ID_CONFLICT", detect_detail="This request ID belongs to different content")
                return json.loads(row[1]) if row[1] else result_for(body, error="IN_PROGRESS_OR_UNKNOWN", motion_attempted=None, detect_detail="Do not issue a new motion request until the previous outcome is checked")
            pending = db.execute("SELECT id FROM requests WHERE result IS NULL OR needs_review=1 LIMIT 1").fetchone()
            if pending:
                return result_for(body, error="BUSY_OR_UNKNOWN", detect_detail="An earlier request is active or its outcome needs operator review")
            db.execute("INSERT INTO requests (id,digest,command,result,created) VALUES (?,?,?,NULL,?)", (rid, digest, canonical, datetime.now(timezone.utc).isoformat()))
        if not self.lock.acquire(blocking=False):
            result = result_for(body, error="BUSY", detect_detail="Another job is active; nothing started for this request")
        else:
            try:
                result = self.execute(body, name)
            except Exception as exc:
                result = result_for(body, final_state="failed", error="JOB_FAILED_OR_UNKNOWN", motion_attempted=None,
                                    requires_operator_review=True, detect_detail=f"{type(exc).__name__}: {exc}")
            finally:
                self.lock.release()
        if result.get("requires_operator_review"):
            result["detect_detail"] = "Outcome unverified; check the physical arm before another request. " + result["detect_detail"]
        encoded = json.dumps(result, allow_nan=False)
        with self.db() as db:
            db.execute("UPDATE requests SET result=?,needs_review=? WHERE id=?", (encoded, int(bool(result.get("requires_operator_review"))), rid))
        return result

    def execute(self, body, name):
        kind = body["message_type"]
        if kind == "check_taken":
            return {"schema": SCHEMA, "message_type": "check_taken_result", "request_id": body["request_id"],
                    "ok": True, "taken": None, "via": "unavailable",
                    "detail": "Camera evidence does not establish ingestion; ask the person by voice", "error": None}
        if kind == "move_object":
            # Buddy speaks `error` aloud for this route, so it carries the sentence
            # and `error_code` carries the machine code.
            wanted = " ".join(body.get("destination", "bowl").casefold().replace("the ", " ").split())
            if wanted not in DESTINATIONS:
                return result_for(body, error_code="UNSUPPORTED_DESTINATION", detect_detail="Nothing moved",
                                  error=f"I can only place things at: {', '.join(sorted(set(DESTINATIONS.values())))}")
            result = self.run_object(body, name, kind, DESTINATIONS[wanted])
            if result["error"]:
                result.update(error_code=result["error"], error=result["detect_detail"] or result["error"])
            return result
        return self.run_object(body, name, kind, "bowl" if kind == "dispense" else "handover")

    def run_object(self, body, name, kind, destination):
        target = resolve_target(name, self.items, medicine=kind == "dispense")
        if target["status"] != "RESOLVED":
            return result_for(body, error="UNSUPPORTED_OBJECT" if target["status"] == "REFUSED" else "NEEDS_CLARIFICATION",
                              detect_detail=target["reason"], resolution=target)
        dry_run = body["dry_run"] or not self.enable_motion or kind == "locate" or self.fake
        if self.fake:
            return result_for(body, error="FAKE_MODE", detect_detail="Text resolved only; no images were processed and nothing moved",
                              resolution=target, dry_run=True, preview_only=True)
        report = self.pipeline.run(body, target, dry_run=dry_run, destination=destination)
        placed = not dry_run and report.get("status") == "PLACED" and report.get("place_verified") is True
        selected = report.get("selection", {}).get("status") == "GO"
        detail = report.get("placement_detail") or report.get("error") or "; ".join(report.get("execution_blocks", [])) or report["status"]
        if dry_run:
            detail = "Preview only; nothing moved. " + detail
        return result_for(body, ok=placed if kind != "locate" else selected,
            final_state="placed" if placed else "aborted" if dry_run else "failed",
            label=body["medicine"]["name"] if kind == "dispense" and placed else None,
            score=report.get("ocr_score"), place_verified=placed, detect_detail=detail,
            error=None if placed or (kind == "locate" and selected) else "DRY_RUN" if dry_run else report.get("status", "FAILED"),
            motion_attempted=report.get("motion_attempted", False), dry_run=dry_run, preview_only=dry_run,
            requires_operator_review=report.get("motion_attempted") is True and report.get("status") in ("MOTION_RESULT_UNKNOWN", "PLACE_UNVERIFIED"),
            resolution=target, destination=destination, vision={k: report.get(k) for k in ("status", "object_present", "selection", "region", "execution_blocks", "output_dir")})


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_json(self, code, result):
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, service.health())
            elif self.path == "/debug/last" and self.client_address[0] in ("127.0.0.1", "::1"):
                self.send_json(200, service.last())
            else:
                self.send_json(404, {"ok": False, "error": "Not found"})

        def do_POST(self):
            if self.path not in ROUTES:
                return self.send_json(404, {"ok": False, "error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Provide a JSON body of 1..16384 bytes with Content-Length")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete body")
                body = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                result = service.handle(self.path, body)
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                return self.send_json(400, {"schema": SCHEMA, "ok": False, "error": str(exc)})
            self.send_json(200, result)

        def log_message(self, *_):
            pass  # Patient data stays in the local journal, not terminal access logs.
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--arm-url", default="http://localhost:8890")
    parser.add_argument("--items", default="items_v1.json")
    parser.add_argument("--workspace")
    parser.add_argument("--validation")
    parser.add_argument("--destinations")
    parser.add_argument("--image", help="saved stand image: preview only")
    parser.add_argument("--hand-image")
    parser.add_argument("--out", default="review_outputs/reachy")
    parser.add_argument("--journal", default="config/local/reachy_requests.sqlite3")
    parser.add_argument("--fake", action="store_true", help="text/protocol test only; no model or arm")
    parser.add_argument("--enable-motion", action="store_true", help="allow confirmed Buddy live requests after all gates")
    parser.add_argument("--acknowledge-unknown", metavar="REQUEST_ID", help="after checking the arm, retire an unresolved request without retrying; then exit")
    args = parser.parse_args()
    if args.acknowledge_unknown:
        Service({}, journal=args.journal).acknowledge_unknown(args.acknowledge_unknown)
        print("Unknown outcome acknowledged; no request replayed and no arm connection made.")
        return
    if args.enable_motion and (args.fake or args.image or not all((args.workspace, args.validation, args.destinations))):
        parser.error("Motion needs live cameras, workspace, validation and measured destinations")
    items = json.loads(Path(args.items).read_text())
    pipeline = None
    if not args.fake:
        from dispense_pipeline import Pipeline
        pipeline = Pipeline(arm_url=args.arm_url, items=args.items, workspace=args.workspace, validation=args.validation,
                            destinations=args.destinations, out=args.out, image=args.image, hand_image=args.hand_image)
    service = Service(items, pipeline=pipeline, fake=args.fake, enable_motion=args.enable_motion, journal=args.journal)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Reachy arm service at http://{args.host}:{server.server_port}; SAM-only; motion enabled={args.enable_motion}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
