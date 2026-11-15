"""Basic backend for arm_service.py: the pick that already works on the real arm.

Find the thing in the fixed camera, grab it from the top, put it down at a spot
you give, park. It presses the arm page's own buttons (click, grab, place, park),
so the page must run WITHOUT --read-only with the follower on (Self-controlled).

    python arm_service.py --basic --bowl -20 15 --host 0.0.0.0              # preview
    python arm_service.py --basic --bowl -20 15 --host 0.0.0.0 --enable-motion

Basic on purpose: no workspace, calibration-validation or item profile is needed,
and the placement is not checked by camera afterwards. The result says so.
"""
import io

import numpy as np
import requests
from PIL import Image

import grab_text as G
import target_selector_v5 as ts

# Broad outlines SAM 3 may draw around a label that was read.
AROUND_LABEL = ["medicine", "medicine box", "blister pack", "bottle", "box", "object"]


class BasicPipeline:
    def __init__(self, *, arm_url, places, close=5.0, above=14.0, place_height=8.0):
        self.arm_url, self.places = arm_url.rstrip("/"), places
        self.close, self.above, self.place_height = close, above, place_height

    def page(self, name, body=None):
        """Press one of the arm page's buttons; raises with the page's own error."""
        reply = requests.post(f"{self.arm_url}/api/{name}", json=body or {}, timeout=180).json()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or f"{name} failed")
        return reply

    def locate(self, command, target):
        """(u, v, score, how) of the thing in the fixed camera, or None.

        A medicine is looked for by the name written on it first; anything else,
        and a medicine whose name could not be read, by its look."""
        frame = requests.get(f"{self.arm_url}/frame.jpg", timeout=10).content
        img = Image.open(io.BytesIO(frame)).convert("RGB")
        if command["message_type"] == "dispense":
            try:
                hit = G.find_word(img, command["medicine"]["name"])
            except ImportError:   # easyocr not installed: fall back to its look
                hit = None
            if hit is not None and hit[3] >= command["ocr_min_confidence"]:
                found = G.object_at(img, hit[0], hit[1], AROUND_LABEL)
                if found is not None:
                    return (*G.base_pixel(found["mask"]), float(hit[3]), "name read on it", img.width)
        ts.VLM_BACKEND = "none"
        extra = target.get("sam_prompts") or []
        result = ts.decide(img, target["target"], target.get("category"), *([extra] if extra else []))
        if not result["candidates"]:
            return None
        best = max(result["candidates"], key=lambda c: c["target_score"])
        if best["target_score"] <= 0:   # only the broad category matched, not the thing asked for
            return None
        return (*G.base_pixel(np.asarray(best["mask"])), float(best["target_score"]), "by its look", img.width)

    def run(self, command, target, *, dry_run, destination=None):
        report = {"status": "FAILED", "selection": {"status": "NOT_FOUND"}, "motion_attempted": False,
                  "place_verified": False, "execution_blocks": []}
        spot = self.places.get(destination)
        try:
            status = requests.get(f"{self.arm_url}/status", timeout=10).json()
            if status.get("pending"):
                raise RuntimeError("The arm page is waiting for a calibration click")
            found = self.locate(command, target)
            if found is None:
                report.update(status="NOT_FOUND", error=f"I could not find the {target['target']}")
                return report
            u, v, score, how, size = found
            where = self.page("click", {"u": u, "v": v, "size": size})
            seen = f"Found {how} at x={where.get('x')}, y={where.get('y')} cm"
            report.update(selection={"status": "GO"}, object_present=True, placement_detail=seen)
            if command["message_type"] == "dispense":
                report["ocr_score"] = round(score, 3) if how == "name read on it" else None
            if dry_run:
                report["status"] = "PREVIEW"
                return report
            if spot is None:
                raise RuntimeError(f"No table position was given for the {destination}")
            if status.get("read_only"):
                raise RuntimeError("The arm page runs with --read-only")
            report["motion_attempted"] = True
            grabbed = self.page("grab", {"u": u, "v": v, "size": size, "height": self.close,
                                         "above": self.above, "method": "top"})
            if not grabbed.get("holding"):
                self.page("park")
                report.pop("placement_detail")
                report.update(status="PICK_FAILED", error=seen + ", but the gripper closed on nothing")
                return report
            self.page("place", {"x": spot[0], "y": spot[1], "height": self.place_height, "above": self.above})
            self.page("park")
            report.update(status="PLACED", released=True,
                          placement_detail=seen + f"; put at the {destination} (not checked by camera)")
        except (RuntimeError, requests.RequestException, ValueError, KeyError) as exc:
            report.pop("placement_detail", None)
            report["error"] = str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
        return report
