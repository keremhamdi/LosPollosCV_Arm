"""Reachy adapter for the existing two-camera pipeline and measured destinations.

No hardware is accessed until run(). Placement verification is conservative 2D
evidence, not proof of ingestion, 3D containment, or medical identity.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

import calibration_validation
import text_pick
from reachy_target import normalize
from selection_policy import select_candidates, workspace_polygon


def label_score(tsv, expected):
    """Exact token sequence on one OCR line; minimum word confidence / 100."""
    lines = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        score = float(row["conf"]) / 100
        if not np.isfinite(score) or not 0 <= score <= 1:
            continue
        key = tuple(row.get(k) for k in ("page_num", "block_num", "par_num", "line_num"))
        lines.setdefault(key, []).extend((word, score) for word in normalize(row["text"]).split())
    wanted = normalize(expected).split()
    found = []
    for line in lines.values():
        for i in range(len(line) - len(wanted) + 1):
            chunk = line[i:i+len(wanted)]
            if wanted and [word for word, _ in chunk] == wanted:
                found.append(min(score for _, score in chunk))
    return max(found, default=0.)


def read_label(image, mask, expected):
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("OCR_UNAVAILABLE: install Tesseract for live medicine label verification")
    matte = Image.fromarray(np.asarray(mask, np.uint8) * 255)
    crop = Image.composite(image, Image.new("RGB", image.size, "white"), matte).crop(matte.getbbox())
    stream = io.BytesIO()
    crop.save(stream, format="PNG")
    result = subprocess.run([executable, "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"],
                            input=stream.getvalue(), capture_output=True, timeout=15, check=True)
    return label_score(result.stdout.decode("utf-8"), expected)


def destination_mask(config, image_size, camera_id):
    polygon = workspace_polygon(config, image_size, camera_id)
    roi = np.zeros((image_size[1], image_size[0]), np.uint8)
    cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 1)
    return roi.astype(bool)


def placement_evidence(result, roi, source_mask):
    if result["status"] != "GO":
        return False, "No unique eligible target visible at the destination"
    mask = result["chosen"]["mask"]
    area = max(1, int(mask.sum()))
    if np.count_nonzero(mask & roi) / area < .9:
        return False, "Target is not sufficiently inside the configured destination image region"
    if np.count_nonzero(mask & source_mask) / area > .1:
        return False, "Target still overlaps its original location"
    return True, "target_visible_in_destination_region; 2D evidence only"


class Pipeline:
    def __init__(self, *, arm_url, items, workspace=None, validation=None, destinations=None,
                 out="review_outputs/reachy", image=None, hand_image=None, ocr=read_label):
        self.arm_url, self.items = arm_url, items
        self.workspace, self.validation, self.destinations = workspace, validation, destinations
        self.out, self.image, self.hand_image, self.ocr = out, image, hand_image, ocr

    def run(self, command, target, *, dry_run, destination=None):
        kind = command["message_type"]
        destination = destination or ("bowl" if kind == "dispense" else "handover")
        prepared = {}
        extra = target.get("sam_prompts") or None

        def before(report, images, results, status, metadata, bridge):
            profile = json.loads(Path(self.items).read_text()).get(target["target"], {})
            if profile.get("fetch_allowed") is not True:
                raise ValueError("Item profile must explicitly enable fetch_allowed after physical assessment")
            config = json.loads(Path(self.destinations).read_text())[destination] if self.destinations else None
            if not config or config.get("measurements_verified") is not True:
                raise ValueError(f"Measured {destination} destination configuration required before pickup")
            roi = destination_mask(config, images["stand"].size, metadata["stand"]["camera_id"])
            point = config.get("target_px")
            if (not isinstance(point, list) or len(point) != 2 or not np.isfinite(point).all() or
                not 0 <= point[0] < images["stand"].width or not 0 <= point[1] < images["stand"].height or
                not roi[int(point[1]), int(point[0])]):
                raise ValueError("Destination target_px must lie inside its image region")
            evidence = calibration_validation.validate(json.loads(Path(self.validation).read_text()), status,
                        status.get("table_calibration"), images["stand"].size, metadata["stand"]["camera_id"], point)
            if not evidence["passed"]:
                raise ValueError("Destination calibration: " + "; ".join(evidence["errors"]))
            # An existing matching object would make a later image check misleading.
            if any(np.any(c["mask"] & roi) for c in results["stand"]["candidates"]):
                raise ValueError("Destination already overlaps a target proposal; clear it before a new request")
            calib = text_pick.pick.Calib(status["pairs"], status["size"], images["stand"].height / images["stand"].width)
            scale = status["size"] / images["stand"].width
            x, y = calib.to_table(point[0]*scale, point[1]*scale)
            body = {"x": x, "y": y, "height": config["release_height_cm"], "above": config["above_cm"]}
            if body["above"] < report["approach"]["approach_height_cm"]:
                raise ValueError("Destination approach must be at least as high as the pickup approach")
            if kind == "dispense":
                score = self.ocr(images["stand"], results["stand"]["chosen"]["mask"], command["medicine"]["name"])
                if type(score) not in (float, int) or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid standing-camera OCR score")
                report.update(ocr_score=score, ocr_required_view="stand")
                if score < max(.65, command["ocr_min_confidence"]):
                    raise ValueError("LABEL_UNVERIFIED: exact medicine label must pass OCR on the stand-selected object")
                # A wrist match may be another instance. Record supplementary
                # evidence; it can neither replace nor veto the stand label check.
                report["hand_ocr"] = {"status": "NO_UNIQUE_TARGET", "required": False}
                if results.get("hand", {}).get("status") == "GO":
                    try:
                        hand_score = self.ocr(images["hand"], results["hand"]["chosen"]["mask"], command["medicine"]["name"])
                        if type(hand_score) not in (float, int) or not np.isfinite(hand_score) or not 0 <= hand_score <= 1:
                            raise ValueError("Invalid supplementary OCR score")
                        report["hand_ocr"].update(status="OBSERVED", score=hand_score)
                    except (ValueError, OSError, subprocess.SubprocessError) as exc:
                        report["hand_ocr"].update(status="UNAVAILABLE", error=str(exc))
            bridge.place(body, preflight=True)  # IK only; controller reports no motion.
            prepared.update(body=body, roi=roi, config=config)
            report["destination"] = {"name": destination, "request": body, "preflight_passed": True}

        def after(report, images, results, status, metadata, bridge):
            current_status = bridge.get("/status")
            if any(current_status.get(k) != status.get(k) for k in ("cam", "pairs", "size", "table_calibration")):
                raise ValueError("Calibration changed after pickup; placement was not attempted")
            report["placement_attempted"] = True
            text_pick.save_report(Path(report["output_dir"]), report)
            report["place_controller_result"] = bridge.place(prepared["body"])
            # Require frames received after the completed release/retreat response.
            _, _, first = text_pick.read_live(bridge)
            deadline = time.monotonic() + 3
            while True:
                fresh, fresh_status, meta = text_pick.read_live(bridge)
                if all(meta[r]["sequence"] > first[r]["sequence"] for r in ("stand", "hand")):
                    break
                if time.monotonic() >= deadline:
                    raise ValueError("No new camera pair after release")
                time.sleep(.05)
            if any(meta[r]["camera_id"] != metadata[r]["camera_id"] for r in meta):
                raise ValueError("Camera identity changed after release")
            if any(fresh_status.get(k) != status.get(k) for k in ("cam", "pairs", "size", "table_calibration")):
                raise ValueError("Calibration changed during release")
            checked = select_candidates(text_pick.ts.decide(fresh["stand"], target["target"], target["category"], *([extra] if extra else [])),
                      fresh["stand"].size, prepared["config"], meta["stand"]["camera_id"])
            verified, detail = placement_evidence(checked, prepared["roi"], results["stand"]["chosen"]["mask"])
            if verified and kind == "dispense":
                score = self.ocr(fresh["stand"], checked["chosen"]["mask"], command["medicine"]["name"])
                verified = type(score) in (float, int) and np.isfinite(score) and max(.65, command["ocr_min_confidence"]) <= score <= 1
                detail = detail if verified else "Medicine label not verified after placement"
            fresh["stand"].save(Path(report["output_dir"]) / "placement_input.jpg")
            text_pick.ts.draw_sam3_style(fresh["stand"], checked, str(Path(report["output_dir"]) / "placement_preview.jpg"), True)
            report.update(status="PLACED" if verified else "PLACE_UNVERIFIED", place_verified=bool(verified),
                          placement_detail=detail)

        args = argparse.Namespace(sam_prompts=extra, text=target["target"], category=target["category"], image=self.image,
            hand_image=self.hand_image, arm_url=None if self.image else self.arm_url, calib=None,
            workspace=self.workspace, validation=self.validation, pick_at=None, vlm="none", items=self.items,
            grasp="auto", height=None, above=None, go=not dry_run, show=False, out=self.out)
        return text_pick.run(args, before_motion=before, after_motion=after)
