"""Hardware-free text workflow tests. No model, network, camera or servo use."""
import argparse
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import text_pick as workflow


def fixture():
    image = Image.new("RGB", (160, 120), "gray")
    mask = np.zeros((120, 160), bool)
    mask[30:100, 60:100] = True
    candidate = {"mask": mask, "box": [60, 30, 100, 100], "sam_score": .95,
                 "target_score": .95, "prompts": {"test bottle"}, "is_match": True,
                 "vlm": {"match": True, "confidence": .95}}
    result = {"status": "GO", "chosen": candidate, "candidates": [candidate]}
    calibration = {"size": 160, "pairs": [[u, v, u/10, v/10] for u, v in
        ((0, 0), (159, 0), (0, 119), (159, 119), (80, 0), (80, 119), (0, 60), (159, 60))]}
    profile = {"width_cm": 4., "height_cm": 12., "grasp": "side"}
    return image, result, calibration, profile


def live_fixture():
    image, result, calibration, profile = fixture()
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    frames = {role: {"camera_id": role, "role": role, "sequence": i + 1,
                     "image_size": list(image.size), "receive_age_s": .01,
                     "jpeg_base64": base64.b64encode(buf.getvalue()).decode()}
              for i, role in enumerate(("stand", "hand"))}
    status = {**calibration, "camera_role": "stand", "calibrated": True, "cam": 0}
    packet = {"frames": frames}
    bridge = Mock()
    bridge.get.side_effect = lambda path: copy.deepcopy(status if path == "/status" else packet)
    bridge.submit_grasp.return_value = {"ok": True, "holding": True, "method": "side"}
    return bridge, image, result, calibration, profile, status, packet


class PlanningTests(unittest.TestCase):
    def test_mask_area_and_pixel_grasp_without_calibration(self):
        image, result, _, profile = fixture()
        report = workflow.plan_selection(image, result, profile)
        self.assertTrue(report["object_present"])
        self.assertEqual(report["region"]["area_px"], 2800)
        self.assertEqual(report["region"]["bbox_xyxy_px"], [60, 30, 100, 100])
        self.assertIn("point_px", report["grasp"])
        self.assertNotIn("width_cm", report["grasp"])
        self.assertIsNone(report["controller_request"])
        self.assertTrue(report["execution_blocks"])

    def test_absent_ambiguous_or_failed_never_plan(self):
        image, result, _, profile = fixture()
        for status in ("NOT_FOUND", "AMBIGUOUS", "CHECK_FAILED"):
            report = workflow.plan_selection(image, {**result, "status": status, "chosen": None}, profile)
            self.assertIsNone(report["controller_request"])
            self.assertNotIn("grasp", report)
            self.assertEqual(report["status"], status)

    def test_side_plan_preserves_cm_and_approach(self):
        image, result, calibration, profile = fixture()
        report = workflow.plan_selection(image, result, profile, calibration=calibration)
        self.assertEqual(report["execution_blocks"], [])
        self.assertEqual(report["controller_request"]["method"], "side")
        self.assertAlmostEqual(report["controller_request"]["jaws"], 4.)
        self.assertEqual(report["approach"]["approach_height_cm"], 17.)
        self.assertFalse(report["approach"]["collision_checked"])

    def test_missing_physical_dimensions_block_motion(self):
        image, result, calibration, _ = fixture()
        report = workflow.plan_selection(image, result, {}, calibration=calibration, method="side")
        self.assertTrue(any("width" in block for block in report["execution_blocks"]))
        self.assertTrue(any("height" in block for block in report["execution_blocks"]))

    def test_top_angle_is_not_silently_discarded(self):
        image, result, calibration, profile = fixture()
        report = workflow.plan_selection(image, result, profile, calibration=calibration, method="top")
        self.assertIn("closing_angle_table_deg", report["controller_request"])
        self.assertTrue(any("ORIENTATION_UNSUPPORTED" in block for block in report["execution_blocks"]))

    def test_oversize_and_weak_calibration_are_blocked(self):
        image, result, calibration, profile = fixture()
        calibration["pairs"] = [[u, v, 2*x, 2*y] for u, v, x, y in calibration["pairs"][:5]]
        report = workflow.plan_selection(image, result, profile, calibration=calibration)
        self.assertEqual(report["status"], "NO_GRASP")
        self.assertTrue(any("six calibration" in block for block in report["execution_blocks"]))

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            workflow.object_region(np.zeros((20, 20), bool))


class LiveTests(unittest.TestCase):
    def test_both_roles_read_from_controller_without_post(self):
        bridge, *_ = live_fixture()
        images, _, metadata = workflow.read_live(bridge)
        self.assertEqual(set(images), {"stand", "hand"})
        self.assertEqual(metadata["hand"]["camera_id"], "hand")
        self.assertNotIn("jpeg_base64", metadata["stand"])
        bridge.submit_grasp.assert_not_called()

    def test_missing_or_stale_hand_feed_is_rejected(self):
        for age in (None, 5., float("nan"), -1., True):
            bridge, *_, packet = live_fixture()
            packet["frames"]["hand"]["receive_age_s"] = age
            with self.assertRaises(ValueError):
                workflow.read_live(bridge)
        bridge, *_, packet = live_fixture()
        del packet["frames"]["hand"]
        with self.assertRaises(ValueError):
            workflow.read_live(bridge)

    def test_hand_homography_is_rejected(self):
        bridge, _, _, _, _, status, _ = live_fixture()
        status["camera_role"] = "hand"
        with self.assertRaises(ValueError):
            workflow.read_live(bridge)

    def test_fresh_scene_submits_exactly_once(self):
        bridge, image, result, calibration, profile, status, packet = live_fixture()
        images, _, metadata = workflow.read_live(bridge)
        report = workflow.plan_selection(image, result, profile, calibration=calibration)
        for frame in packet["frames"].values():
            frame["sequence"] += 2
        workflow.execute_live(report, images, {r: result for r in images}, status, metadata, bridge)
        bridge.submit_grasp.assert_called_once_with(report["controller_request"])
        self.assertEqual(report["status"], "HOLDING_REPORTED")

    def test_calibration_change_or_frozen_feed_blocks_post(self):
        for change in ("calibration", "frozen"):
            bridge, image, result, calibration, profile, status, packet = live_fixture()
            images, initial, metadata = workflow.read_live(bridge)
            report = workflow.plan_selection(image, result, profile, calibration=calibration)
            if change == "calibration":
                status["pairs"][0][2] += 1
            with self.assertRaises(ValueError):
                workflow.execute_live(report, images, {r: result for r in images}, initial, metadata, bridge)
            bridge.submit_grasp.assert_not_called()

    def test_scene_change_blocks_post(self):
        bridge, image, result, calibration, profile, status, packet = live_fixture()
        images, _, metadata = workflow.read_live(bridge)
        report = workflow.plan_selection(image, result, profile, calibration=calibration)
        buf = io.BytesIO()
        Image.new("RGB", image.size, "white").save(buf, format="JPEG")
        for frame in packet["frames"].values():
            frame["sequence"] += 2
        packet["frames"]["hand"]["jpeg_base64"] = base64.b64encode(buf.getvalue()).decode()
        with self.assertRaises(ValueError):
            workflow.execute_live(report, images, {r: result for r in images}, status, metadata, bridge)
        bridge.submit_grasp.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def args(self, root, **kw):
        return argparse.Namespace(text="pick up the test bottle", image=str(root / "frame.jpg"),
                                  arm_url=None, hand_image=str(root / "frame.jpg"),
                                  calib=None, category=None, vlm="none", items=str(root / "items.json"),
                                  grasp="auto", height=None, above=None, go=False, show=False,
                                  out=str(root / "output"), **kw)

    def test_saved_pair_exports_masks_previews_and_report_without_controller(self):
        image, result, _, profile = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image.save(root / "frame.jpg")
            (root / "items.json").write_text(json.dumps({"test bottle": profile}))
            with patch.object(workflow.ts, "decide", return_value=result), patch.object(workflow, "ArmBridge") as bridge:
                report = workflow.run(self.args(root))
            bridge.assert_not_called()
            out = Path(report["output_dir"])
            self.assertEqual(report["target"], "test bottle")
            self.assertEqual(report["hand_view"]["status"], "PREVIEW")
            self.assertEqual(json.loads((out / "result.json").read_text())["region"]["area_px"], 2800)
            for role in ("stand", "hand"):
                self.assertEqual(int((np.asarray(Image.open(out / f"{role}_mask.png")) > 0).sum()), 2800)
                self.assertTrue((out / f"{role}_preview.jpg").exists())

    def test_cancellation_does_not_read_images_or_load_models(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args.text = "cancel"
            with patch.object(workflow.ts, "decide") as selector:
                report = workflow.run(args)
            selector.assert_not_called()
            self.assertEqual(report["status"], "CANCELLED")

    def test_saved_image_go_rejected_before_loading(self):
        with self.assertRaises(SystemExit), patch.object(workflow, "run") as run:
            workflow.main(["--image", "missing.jpg", "--text", "cup", "--go"])
        run.assert_not_called()

    def test_hand_negative_blocks_live_pick(self):
        bridge, _, result, _, profile, _, _ = live_fixture()
        negative = {"status": "NOT_FOUND", "candidates": [], "chosen": None}
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args.image = args.hand_image = None
            args.arm_url, args.go = "http://test.invalid", True
            with patch.object(workflow, "ArmBridge", return_value=bridge), \
                 patch.object(workflow.ts, "decide", side_effect=[result, negative]), \
                 patch.object(workflow.gp, "load_profile", return_value=profile):
                report = workflow.run(args)
            self.assertEqual(report["status"], "EXECUTION_BLOCKED")
            bridge.submit_grasp.assert_not_called()

    def test_motion_timeout_is_unknown_and_never_retried(self):
        bridge, image, result, calibration, profile, status, packet = live_fixture()
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            args.image = args.hand_image = None
            args.arm_url, args.go = "http://test.invalid", True
            meta = {role: {"camera_id": role, "sequence": 1} for role in ("stand", "hand")}
            fresh = {role: {"camera_id": role, "sequence": 2} for role in meta}
            bridge.submit_grasp.side_effect = TimeoutError("connection lost")
            with patch.object(workflow, "ArmBridge", return_value=bridge), \
                 patch.object(workflow, "read_live", side_effect=[({r: image for r in meta}, status, meta),
                                                                ({r: image for r in meta}, status, fresh)]), \
                 patch.object(workflow.ts, "decide", return_value=result), \
                 patch.object(workflow.gp, "load_profile", return_value=profile):
                report = workflow.run(args)
            self.assertEqual(report["status"], "MOTION_RESULT_UNKNOWN")
            bridge.submit_grasp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
