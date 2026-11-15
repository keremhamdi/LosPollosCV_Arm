"""Offline contract, controller-routing, observation, and calibrated-pair tests.

Station modules are replaced BEFORE importing controller code. No sockets, servo
drivers, camera devices, or subprocesses are opened by these tests.
"""
import asyncio
import base64
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
from PIL import Image

import arm_bridge
import camera_capture
import calibrated_pair
import pick_v5 as pick
from controller.arm.kinematics import ArmModel
from controller.arm.observations import ObservationStore, measured_pose, jpeg_dimensions
from controller.arm.vision_contract import normalize_grab, capabilities, width_fits


def jpeg():
    out = io.BytesIO()
    Image.new("RGB", (640, 480), "white").save(out, format="JPEG")
    return out.getvalue()


def joint(raw=100):
    return types.SimpleNamespace(raw=raw, goal=3000, range_min=0, range_max=4095,
                                 calibrated=True, torque=True)


class FakeModel:
    def angles_true(self, joints):
        return {str(mid): j.raw / 1000 for mid, j in joints.items()}

    def frames(self, angles):
        matrix = np.eye(4)
        matrix[0, 3] = angles["1"]
        return matrix.tolist(), []


def calibrated_fixture():
    k = [[800., 0, 320], [0, 800., 240], [0, 0, 1]]
    hand = np.eye(4)
    hand[0, 3] = .1
    packet = {"schema_version": 1, "synchronized": True, "frames": {}}
    calibration = {"schema_version": 1, "validated": True, "calibration_id": "synthetic-test",
                   "translation_units": "m", "max_pair_skew_s": .03, "cameras": {}}
    for role, pose in (("stand", np.eye(4)), ("hand", hand)):
        packet["frames"][role] = {"camera_id": role, "image_size": [640, 480], "sequence": 1,
                                  "timestamp_basis": "exposure_common_clock", "capture_timestamp_s": 5.,
                                  "jpeg_base64": base64.b64encode(jpeg()).decode(),
                                  "pose": {"pose_valid": True, "translation_units": "m",
                                           "T_base_gripper": pose.tolist(), "timestamp_basis": "exposure_common_clock",
                                           "capture_timestamp_s": 5.}}
        calibration["cameras"][role] = {"camera_id": role, "image_size": [640, 480], "K": k,
                                        "distortion": None, "T_base_camera": pose.tolist(),
                                        "T_gripper_camera": np.eye(4).tolist()}
    matches = {"calibration_id": "synthetic-test", "stand_sequence": 1, "hand_sequence": 1,
               "stand_points_px": [[320, 240]], "hand_points_px": [[240, 240]]}
    return packet, calibration, matches


class ContractTests(unittest.TestCase):
    def body(self):
        return arm_bridge.grasp_request(1, 2, 4, 12, {"type": "side", "width_cm": 4.7})

    def test_same_limits_on_both_sides(self):
        self.assertTrue(width_fits(4.7))
        self.assertFalse(width_fits(4.70001))
        self.assertEqual(normalize_grab(self.body())["method"], "side")

    def test_invalid_requests_fail_before_network(self):
        for change in ({"jaws": 4.8}, {"x": float("nan")}, {"method": "sideways"},
                       {"z": -1}, {"above": 3}, {"closing_angle_table_deg": 90},
                       {"protocol_version": 99}):
            with self.subTest(change=change):
                session = Mock()
                with self.assertRaises((ValueError, KeyError)):
                    arm_bridge.ArmBridge(session=session).submit_grasp({**self.body(), **change})
                session.get.assert_not_called()
                session.post.assert_not_called()

    def test_structured_response_and_side_request(self):
        session = Mock()
        session.get.return_value.json.return_value = {"capabilities": capabilities(), "busy": False, "pending": None}
        session.post.return_value.json.return_value = {"ok": True, "holding": False, "method": "side"}
        result = arm_bridge.ArmBridge(session=session).submit_grasp(self.body())
        self.assertFalse(result["holding"])
        self.assertEqual(session.post.call_args.kwargs["json"]["jaws"], 4.7)
        self.assertNotIn("u", session.post.call_args.kwargs["json"])

    def test_old_or_busy_controller_never_gets_post(self):
        for status in ({}, {"capabilities": capabilities(), "busy": True},
                       {"capabilities": capabilities(), "read_only": True},
                       {"capabilities": {**capabilities(), "gripper_open_cm": 6}}):
            session = Mock()
            session.get.return_value.json.return_value = status
            with self.assertRaises(ValueError):
                arm_bridge.ArmBridge(session=session).submit_grasp(self.body())
            session.post.assert_not_called()

    def test_top_angle_is_retained_and_explicitly_blocked(self):
        body = arm_bridge.grasp_request(1, 2, 4, 12, {"type": "top", "width_cm": 3, "closing_angle_table_deg": 40})
        self.assertEqual(body["closing_angle_table_deg"], 40)
        self.assertIn("ORIENTATION_UNSUPPORTED", arm_bridge.request_errors(body)[0])

    def test_legacy_pixel_request_still_normalizes(self):
        body = normalize_grab({"u": 4, "v": 5, "height": 4, "above": 12}, locate=lambda u, v: (u*2, v*2))
        self.assertEqual((body["x"], body["y"], body["method"]), (8, 10, "top"))


class PickWorkflowTests(unittest.TestCase):
    def test_saved_image_side_plan_never_contacts_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (160, 120), "gray").save(root/"frame.jpg")
            pairs = [[u, v, u/10, v/10] for u, v in ((0, 0), (159, 0), (0, 119), (159, 119), (80, 0), (80, 119), (0, 60), (159, 60))]
            (root/"calib.json").write_text(json.dumps({"size": 160, "pairs": pairs}))
            (root/"items.json").write_text(json.dumps({"test bottle": {"height_cm": 12, "width_cm": 4, "grasp": "side"}}))
            mask = np.zeros((120, 160), bool)
            mask[30:100, 60:100] = True
            candidate = {"mask": mask, "box": [60, 30, 100, 100], "sam_score": .9, "target_score": .9,
                         "is_match": True, "prompts": {"bottle"}, "vlm": {"match": True, "confidence": .9}}
            result = {"status": "GO", "chosen": candidate, "candidates": [candidate]}
            output = io.StringIO()
            with patch.object(pick.ts, "decide", return_value=result), patch.object(pick.requests, "get", side_effect=AssertionError("Network forbidden")), patch.object(pick.requests, "post", side_effect=AssertionError("Network forbidden")), patch.object(arm_bridge.ArmBridge, "submit_grasp", side_effect=AssertionError("Motion forbidden")), redirect_stdout(output):
                status = pick.main(["--image", str(root/"frame.jpg"), "--calib", str(root/"calib.json"),
                                    "--target", "test bottle", "--items", str(root/"items.json"), "--no-show", "--out", str(root/"preview.jpg")])
            self.assertEqual(status, "DRY_RUN")
            text = output.getvalue()
            plan, _ = json.JSONDecoder().raw_decode(text[text.index('{\n  "status"'):])
            self.assertEqual(plan["controller_request"]["method"], "side")
            self.assertAlmostEqual(plan["controller_request"]["jaws"], 4.)
            self.assertEqual(plan["execution_blocks"], [])
            self.assertTrue((root/"preview.jpg").exists())

    def test_offline_go_rejected_before_io(self):
        with patch.object(pick.requests, "get", side_effect=AssertionError("Network forbidden")):
            with self.assertRaises(SystemExit):
                pick.main(["--image", "not-read.jpg", "--calib", "not-read.json", "--target", "cup", "--go"])


class ObservationTests(unittest.TestCase):
    def test_measured_raw_not_goal_and_individual_freshness(self):
        joints = {i: joint() for i in range(1, 8)}
        times = {i: 10 for i in joints}
        result = measured_pose(joints, times, FakeModel(), 10.1)
        self.assertTrue(result["pose_valid"])
        self.assertEqual(result["T_base_gripper"][0][3], .1)
        times[3] = 9
        self.assertFalse(measured_pose(joints, times, FakeModel(), 10.1)["pose_valid"])

    def test_missing_joint_rejected(self):
        result = measured_pose({1: joint()}, {1: 10}, FakeModel(), 10)
        self.assertFalse(result["pose_valid"])
        self.assertIsNone(result["T_base_gripper"])

    def test_immutable_frame_pose_and_jpeg_export(self):
        pose = {"pose_valid": True, "T_base_gripper": np.eye(4).tolist()}
        store = ObservationStore()
        store.record("camera-a", "stand", jpeg(), pose, 10.)
        pose["T_base_gripper"][0][3] = 99
        packet = store.snapshot(10.2)
        self.assertEqual(packet["frames"]["stand"]["pose"]["T_base_gripper"][0][3], 0)
        self.assertFalse(packet["synchronized"])
        with tempfile.TemporaryDirectory() as folder:
            result = camera_capture.save_packet(packet, folder)
            self.assertFalse(result["motion_allowed"])
            self.assertTrue((Path(folder)/"stand.jpg").exists())
        self.assertEqual(jpeg_dimensions(jpeg()), [640, 480])

    def test_geometry_replay_and_missing_calibration(self):
        packet, calib, matches = calibrated_fixture()
        result = calibrated_pair.reconstruct(packet, calib, matches)
        np.testing.assert_allclose(result["points_base_m"], [[0, 0, 1]], atol=1e-8)
        self.assertFalse(result["motion_allowed"])
        template = json.loads(Path("two_camera_setup.json").read_text())
        with self.assertRaises(ValueError):
            calibrated_pair.prepare_pair(packet, template)

    def test_pair_identity_resolution_timing_and_pose_gates(self):
        mutations = [lambda p: p.update(synchronized=False),
                     lambda p: p["frames"]["hand"].update(camera_id="other"),
                     lambda p: p["frames"]["hand"].update(image_size=[320, 240]),
                     lambda p: p["frames"]["hand"].update(capture_timestamp_s=6),
                     lambda p: p["frames"]["hand"].update(timestamp_basis="host_receive"),
                     lambda p: p["frames"]["hand"]["pose"].update(pose_valid=False),
                     lambda p: p["frames"]["hand"]["pose"].update(capture_timestamp_s=4)]
        for mutate in mutations:
            p, c, _ = calibrated_fixture()
            mutate(p)
            with self.assertRaises(ValueError):
                calibrated_pair.prepare_pair(p, c)

    def test_wrong_frame_correspondences_blocked(self):
        p, c, m = calibrated_fixture()
        m["hand_sequence"] = 2
        with self.assertRaises(ValueError):
            calibrated_pair.reconstruct(p, c, m)

    def test_urdf_fk_loads_without_station(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.urdf"
            path.write_text('''<robot name="synthetic"><link name="base"/><link name="Gripper_Base_v1_1"/>
              <joint name="rev_motor_01" type="revolute"><parent link="base"/><child link="Gripper_Base_v1_1"/>
              <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint></robot>''')
            model = ArmModel(path)
            matrix, _ = model.frames({"rev_motor_01": np.pi/2})
            np.testing.assert_allclose(np.array(matrix)[:3, 3], [.1, 0, 0])
            np.testing.assert_allclose(np.array(matrix)[:2, :2], [[0, -1], [1, 0]], atol=1e-9)


class ControllerRoutingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        station = types.ModuleType("arm.station")
        station.REPO_ROOT = Path("controller").resolve()
        for name in ("FOLLOWER_BUS", "STATION_HOST"):
            setattr(station, name, "offline-test")
        for name in ("commands", "drivers", "st3215"):
            setattr(station, name, Mock())
        for name in ("new_station_client", "send_commands"):
            setattr(station, name, Mock(side_effect=AssertionError("Hardware I/O forbidden")))
        inference = types.ModuleType("target.gen_python.protobuf.drivers.inferences")
        inference.normvla = Mock()
        with patch.dict(sys.modules, {"arm.station": station, inference.__name__: inference}), patch.object(sys, "path", [str(Path("controller").resolve()), *sys.path]):
            cls.App = importlib.import_module("arm.click").App
            cls.reach_module = importlib.import_module("arm.reach")

    def setUp(self):
        self.arm = types.SimpleNamespace(joints={i: joint() for i in range(1, 9)},
                                         joint_updated_at={i: time.monotonic() for i in range(1, 9)})
        self.app = self.App(self.arm)
        self.reacher = types.SimpleNamespace(model=FakeModel(), grab=AsyncMock(return_value=True),
                                            grab_side=AsyncMock(return_value=False), state=lambda: (1, 2, 3, 0))
        self.app.reacher = self.reacher
        self.app.observation_model = self.reacher.model

    async def test_side_route_honors_lift_height_and_structured_result(self):
        result = await self.app.api("grab", {"x": 1, "y": 2, "z": 4, "above": 15, "jaws": 3, "method": "side", "protocol_version": 1})
        self.reacher.grab_side.assert_awaited_once_with(1, 2, 4, lift_cm=11, approach_h=15)
        self.reacher.grab.assert_not_awaited()
        self.assertFalse(result["holding"])
        self.assertEqual(result["method"], "side")

    async def test_legacy_top_route(self):
        result = await self.app.api("grab", {"x": 1, "y": 2, "z": 4, "above": 12})
        self.reacher.grab.assert_awaited_once_with(1, 2, 4, 12)
        self.assertTrue(result["holding"])

    async def test_bridge_to_controller_end_to_end_without_sockets(self):
        session = Mock()
        session.get.side_effect = lambda *a, **k: Mock(json=lambda: self.app.status())
        def post(url, json, timeout):
            result = asyncio.run(self.app.api("grab", json))
            return Mock(json=lambda: {"ok": True, **result})
        session.post.side_effect = post
        bridge = arm_bridge.ArmBridge(session=session)
        body = arm_bridge.grasp_request(1, 2, 4, 12, {"type": "side", "width_cm": 3.4})
        result = await asyncio.to_thread(bridge.submit_grasp, body)
        self.assertEqual(result["method"], "side")
        self.assertFalse(result["holding"])
        self.reacher.grab_side.assert_awaited_once_with(1, 2, 4, lift_cm=8, approach_h=12)

    async def test_gripper_feedback_stale_blocks_command(self):
        self.arm.joint_updated_at[8] -= 5
        with self.assertRaises(RuntimeError):
            await self.app.api("grab", {"x": 1, "y": 2, "z": 4, "above": 12})
        self.reacher.grab.assert_not_awaited()

    async def test_read_only_mode_rejects_all_actions(self):
        self.app.read_only = True
        for name in ("grab", "place", "hold", "release", "reset", "park", "reload"):
            with self.assertRaisesRegex(RuntimeError, "READ_ONLY"):
                await self.app.api(name, {})
        self.reacher.grab.assert_not_awaited()

    async def test_reload_reads_calibration_without_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text(json.dumps({"cam": 0, "size": 160,
                "pairs": [[0, 0, 0, 0], [160, 0, 16, 0], [0, 120, 0, 12], [160, 120, 16, 12]]}))
            self.app.pending = [1, 2]
            with patch.dict(self.App.api.__globals__, {"CAMERA_FILE": path}):
                await self.app.api("reload", {})
            self.assertEqual(len(self.app.camera.pairs), 4)
            self.assertEqual(self.app.camera.size, 160)
            self.assertIsNone(self.app.pending)
            self.reacher.grab.assert_not_awaited()
            self.reacher.grab_side.assert_not_awaited()

    async def test_actual_side_routine_uses_requested_approach_and_lift(self):
        reacher = object.__new__(self.reach_module.Reacher)
        reacher.arm = types.SimpleNamespace(move=AsyncMock(), joints={8: types.SimpleNamespace(raw=10, raw_for=lambda p: 0)})
        reacher.pose_for = Mock(return_value=([{"q": 1}], 90))
        reacher.plan_swing = Mock(return_value=["swing"])
        reacher.plan_line = Mock(return_value=["line"])
        reacher.duration = lambda path: 1.0
        reacher.follow = AsyncMock()
        with patch.object(self.reach_module.asyncio, "sleep", new=AsyncMock()):
            holding = await reacher.grab_side(1, 24, grab_h=4, lift_cm=11, approach_h=15)
        self.assertFalse(holding)
        reacher.plan_swing.assert_called_once_with({"q": 1}, 15)
        self.assertEqual(reacher.pose_for.call_args_list[0].args[2], [15])
        self.assertEqual(reacher.plan_line.call_args_list[-1].args, (1, 24, 15))

    async def test_rejections_never_invoke_motion(self):
        body = {"x": 1, "y": 2, "z": 4, "above": 12, "jaws": 3, "method": "side"}
        for change in ({"jaws": 5.4}, {"closing_angle_table_deg": 90}, {"method": "unknown"}):
            with self.assertRaises(ValueError):
                await self.app.api("grab", {**body, **change})
        self.arm.joint_updated_at[1] -= 5
        with self.assertRaises(RuntimeError):
            await self.app.api("grab", body)
        self.reacher.grab.assert_not_awaited()
        self.reacher.grab_side.assert_not_awaited()

    async def test_observation_http_get_is_read_only(self):
        self.app.record_frames({0: jpeg(), 1: jpeg()}, "test")
        reader = asyncio.StreamReader()
        reader.feed_data(b"GET /observation HTTP/1.1\r\nHost: test\r\n\r\n")
        reader.feed_eof()
        writer = Mock()
        writer.drain = AsyncMock()
        await self.app.handle(reader, writer)
        response = writer.write.call_args.args[0]
        packet = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(set(packet["frames"]), {"stand", "hand"})
        self.assertFalse(packet["synchronized"])
        self.reacher.grab.assert_not_awaited()
        self.reacher.grab_side.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
