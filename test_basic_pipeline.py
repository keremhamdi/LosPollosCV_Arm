"""Basic backend: the arm page, camera, OCR and SAM are replaced; nothing real is contacted."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import basic_pipeline as bp
from arm_service import Service
from test_arm_service import command


def reply(data=None, content=None):
    return Mock(json=Mock(return_value=data), content=content)


class BasicPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        frame = io.BytesIO()
        Image.new("RGB", (64, 48)).save(frame, format="JPEG")
        self.status = {"read_only": False}
        self.pressed = []
        self.holding = True
        mask = np.zeros((48, 64), bool)
        mask[20:40, 10:30] = True
        self.candidates = [{"mask": mask, "target_score": .8, "sam_score": .8, "box": [10, 20, 30, 40]}]
        self.word = None

        def get(url, **_):
            return reply(self.status) if url.endswith("/status") else reply(content=frame.getvalue())

        def post(url, json=None, **_):
            name = url.rsplit("/", 1)[1]
            self.pressed.append((name, json))
            return reply({"ok": True, "x": -3.0, "y": 22.0, "holding": self.holding})

        for target, new in ((bp.requests, {"get": get, "post": post}),):
            for attr, value in new.items():
                patcher = patch.object(target, attr, side_effect=value)
                patcher.start()
                self.addCleanup(patcher.stop)
        for attr, value in (("find_word", lambda img, word: self.word), ("object_at", lambda *a: self.candidates[0])):
            patcher = patch.object(bp.G, attr, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(bp.ts, "decide", side_effect=lambda *a: {"candidates": self.candidates})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pipe = bp.BasicPipeline(arm_url="http://arm.invalid", places={"bowl": (-20., 15.)})
        self.service = Service({}, pipeline=self.pipe, basic=True, enable_motion=True,
                               journal=str(Path(self.temp.name)/"j.sqlite3"))

    def buttons(self):
        return [name for name, _ in self.pressed]

    def test_medicine_is_read_grabbed_put_in_the_bowl_and_reported_unverified(self):
        self.word = (15., 30., "Aspirin", .9)
        result = self.service.handle("/v1/dispense", command("dispense", dry_run=False))
        self.assertEqual(self.buttons(), ["click", "grab", "place", "park"])
        self.assertEqual(self.pressed[2][1], {"x": -20., "y": 15., "height": 8., "above": 14.})
        self.assertTrue(result["ok"])
        self.assertEqual((result["final_state"], result["label"], result["score"]), ("placed", "Aspirin", .9))
        self.assertFalse(result["place_verified"])
        self.assertIn("not checked by camera", result["detect_detail"])

    def test_everyday_object_is_found_by_look_and_unknown_place_is_refused(self):
        def move(**changes):
            return command("move_object", **{"object": {"label": "water bottle"}, "destination": "bowl",
                                             "actions": ["locate", "pick", "place"], "dry_run": False, **changes})
        self.assertTrue(self.service.handle("/v1/move_object", move())["ok"])
        self.assertEqual(self.buttons(), ["click", "grab", "place", "park"])
        self.pressed.clear()
        result = self.service.handle("/v1/move_object", move(destination="handover"))
        self.assertFalse(result["ok"])
        self.assertIn("No table position", result["error"])
        self.assertEqual(self.buttons(), ["click"])

    def test_preview_not_found_empty_gripper_and_read_only_never_claim_placement(self):
        preview = self.service.handle("/v1/dispense", command("dispense"))
        self.assertEqual((preview["ok"], preview["error_code"], self.buttons()), (False, "DRY_RUN", ["click"]))
        self.pressed.clear()
        self.holding = False
        empty = self.service.handle("/v1/dispense", command("dispense", dry_run=False))
        self.assertEqual((empty["ok"], empty["final_state"], self.buttons()), (False, "failed", ["click", "grab", "park"]))
        self.assertIn("closed on nothing", empty["error"])
        self.pressed.clear()
        self.status["read_only"] = True
        blocked = self.service.handle("/v1/dispense", command("dispense", dry_run=False))
        self.assertEqual((blocked["ok"], self.buttons()), (False, ["click"]))
        self.candidates = []
        self.pressed.clear()
        missing = self.service.handle("/v1/dispense", command("dispense", dry_run=False))
        self.assertEqual((missing["ok"], missing["error_code"], self.buttons()), (False, "NOT_FOUND", []))


if __name__ == "__main__":
    unittest.main()
