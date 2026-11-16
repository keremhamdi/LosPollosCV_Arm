"""Buddy -> arm server: the arm page, camera and SAM are replaced; nothing real is contacted."""
import argparse
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import dispense_server as ds


def request(kind, **changes):
    body = {"schema": ds.SCHEMA, "message_type": kind, "request_id": "r1", "dry_run": False, "one_shot_detect": True}
    if kind == "dispense":
        body.update(medicine={"id": 1, "name": "Aspirin", "person": None}, ocr_min_confidence=.65,
                    actions=["locate", "pick_to_bowl", "verify_place"], slot_hint=None)
    else:
        body.update(object={"label": "water bottle"}, destination="bowl", actions=["locate", "pick", "place"])
    return {**body, **changes}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.a = argparse.Namespace(arm_url="http://arm.invalid", meds={"aspirin": "small bottle"}, read_labels=False,
                                    bowl=[-20., 15.], bowl_text="bowl", handover=None, listen_only=False,
                                    bowl_height=8., close=None, above=14., tries=2)
        frame = io.BytesIO()
        Image.new("RGB", (64, 48)).save(frame, format="JPEG")
        self.pressed, self.holding, self.asked = [], [True], []
        mask = np.zeros((48, 64), bool)
        mask[20:40, 10:30] = True
        self.candidates = [{"mask": mask, "target_score": .8}]
        # the fixed camera's calibration: pixel u, v shows table x = (u - 32) / 2, y = 10 + v / 2 cm
        marks = [[u, v, (u - 32) / 2, 10 + v / 2] for u in (0, 21, 42, 63) for v in (0, 24, 47)]
        self.status = {"read_only": False, "pairs": marks, "size": 64, "log": "done"}

        def get(url, **_):
            return Mock(json=Mock(return_value=self.status), content=frame.getvalue())

        def post(url, json=None, **_):
            name = url.rsplit("/", 1)[1]
            self.pressed.append((name, json))
            holding = (self.holding.pop(0) if len(self.holding) > 1 else self.holding[0]) if name == "grab" else None
            return Mock(json=Mock(return_value={"ok": True, "x": -3., "y": 22., "holding": holding,
                                                "report": {"contact": True, "met_at_cm": .1}}))

        def decide(img, text, category):
            self.asked.append(text)
            return {"candidates": self.candidates}

        out = contextlib.redirect_stdout(io.StringIO())
        out.__enter__()
        self.addCleanup(out.__exit__, None, None, None)
        ds.G._maps.clear()
        for target, attr, value in ((ds.requests, "get", get), (ds.requests, "post", post), (ds.ts, "decide", decide),
                                    (ds.G, "show_view", lambda *a, **k: None)):
            patcher = patch.object(target, attr, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(ds.G, "HAND_FILE", "no/such/hand.json")      # no gripper camera setup here
        patcher.start()
        self.addCleanup(patcher.stop)

    def buttons(self):
        return [name for name, _ in self.pressed]

    def test_medicine_still_works_as_before(self):
        out = ds.dispense(self.a, request("dispense"))
        self.assertEqual((out["ok"], out["final_state"], out["message_type"]), (True, "placed", "dispense_result"))
        self.assertEqual(self.asked, ["small bottle"])
        self.assertEqual(self.buttons(), ["grab", "place", "park"])

    def test_it_picks_the_careful_way_at_the_middle_of_where_the_object_stands(self):
        ds.dispense(self.a, request("dispense"))
        grab = self.pressed[0][1]
        self.assertEqual((grab["guard"], grab["feel_all_the_way"], grab["method"]), (True, True, "top"))
        self.assertNotIn("u", grab)                           # table cm, not a pixel for the page to map
        self.assertAlmostEqual(grab["x"], (19.5 - 32) / 2, delta=.5)      # the outline spans u 10..29, v 20..39
        self.assertTrue(10 + 29.5 / 2 - .5 < grab["y"] < 10 + 39 / 2)     # its middle or nearer the camera, never beyond
        self.a.close = 5.
        self.pressed.clear()
        ds.dispense(self.a, request("dispense"))
        self.assertEqual((self.pressed[0][1]["height"], "feel_all_the_way" in self.pressed[0][1]), (5., False))

    def test_a_miss_is_followed_by_a_second_look_and_grab(self):
        self.holding = [False, True]
        out = ds.dispense(self.a, request("dispense"))
        self.assertEqual((out["ok"], out["final_state"]), (True, "placed"))
        self.assertEqual(self.buttons(), ["grab", "release", "park", "grab", "place", "park"])
        self.assertEqual(self.asked, ["small bottle", "small bottle"])
        self.assertIn("2 grabs", out["detect_detail"])
        self.holding, self.pressed = [False], []
        out = ds.dispense(self.a, request("dispense"))
        self.assertEqual((out["ok"], out["final_state"]), (False, "failed"))
        self.assertIn("pick failed", out["error"])
        self.assertEqual(self.buttons(), ["grab", "release", "park", "grab", "park"])
        self.assertNotIn("place", self.buttons())

    def test_everyday_object_goes_to_the_bowl_or_the_handover_spot(self):
        out = ds.dispense(self.a, request("move_object"), "move_object")
        self.assertEqual((out["ok"], out["final_state"], out["message_type"], out["label"]),
                         (True, "placed", "move_object_result", "water bottle"))
        self.assertEqual(self.asked, ["water bottle"])
        self.assertEqual(self.pressed[1][1]["x"], -20.)
        refused = ds.dispense(self.a, request("move_object", destination="handover"), "move_object")
        self.assertEqual((refused["ok"], refused["final_state"]), (False, "aborted"))
        self.a.handover = [-5., 30.]
        self.pressed.clear()
        self.assertTrue(ds.dispense(self.a, request("move_object", destination="handover"), "move_object")["ok"])
        self.assertEqual((self.pressed[1][1]["x"], self.pressed[1][1]["y"]), (-5., 30.))

    def test_refusals_and_failures_never_move_or_claim_placement(self):
        for changes in ({"object": {"label": "a knife"}}, {"object": {"label": ""}}, {"destination": "kitchen"},
                        {"message_type": "dispense"}):
            out = ds.dispense(self.a, request("move_object", **changes), "move_object")
            self.assertEqual((out["ok"], out["final_state"]), (False, "aborted"), changes)
        self.assertEqual(self.pressed, [])
        self.candidates = []
        out = ds.dispense(self.a, request("move_object"), "move_object")
        self.assertEqual((out["ok"], out["detect_detail"], self.pressed), (False, "not_found", []))

    def test_listen_only_shows_the_request_and_touches_nothing(self):
        self.a.listen_only = True
        for kind in ("dispense", "move_object"):
            out = ds.dispense(self.a, request(kind), kind)
            self.assertEqual((out["ok"], out["final_state"]), (False, "aborted"))
            self.assertIn("connection test", out["error"])
        self.assertEqual((self.pressed, self.asked), ([], []))
        self.assertIsNone(ds.check_taken(request("dispense"))["taken"])


if __name__ == "__main__":
    unittest.main()
