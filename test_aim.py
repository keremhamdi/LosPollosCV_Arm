"""Aiming at small things, tested on a pretend table (sim_scene.py). No arm, camera, model or network."""
import argparse
import contextlib
import importlib
import io
import itertools
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image

import aim
import grab_text as G
import sim_scene as sim


def quiet(test):
    """Hide what the code prints while a test runs."""
    out = contextlib.redirect_stdout(io.StringIO())
    out.__enter__()
    test.addCleanup(out.__exit__, None, None, None)


def controller():
    """The arm page's code, loaded with Station replaced: nothing can reach hardware."""
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
    with patch.dict(sys.modules, {"arm.station": station, inference.__name__: inference}), \
            patch.object(sys, "path", [str(Path("controller").resolve()), *sys.path]):
        return importlib.import_module("arm.click"), importlib.import_module("arm.reach")


SPOTS = np.array([[x, y, 0.0] for x in (-18, -10, -2, 6, 14) for y in (10, 18, 26, 34)])


class TableMapTests(unittest.TestCase):
    def test_without_lens_correction_it_gives_the_arm_pages_numbers(self):
        click, _ = controller()
        cam = sim.FixedCamera()
        for pairs in (sim.boards(cam), sim.boards(cam)[:54:6]):          # three boards; nine loose marks
            page = click.Camera.__new__(click.Camera)
            page.cam, page.size, page.pairs = 0, sim.W, pairs
            page._fit()
            mine = aim.TableMap(pairs, sim.W, (sim.H, sim.W), lens=False)
            px = cam.project(SPOTS)
            np.testing.assert_allclose(mine.locate(px), [page.locate(u, v) for u, v in px], atol=1e-6)
            np.testing.assert_allclose(mine.locate(px[3]), page.locate(*px[3]), atol=1e-6)

    def test_a_bent_picture_is_straightened(self):
        cam = sim.FixedCamera(k1=-0.32, k2=0.09)
        px = cam.project(SPOTS)
        worst = lambda m: np.linalg.norm(m.locate(px) - SPOTS[:, :2], axis=1).max()
        for where in (sim.BOARDS[:1], sim.BOARDS):           # one checkerboard position; three
            pairs = sim.boards(cam, where=where)
            page, mine = (aim.TableMap(pairs, sim.W, (sim.H, sim.W), lens=on) for on in (False, True))
            self.assertTrue(mine.lens_on)
            self.assertIn("lens correction on", mine.lens_note)
            self.assertGreater(worst(page), 0.2)             # away from the marks the flat mapping drifts
            self.assertLess(worst(mine), 0.03)
            np.testing.assert_allclose(mine.cam[3], cam.C, atol=1.0)      # it also finds where the camera is
        # the arm being a little off for each board is not the lens's doing, and is kept as the page has it
        off = sim.boards(cam, shifts=[(0.3, -0.2), (-0.2, 0.25), (0.1, 0.1)])
        page, mine = (aim.TableMap(off, sim.W, (sim.H, sim.W), lens=on) for on in (False, True))
        self.assertLessEqual(worst(mine), worst(page))
        np.testing.assert_allclose(mine.k, aim.TableMap(sim.boards(cam), sim.W, (sim.H, sim.W)).k, rtol=0.02)

    def test_a_straight_picture_and_poor_marks_are_left_alone(self):
        straight = sim.FixedCamera(k1=0, k2=0)
        pairs = sim.boards(straight)
        mine = aim.TableMap(pairs, sim.W, (sim.H, sim.W))
        self.assertFalse(mine.lens_on)
        px = straight.project(SPOTS)
        np.testing.assert_allclose(mine.locate(px), aim.TableMap(pairs, sim.W, (sim.H, sim.W), lens=False).locate(px))
        self.assertFalse(aim.TableMap(sim.boards(sim.FixedCamera())[:54:6], sim.W, (sim.H, sim.W)).lens_on)   # 9 marks
        for seed in range(8):                               # 41 hand-clicked marks, each some mm to a cm off
            rng = np.random.default_rng(seed)
            for off in (0.3, 1.0):
                loose = [[u, v, x + rng.normal(0, off), y + rng.normal(0, off)] for u, v, x, y in sim.boards(straight)[::4]]
                mine = aim.TableMap(loose, sim.W, (sim.H, sim.W))
                self.assertFalse(mine.lens_on, (seed, off, mine.lens_note))
        with self.assertRaises(ValueError):
            aim.TableMap(pairs[:3], sim.W)

    def test_pixel_of_runs_the_mapping_backwards_and_covers_knows_the_marked_area(self):
        cam = sim.FixedCamera()
        mine = aim.TableMap(sim.boards(cam), sim.W, (sim.H, sim.W))
        for spot in SPOTS[::3]:
            u, v = mine.pixel_of(spot[:2])
            np.testing.assert_allclose(mine.locate([u, v]), spot[:2], atol=0.02)
        self.assertTrue(mine.covers(cam.project([-4.0, 20.0, 0.0])[0], 0))
        self.assertFalse(mine.covers((30.0, 30.0), 100))     # top left corner: the room, not the table


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.cam = sim.FixedCamera()
        self.map = aim.TableMap(sim.boards(self.cam), sim.W, (sim.H, sim.W))
        self.rng = np.random.default_rng(7)

    def looks(self, points):
        """What footprint() says for this object, over a clean outline and five ragged ones."""
        clean = sim.outline(self.cam, points)
        return [aim.footprint(m, self.map, sim.W) for m in [clean] + [sim.ragged(clean, self.rng) for _ in range(5)]]

    def test_small_standing_things_are_aimed_at_their_middle(self):
        things = {"bottle cap": sim.cylinder(4, 30, 3, 1.5), "2 cm cube": sim.box(-12, 16, 2, 2, 2, 0.4),
                  "1.2 cm cube": sim.box(-4, 14, 1.2, 1.2, 1.2), "small bottle": sim.cylinder(8, 12, 2.5, 5),
                  "bottle": sim.cylinder(-6, 22, 3, 8), "lip balm": sim.cylinder(5, 28, 1.6, 7)}
        for name, points in things.items():
            true, tall = points[:, :2].mean(axis=0), points[:, 2].max()
            for fp in self.looks(points):
                self.assertEqual(fp["how"], "stands", name)
                self.assertLess(np.linalg.norm(np.array(fp["center"]) - true), 0.45, name)
                self.assertLess(abs(fp["height"] - tall), max(0.6, 0.2 * tall), name)
                self.assertLess(abs(fp["width"] - np.ptp(points[:, 0])), 0.9, name)
        # the old aim (the outline's lowest fifth) is the near edge: about 1 cm off for short things
        cap = things["bottle cap"]
        old = self.map.locate(np.array(G.base_pixel(sim.outline(self.cam, cap))))
        self.assertGreater(np.linalg.norm(old - cap[:, :2].mean(axis=0)), 0.8)

    def test_flat_things_are_aimed_at_the_middle_of_their_outline(self):
        things = {"blister pack": sim.box(6, 18, 10, 6, 0.5, 0.5), "coin": sim.cylinder(0, 30, 2.5, 0.2),
                  "pen lying across": sim.box(-2, 34, 14, 1, 1)}
        for name, points in things.items():
            for fp in self.looks(points):
                self.assertTrue(fp["flat"], name)
                self.assertIsNone(fp["height"], name)
                self.assertLess(np.linalg.norm(np.array(fp["center"]) - points[:, :2].mean(axis=0)), 0.7, name)

    def test_something_lying_towards_the_camera_is_not_taken_for_something_tall(self):
        book = sim.box(2, 20, 9, 13, 1.0)
        for fp in self.looks(book):
            self.assertEqual((fp["how"], fp["flat"], fp["height"]), ("long", True, None))
            self.assertLess(np.linalg.norm(np.array(fp["center"]) - book[:, :2].mean(axis=0)), 1.0)
        box = sim.box(0, 20, 8, 3, 12)                       # a medicine box standing, wide side to the camera
        for fp in self.looks(box):
            self.assertEqual(fp["how"], "stands")
            self.assertLess(np.linalg.norm(np.array(fp["center"]) - box[:, :2].mean(axis=0)), 1.0)

    def test_an_enlarged_cut_out_gives_the_same_place_only_sharper(self):
        cube = sim.box(-4, 14, 1.2, 1.2, 1.2)
        true = cube[:, :2].mean(axis=0)
        u, v = self.cam.project(cube).mean(axis=0)
        offset, zoom = (int(u) - 60, int(v) - 60), 4.0
        big = sim.outline(self.cam, cube, zoom, offset, (480, 480))
        fine = aim.footprint(big, self.map, sim.W, offset=offset, zoom=zoom)
        self.assertLess(np.linalg.norm(np.array(fine["center"]) - true), 0.12)
        self.assertLess(abs(fine["height"] - 1.2), 0.25)

    def test_empty_or_split_outlines(self):
        self.assertIsNone(aim.footprint(np.zeros((sim.H, sim.W), bool), self.map, sim.W))
        cap = sim.cylinder(4, 30, 3, 1.5)
        mask = sim.outline(self.cam, cap)
        mask[100:103, 100:103] = True                        # a speck far away must not drag the aim
        fp = aim.footprint(mask, self.map, sim.W)
        self.assertLess(np.linalg.norm(np.array(fp["center"]) - cap[:, :2].mean(axis=0)), 0.3)

    def test_without_knowing_where_the_camera_is_it_still_aims_inside_the_object(self):
        self.map.cam = None
        cap = sim.cylinder(4, 30, 3, 1.5)
        fp = aim.footprint(sim.outline(self.cam, cap), self.map, sim.W)
        self.assertEqual(fp["how"], "unknown height")
        self.assertLess(np.linalg.norm(np.array(fp["center"]) - cap[:, :2].mean(axis=0)), 0.5)


class HandCameraTests(unittest.TestCase):
    """The gripper camera's aim correction, with a gripper that leans out the further it reaches."""
    SPOTS = [(2, 20, 3), (-8, 16, 3), (10, 24, 3), (-4, 28, 2), (6, 32, 5), (-14, 30, 1.5), (12, 12, 8)]

    def setUp(self):
        quiet(self)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.world = sim.HandWorld((2.0, 20.0, 3.0), cam_back=7.5)
        self.a = G.options(arm_url="http://arm.invalid")
        self.rng = np.random.default_rng(0)
        self.sees = True

        def post(url, json=None, **_):
            name = url.rsplit("/", 1)[1]
            return sim.Reply(self.world.goto(json["x"], json["y"], json["height"], json.get("tilt"))
                             if name == "goto" else {"ok": True})

        def find(a, text, near=None, tries=3):
            p = self.world.pixel() if self.sees else None
            if p is None:
                return None, "picture", None
            return (p[0] + self.rng.normal(0, 0.7), p[1] + self.rng.normal(0, 0.7)), "picture", {"mask": None}

        answers = itertools.cycle(["", "3", ""])             # jaws around it; 3 cm tall; arm may move
        for target, name, value in ((G, "HAND_FILE", os.path.join(folder.name, "hand.json")),
                                    (G.requests, "post", post), (G, "find_in_hand", find),
                                    (G.requests, "get", lambda url, **_: sim.Reply({"tip": [2.0, 20.0, 0.4]})),
                                    (G, "save_hand_view", lambda *a, **k: None)):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("builtins.input", lambda *a: next(answers))
        p.start()
        self.addCleanup(p.stop)
        G._told.clear()

    def setup(self, **drop):
        self.world.obj = np.array((2.0, 20.0, 3.0))          # the object the jaws were put around
        G.hand_calibrate(self.a, "thing")
        cal = json.load(open(G.HAND_FILE))
        for key in drop:
            cal.pop(key, None)
        json.dump(cal, open(G.HAND_FILE, "w"))
        return cal

    def errors(self):
        """How far from each object the corrected aim ends up (cm), and the lean it was looked at with.
        The fixed camera's aim starts 2 cm off and its height guess 0.7 cm off."""
        out = []
        for spot in self.SPOTS:
            self.world.obj = np.array(spot, float)
            x, y, info = G.hand_correct(self.a, "thing", spot[0] + 1.5, spot[1] - 1.2, spot[2] + 0.7)
            out.append((float(np.hypot(x - spot[0], y - spot[1])), self.world.tilt, info))
        return out

    def test_setup_measures_the_lean_and_keeps_one_lean_for_its_looks(self):
        cal = self.setup()
        self.assertEqual(cal["tilt"], 0.0)
        self.assertEqual(cal["lean"]["deg"], [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])   # as far as this arm leans there
        self.assertEqual(cal["lean"]["px"][0], cal["target_px"])
        looks = [m for m in self.world.moves if abs(m[2] - 16.0) < 1e-9]
        self.assertEqual([m[3] for m in looks][:5], [0.0, 0.0, 0.0, 5.0, 10.0])
        J = np.array(cal["px_per_cm"])
        for tilt in (5.0, 10.0, 12.5, 20.0):                 # what it measured is close to what geometry says
            seen, geometry = (G.lean_shift(c, J, 2.0, 20.0, tilt) for c in (cal, {**cal, "lean": None}))
            self.assertLess(np.linalg.norm(seen - geometry), 0.25 * np.linalg.norm(geometry) + 3)
        np.testing.assert_allclose(G.lean_shift(cal, J, 2.0, 20.0, 0.0), 0)

    def test_jaws_end_up_over_the_object_whatever_the_lean(self):
        for drop, close in (({}, 0.4), ({"lean": 1}, 0.6)):    # lean as measured in setup; lean from geometry only
            self.world.moves.clear()
            self.setup(**drop)
            results = self.errors()
            self.assertGreater(max(t for _, t, _ in results), 25, "some spots need a strong lean")
            for err, tilt, info in results:
                self.assertLess(err, close, (drop, tilt))
                self.assertEqual(info, {"used": True, "lean_aware": True})

    def test_a_setup_from_before_leaves_the_arm_short_where_it_leans(self):
        self.setup(lean=1, tilt=1)
        results = self.errors()
        self.assertTrue(all(not info["lean_aware"] for _, _, info in results))
        self.assertTrue(all(err < 0.45 for err, tilt, _ in results if tilt == 0))
        self.assertTrue(all(err > 1.5 for err, tilt, _ in results if tilt >= 10))
        self.assertIn("Run handcal once more", sys.stdout.getvalue())

    def test_it_looks_with_the_setups_lean_where_the_arm_can(self):
        self.setup()
        self.world.lean_from = 0.0                            # now even the setup spot needs a lean ...
        self.world.obj = np.array((2.0, 20.0, 3.0))
        self.world.moves.clear()
        x, y, info = G.hand_correct(self.a, "thing", 3.0, 19.0, 3.0)
        self.assertTrue(all(m[3] > 30 for m in self.world.moves))    # ... so it can't look as in the setup,
        self.assertLess(np.hypot(x - 2.0, y - 20.0), 0.6)            # and allows for the lean, past what was measured
        self.world.lean_from = 12.0                           # a lean the setup did measure: spot on
        self.world.moves.clear()
        x, y, info = G.hand_correct(self.a, "thing", 3.0, 19.0, 3.0)
        self.assertTrue(all(0 < m[3] <= 30 for m in self.world.moves))
        self.assertLess(np.hypot(x - 2.0, y - 20.0), 0.3)

    def test_not_seeing_it_keeps_the_fixed_cameras_aim(self):
        self.setup()
        self.sees = False
        x, y, info = G.hand_correct(self.a, "thing", 5.0, 25.0, 3.0)
        self.assertEqual((x, y, info["used"]), (5.0, 25.0, False))
        self.a.no_hand = True
        self.assertEqual(G.hand_correct(self.a, "thing", 5.0, 25.0, 3.0)[2], {"used": False, "lean_aware": False})

    def test_an_older_arm_page_that_does_not_say_the_lean_still_works(self):
        self.world.says_lean = False
        cal = self.setup()
        self.assertNotIn("tilt", cal)
        self.world.obj = np.array((-8.0, 16.0, 3.0))
        x, y, info = G.hand_correct(self.a, "thing", -7.0, 15.0, 3.0)
        self.assertEqual(info, {"used": True, "lean_aware": False})
        self.assertLess(np.hypot(x + 8.0, y - 16.0), 0.45)


class Table:
    """A pretend arm page over a pretend table: the fixed camera sees the objects, and a grab
    succeeds only if the jaws come down within `tolerance` cm of the object's middle."""

    def __init__(self, test, things, tolerance=0.8):
        self.cam = sim.FixedCamera()
        self.pairs = sim.boards(self.cam)
        self.things, self.tolerance = dict(things), tolerance
        self.posts, self.asked, self.status = [], [], {"read_only": False, "pending": None, "log": "done"}
        self.miss_first = 0
        self.park_ok = True
        jpeg = io.BytesIO()
        Image.new("RGB", (sim.W, sim.H)).save(jpeg, format="JPEG")
        self.jpeg = jpeg.getvalue()
        for target, name, value in ((G.requests, "get", self.get), (G.requests, "post", self.post),
                                    (G.ts, "decide", self.decide), (G, "show_view", lambda *a, **k: None),
                                    (G, "HAND_FILE", "no/such/hand.json")):
            p = patch.object(target, name, value)
            p.start()
            test.addCleanup(p.stop)
        G._maps.clear()

    def get(self, url, **_):
        return sim.Reply({**self.status, "pairs": self.pairs, "size": sim.W}, self.jpeg)

    def post(self, url, json=None, **_):
        name = url.rsplit("/", 1)[1]
        self.posts.append((name, json))
        if name == "grab":
            if self.status.get("refuse"):
                return sim.Reply({"ok": False, "error": self.status["refuse"]})
            near = [n for n, p in self.things.items()
                    if np.hypot(*(p[:, :2].mean(axis=0) - [json["x"], json["y"]])) <= self.tolerance]
            got = bool(near) and self.miss_first <= 0
            self.miss_first -= 1
            return sim.Reply({"ok": True, "holding": got,
                              "report": {"contact": True, "met_at_cm": 0.1, "jaws_gap_cm": 2.0 if got else 0.0}})
        if name == "park":
            return sim.Reply({"ok": self.park_ok, "error": None if self.park_ok else "No park pose yet."})
        return sim.Reply({"ok": True, "tilt": 0})

    def decide(self, img, text, category):
        self.asked.append(text)
        cands = [{"mask": sim.outline(self.cam, p), "target_score": 0.9, "sam_score": 0.9}
                 for n, p in self.things.items() if n == text]
        return {"status": "GO" if cands else "NOT_FOUND", "candidates": cands, "chosen": None}

    def names(self):
        return [name for name, _ in self.posts]


class GrabFlowTests(unittest.TestCase):
    def setUp(self):
        quiet(self)
        self.cap = sim.cylinder(4, 30, 3, 1.5)
        self.table = Table(self, {"bottle cap": self.cap, "far cup": sim.cylinder(-30, 60, 6, 8)})
        self.a = G.options(arm_url="http://arm.invalid")

    def test_a_small_thing_is_grabbed_at_its_middle_by_feel(self):
        res = G.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["ok"], res["holding"], res["attempts"]), (True, True, 1))
        self.assertEqual(self.table.names(), ["grab"])
        body = self.table.posts[0][1]
        self.assertLess(np.hypot(body["x"] - 4, body["y"] - 30), 0.3)
        self.assertEqual((body["guard"], body["feel_all_the_way"], body["method"]), (True, True, "top"))
        self.assertAlmostEqual(body["object_height"], 1.5, delta=0.5)
        self.assertNotIn("u", body)

    def test_the_old_aim_would_have_missed_it(self):
        mapping = aim.TableMap(self.table.pairs, sim.W, (sim.H, sim.W), lens=False)
        old = mapping.locate(np.array(G.base_pixel(sim.outline(self.table.cam, self.cap))))
        self.assertGreater(np.hypot(old[0] - 4, old[1] - 30), self.table.tolerance)

    def test_a_miss_makes_it_look_again_and_grab_again(self):
        self.table.miss_first = 1
        res = G.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"]), (True, 2))
        self.assertEqual(self.table.names(), ["grab", "release", "park", "grab"])
        self.assertEqual(self.table.asked.count("bottle cap"), 2)
        self.assertIn("Looking again (try 2 of 2)", sys.stdout.getvalue())

    def test_without_a_park_pose_it_pulls_back_instead(self):
        self.table.miss_first, self.table.park_ok = 1, False
        G.find_and_grab(self.a, "bottle cap")
        self.assertEqual(self.table.names(), ["grab", "release", "park", "goto", "grab"])
        back = self.table.posts[3][1]
        self.assertLess(np.hypot(back["x"], back["y"] - 4.2), np.hypot(4, 30 - 4.2) - 10)

    def test_it_gives_up_after_the_allowed_tries_and_says_why(self):
        self.table.miss_first = 9
        res = G.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"]), (False, 2))
        self.assertEqual(self.table.names().count("grab"), 2)
        self.assertIn("aim a little off", res["why"])
        self.a.tries = 1
        self.table.posts.clear()
        G.find_and_grab(self.a, "bottle cap")
        self.assertEqual(self.table.names(), ["grab"])

    def test_no_second_try_after_a_refusal_and_no_motion_in_dry_mode_or_out_of_reach(self):
        self.table.status["refuse"] = "Can't reach x=4.0, y=30.0"
        res = G.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["ok"], res["holding"], res["error"]), (False, False, "Can't reach x=4.0, y=30.0"))
        self.assertEqual(self.table.names(), ["grab"])
        del self.table.status["refuse"]
        for change in ({"dry": True}, {}):
            self.table.posts.clear()
            G.find_and_grab(G.options(arm_url="http://arm.invalid", **change), "bottle cap" if change else "far cup")
            self.assertEqual(self.table.posts, [])
        self.table.status["read_only"] = True
        self.assertIsNone(G.find_and_grab(self.a, "bottle cap"))
        self.assertEqual(self.table.posts, [])

    def test_reach_out_is_only_added_for_an_old_gripper_camera_setup_or_on_request(self):
        target = G.locate(self.a, "bottle cap")
        for hand, asked, extra in (({"used": True, "lean_aware": True}, None, 0.0),
                                   ({"used": True, "lean_aware": False}, None, 1.0),
                                   ({"used": False, "lean_aware": False}, None, 0.0),
                                   ({"used": True, "lean_aware": True}, -0.5, -0.5)):
            self.table.posts.clear()
            self.a.reach_out = asked
            with patch.object(G, "hand_correct", lambda a, text, x, y, h, hand=hand: (x, y, hand)):
                G.grab_target(self.a, target)
            body = self.table.posts[-1][1]
            moved = np.hypot(body["x"], body["y"] - 4.2) - np.hypot(target["x"], target["y"] - 4.2)
            self.assertAlmostEqual(moved, extra, delta=0.02)

    def test_what_went_wrong_is_read_from_where_the_arm_stopped(self):
        small = {"height": 2.0, "flat": False}
        self.assertIn("on top of the object", G.why_missed({"contact": True, "met_at_cm": 2.1}, small))
        self.assertIn("above the object", G.why_missed({"contact": True, "met_at_cm": 5.5}, small))
        self.assertIn("beside it", G.why_missed({"contact": True, "met_at_cm": 0.2}, small))
        self.assertIn("without feeling the table", G.why_missed({"contact": False}, small))
        self.assertIn("lost it", G.why_missed({"contact": True, "met_at_cm": 0.2, "slipped": True}, small))
        tall = {"height": 12.0, "flat": False}                # stopping on a tall thing's top is how it should be
        self.assertIn("beside it", G.why_missed({"contact": True, "met_at_cm": 6.0}, tall))


class SharpenTests(unittest.TestCase):
    def setUp(self):
        quiet(self)
        self.cam = sim.FixedCamera()
        self.cube = sim.box(-4, 14, 1.2, 1.2, 1.2)
        self.img = Image.new("RGB", (sim.W, sim.H))
        mask = sim.outline(self.cam, self.cube)
        ys, xs = np.nonzero(mask)
        self.cand = {"mask": mask, "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], "prompts": {"cube"},
                     "target_score": 0.8}

    def second_look(self, points):
        """A pretend SAM 3 that outlines `points` in whatever enlarged cut-out it is shown."""
        seen = {}

        def sam(big, prompts):
            seen.update(size=big.size, prompts=prompts)
            crop = self.crop
            zoom = big.size[0] / (crop[2] - crop[0])
            return [{"mask": sim.outline(self.cam, points, zoom, crop[:2], big.size), "sam_score": 0.9}]
        return sam, seen

    def test_small_outline_is_replaced_by_a_sharper_one_of_the_same_thing(self):
        x0, y0, x1, y1 = self.cand["box"]
        pad = max(40.0, 0.8 * max(x1 - x0, y1 - y0))
        self.crop = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
        sam, seen = self.second_look(self.cube)
        with patch.object(G.ts, "sam3_candidates", sam):
            sharp = G.sharpen(self.img, self.cand)
        self.assertEqual(seen["prompts"], ["cube"])
        self.assertGreater(sharp["fine"]["zoom"], 3.0)
        self.assertEqual(sharp["fine"]["offset"], self.crop[:2])
        self.assertNotIn("fine", self.cand)                  # the first outline is left as it was
        mapping = aim.TableMap(sim.boards(self.cam), sim.W, (sim.H, sim.W))
        fp = G.footprint_of(sharp, mapping, sim.W)
        self.assertLess(np.hypot(fp["center"][0] + 4, fp["center"][1] - 14), 0.12)

    def test_a_second_look_that_shows_something_else_is_ignored(self):
        x0, y0, x1, y1 = self.cand["box"]
        pad = max(40.0, 0.8 * max(x1 - x0, y1 - y0))
        self.crop = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
        sam, _ = self.second_look(sim.box(-1.5, 15, 1.2, 1.2, 1.2))          # a neighbour
        with patch.object(G.ts, "sam3_candidates", sam):
            self.assertIs(G.sharpen(self.img, self.cand), self.cand)
        with patch.object(G.ts, "sam3_candidates", Mock(side_effect=RuntimeError("out of memory"))):
            self.assertIs(G.sharpen(self.img, self.cand), self.cand)

    def test_big_things_and_no_zoom_skip_the_second_look(self):
        never = Mock(side_effect=AssertionError("no second look expected"))
        with patch.object(G.ts, "sam3_candidates", never):
            self.assertIs(G.sharpen(self.img, self.cand, argparse.Namespace(no_zoom=True)), self.cand)
            big = {"mask": np.ones((sim.H, sim.W), bool), "box": [100, 100, 700, 600], "prompts": {"box"}}
            self.assertIs(G.sharpen(self.img, big), big)


class FeelFromTests(unittest.IsolatedAsyncioTestCase):
    """The arm starts feeling its way down above the object's top, not at a fixed height."""

    async def feel_from(self, object_h, above=22.0):
        _, reach = controller()
        r = reach.Reacher.__new__(reach.Reacher)
        r.arm = types.SimpleNamespace(move=AsyncMock(), close_gripper=AsyncMock(return_value=(True, 40.0)),
                                      gripper_still_holding=lambda: True)
        lines = []
        r.pose_for = lambda x, y, heights, tilt=None: ([{}, {}], 0)
        r.plan_swing = lambda q, h: ["swing"]
        r.plan_line = lambda x, y, h, **k: lines.append(h) or ["line"]
        r.duration = lambda path: 1.0
        r.follow, r.lift = AsyncMock(), AsyncMock()
        r.descend_guarded = AsyncMock(return_value={"contact": True, "height_cm": 0.1, "final_cm": 0.4})
        r.state = lambda: (1.0, 20.0, 0.4, 0.0)
        report = {}
        self.assertTrue(await r.grab(1.0, 20.0, None, above, object_h=object_h, guard=True, report=report))
        r.descend_guarded.assert_awaited_once_with(1.0, 20.0, reach.GUARD_FLOOR_CM)
        return lines, report

    async def test_tall_object_is_felt_for_from_above_its_top(self):
        lines, report = await self.feel_from(12.0)
        self.assertEqual((lines, report["feel_from_cm"]), ([14.0], 14.0))

    async def test_small_or_unknown_objects_keep_the_usual_start(self):
        for h in (3.0, None):
            lines, report = await self.feel_from(h)
            self.assertEqual((lines, report["feel_from_cm"]), ([7.0], 7.0))

    async def test_it_never_starts_above_the_approach_height(self):
        lines, report = await self.feel_from(25.0, above=22.0)
        self.assertEqual((lines, report["feel_from_cm"]), ([], 22.0))


if __name__ == "__main__":
    unittest.main()
