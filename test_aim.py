"""The opt-in careful aim (aim.py, precise_grab.py), tested on a pretend table (sim_scene.py).
No arm, camera, model or network. The default pickup is covered by test_grab_camera_mode.py."""
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
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import aim
import dispense_server as ds
import grab_text as G
import precise_grab as P
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
        self.a = P.options(arm_url="http://arm.invalid", no_hand=False)
        self.rng = np.random.default_rng(0)
        self.sees = True

        def post(url, json=None, **_):
            name = url.rsplit("/", 1)[1]
            return sim.Reply(self.world.goto(json["x"], json["y"], json["height"], json.get("tilt"))
                             if name == "goto" else {"ok": True})

        def find(a, text, near=None, fits=None, tries=3):
            p = self.world.pixel() if self.sees else None
            if p is None or (fits is not None and not fits(np.array(p))):
                return None, "picture", None
            return (p[0] + self.rng.normal(0, 0.7), p[1] + self.rng.normal(0, 0.7)), "picture", {"mask": None}

        self.real_find = P.find_hand
        answers = itertools.cycle(["", "3", ""])             # jaws around it; 3 cm tall; arm may move
        for target, name, value in ((G, "HAND_FILE", os.path.join(folder.name, "hand.json")),
                                    (G.requests, "post", post), (G, "find_in_hand", find), (P, "find_hand", find),
                                    (G.requests, "get", lambda url, **_: sim.Reply({"tip": [2.0, 20.0, 0.4]})),
                                    (G, "save_hand_view", lambda *a, **k: None)):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("builtins.input", lambda *a: next(answers))
        p.start()
        self.addCleanup(p.stop)
        P._told.clear()

    def setup(self, **drop):
        self.world.obj = np.array((2.0, 20.0, 3.0))          # the object the jaws were put around
        P.hand_calibrate(self.a, "thing")
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
            x, y, info = P.hand_align(self.a, "thing", spot[0] + 1.5, spot[1] - 1.2, spot[2] + 0.7)
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
            seen, geometry = (P.lean_shift(c, J, 2.0, 20.0, tilt) for c in (cal, {**cal, "lean": None}))
            self.assertLess(np.linalg.norm(seen - geometry), 0.25 * np.linalg.norm(geometry) + 3)
        np.testing.assert_allclose(P.lean_shift(cal, J, 2.0, 20.0, 0.0), 0)

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
        x, y, info = P.hand_align(self.a, "thing", 3.0, 19.0, 3.0)
        self.assertTrue(all(m[3] > 30 for m in self.world.moves))    # ... so it can't look as in the setup,
        self.assertLess(np.hypot(x - 2.0, y - 20.0), 0.6)            # and allows for the lean, past what was measured
        self.world.lean_from = 12.0                           # a lean the setup did measure: spot on
        self.world.moves.clear()
        x, y, info = P.hand_align(self.a, "thing", 3.0, 19.0, 3.0)
        self.assertTrue(all(0 < m[3] <= 30 for m in self.world.moves))
        self.assertLess(np.hypot(x - 2.0, y - 20.0), 0.3)

    def test_not_seeing_it_keeps_the_fixed_cameras_aim(self):
        self.setup()
        self.sees = False
        x, y, info = P.hand_align(self.a, "thing", 5.0, 25.0, 3.0)
        self.assertEqual((x, y, info["used"]), (5.0, 25.0, False))
        os.remove(G.HAND_FILE)                               # never set up: nothing moves
        self.world.moves.clear()
        self.assertEqual(P.hand_align(self.a, "thing", 5.0, 25.0, 3.0), (5.0, 25.0, {"used": False, "lean_aware": False}))
        self.assertEqual(self.world.moves, [])

    def test_it_keeps_looking_until_a_look_shows_the_object_under_the_jaws(self):
        self.setup()
        self.world.obj = np.array((-8.0, 16.0, 3.0))
        x, y, info = P.hand_align(self.a, "thing", -6.0, 14.5, 3.0)
        self.assertLess(np.hypot(x + 8.0, y - 16.0), 0.2)
        self.assertIn("aligned to within 0.15 cm", sys.stdout.getvalue())
        last = [line for line in sys.stdout.getvalue().splitlines() if "off by" in line][-1]
        self.assertLessEqual(float(last.split("off by ")[1].split(" cm")[0]), 0.15)   # the last look confirmed it

    def test_a_blob_of_quite_another_size_is_not_the_object(self):
        seen = []

        def decide(img, text, category):
            mask = np.zeros((720, 1280), bool)
            mask[330:342, 620:632] = True                    # a 12 px glint right where the object should be
            seen.append(text)
            return {"candidates": [{"mask": mask, "target_score": 0.9}]}
        fits_cap = lambda c, area=None: area is None or area > 500     # a 1.8 cm cap is far bigger than that
        with patch.object(G.ts, "decide", decide), patch("time.sleep", lambda s: None), \
                patch.object(G, "hand_frame", lambda a: Image.new("RGB", (1280, 720))):
            p, _, best = self.real_find(self.a, "black cap", (626, 336), fits_cap, 1)
        self.assertIsNone(p)
        self.assertIn("small object", seen)                  # it also looked for any small object, of any colour

    def test_something_far_from_where_the_object_must_be_is_not_taken_for_it(self):
        self.setup()
        self.world.obj = np.array((-8.0, 16.0, 3.0))
        x, y, info = P.hand_align(self.a, "thing", -8.0 + 9.5, 16.0, 3.0)     # 9.5 cm off: can't be the object
        self.assertEqual((x, y, info["used"]), (1.5, 16.0, False))

    def test_an_older_arm_page_that_does_not_say_the_lean_still_works(self):
        self.world.says_lean = False
        cal = self.setup()
        self.assertNotIn("tilt", cal)
        self.world.obj = np.array((-8.0, 16.0, 3.0))
        x, y, info = P.hand_align(self.a, "thing", -7.0, 15.0, 3.0)
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
        self.arm_short = 0.0                                 # cm the real arm stops short of where it is sent
        self.miss_report = {"contact": True, "met_at_cm": 0.1, "jaws_gap_cm": 0.0}   # what a miss looks like
        self.park_ok = True
        jpeg = io.BytesIO()
        Image.new("RGB", (sim.W, sim.H)).save(jpeg, format="JPEG")
        self.jpeg = jpeg.getvalue()
        for target, name, value in ((G.requests, "get", self.get), (G.requests, "post", self.post),
                                    (G.ts, "decide", self.decide), (G, "show_view", lambda *a, **k: None),
                                    (G, "HAND_FILE", "no/such/hand.json"), (P, "AIM_FILE", "no/such/aim.json"),
                                    (P, "_session", {"reach": 0.0, "ways": {}})):
            p = patch.object(target, name, value)
            p.start()
            test.addCleanup(p.stop)
        P._maps.clear()

    def get(self, url, **_):
        return sim.Reply({**self.status, "pairs": self.pairs, "size": sim.W}, self.jpeg)

    def post(self, url, json=None, **_):
        name = url.rsplit("/", 1)[1]
        self.posts.append((name, json))
        if name == "grab":
            if self.status.get("refuse"):
                return sim.Reply({"ok": False, "error": self.status["refuse"]})
            # where the jaws really come down: this arm may stop short of where it is sent
            r = np.array([json["x"], json["y"] - sim.BASE[1]])
            real = np.array([json["x"], json["y"]]) - self.arm_short * r / np.linalg.norm(r)
            off = {n: float(np.hypot(*(p[:, :2].mean(axis=0) - real))) for n, p in self.things.items()}
            near = [n for n, d in off.items() if d <= self.tolerance]
            got = bool(near) and self.miss_first <= 0
            self.miss_first -= 1
            report = self.miss_report
            under_a_fingertip = [n for n, d in off.items() if self.tolerance < d <= 3.0]
            if not got and not near and under_a_fingertip:    # a little off: a fingertip comes down on its top
                report = {"contact": True, "jaws_gap_cm": 0.0,
                          "met_at_cm": float(self.things[under_a_fingertip[0]][:, 2].max())}
            return sim.Reply({"ok": True, "holding": got, "report": (
                {"contact": True, "met_at_cm": 0.1, "jaws_gap_cm": 2.0} if got else report)})
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
        self.a = P.options(arm_url="http://arm.invalid")

    def test_a_small_thing_is_grabbed_at_its_middle_and_closed_on_low(self):
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["ok"], res["holding"], res["attempts"]), (True, True, 1))
        self.assertEqual(self.table.names(), ["grab"])
        body = self.table.posts[0][1]
        self.assertLess(np.hypot(body["x"] - 4, body["y"] - 30), 0.3)
        self.assertEqual((body["method"], body["height"]), ("top", 0.8))
        self.assertNotIn("guard", body)                      # small: no contact stop, so no backing up before closing
        self.assertNotIn("feel_all_the_way", body)           # never a search below the table
        self.assertAlmostEqual(body["object_height"], 1.5, delta=0.5)
        self.assertNotIn("u", body)

    def test_a_small_thing_missed_without_touching_is_tried_lower_and_big_things_keep_the_contact_stop(self):
        self.table.miss_first = 1
        self.table.miss_report = {"jaws_gap_cm": 0.0}        # what the arm reports without its contact stop
        P.find_and_grab(self.a, "bottle cap")
        first, second = (body for name, body in self.table.posts if name == "grab")
        self.assertEqual((first["height"], "guard" in first), (0.8, False))
        self.assertEqual((second.get("guard"), second.get("feel_all_the_way")), (True, True))   # low thing: now by touch
        tall = P.locate(self.a, "bottle cap")
        tall["height"] = 9.0
        self.table.posts.clear()
        P.grab_target(self.a, tall)
        self.assertTrue(self.table.posts[-1][1]["guard"])

    def test_closing_height_is_the_usual_plan_except_lower_for_small_and_flat_things(self):
        plan = lambda h, flat=False, **o: P.grab_heights(P.options(**o), h, flat)
        self.assertEqual([plan(h)[0] for h in (3.0, 5.0, 8.0, 20.0, None)], [1.5, 2.5, 4.0, 6.0, 1.5])   # as the default pickup
        self.assertEqual([plan(h)[0] for h in (1.2, 2.0, 2.6)], [0.8, 1.0, 1.3])        # small: lower than its 1.5 cm
        self.assertEqual(plan(None, flat=True)[0], 0.8)
        self.assertEqual([plan(h, min_close=1.5)[0] for h in (1.2, 2.0)], [1.5, 1.5])   # --min-close 1.5: exactly the usual plan
        self.assertEqual(plan(2.0, min_close=0.1)[0], 1.0)
        self.assertEqual(plan(None, flat=True, min_close=0.1)[0], 0.5)                  # never under what the arm page takes
        self.assertEqual((plan(2.0, close=3.0), plan(12.0)[1]), ((3.0, 14.0), 17.0))

    def test_the_old_aim_would_have_missed_it(self):
        mapping = aim.TableMap(self.table.pairs, sim.W, (sim.H, sim.W), lens=False)
        old = mapping.locate(np.array(G.base_pixel(sim.outline(self.table.cam, self.cap))))
        self.assertGreater(np.hypot(old[0] - 4, old[1] - 30), self.table.tolerance)

    def test_a_miss_makes_it_look_again_and_grab_again(self):
        self.table.miss_first = 1
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"]), (True, 2))
        self.assertEqual(self.table.names(), ["grab", "release", "park", "grab"])
        self.assertEqual(self.table.asked.count("bottle cap"), 2)
        self.assertIn("Looking again (try 2 of 2)", sys.stdout.getvalue())

    def test_after_a_stop_in_the_air_the_second_try_goes_down_without_the_contact_stop(self):
        self.table.miss_first, self.a.feel_small = 1, True
        self.table.miss_report = {"contact": True, "met_at_cm": 5.2, "jaws_gap_cm": 0.0}   # the cap is 1.5 cm tall
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertTrue(res["holding"])
        first, second = (body for name, body in self.table.posts if name == "grab")
        self.assertEqual((first.get("guard"), first["height"]), (True, 0.8))
        self.assertNotIn("guard", second)                    # so it can't stop early again
        self.assertEqual(second["height"], 0.8)
        self.assertIn("above the object", sys.stdout.getvalue())

    def test_after_closing_on_nothing_without_touching_anything_the_second_try_goes_lower(self):
        self.table.miss_first, self.a.feel_small = 1, True
        self.table.miss_report = {"contact": False, "closed_at_cm": 0.8, "jaws_gap_cm": 0.0}
        P.find_and_grab(self.a, "bottle cap")
        first, second = (body for name, body in self.table.posts if name == "grab")
        self.assertEqual((first["height"], second.get("guard"), second.get("feel_all_the_way")), (0.8, True, True))
        self.assertIn("closed 0.8 cm above the table without touching anything", sys.stdout.getvalue())
        self.a.close = 2.0                                   # a height you gave is never changed
        self.table.posts.clear()
        self.table.miss_first = 1
        P.find_and_grab(self.a, "bottle cap")
        self.assertEqual([body["height"] for name, body in self.table.posts if name == "grab"], [2.0, 2.0])

    def test_the_gripper_camera_stays_off_unless_asked_for(self):
        with patch.object(P, "hand_align", side_effect=AssertionError("gripper camera not asked for")), \
                patch.object(G, "find_in_hand", side_effect=AssertionError("gripper camera not asked for")):
            P.find_and_grab(self.a, "bottle cap")
        self.assertEqual(self.table.names(), ["grab"])

    def test_flat_and_very_low_things_are_gripped_at_the_table_found_by_touch(self):
        for name, thing in (("coin", sim.cylinder(0, 30, 2.5, 0.2)), ("tablet", sim.cylinder(-6, 24, 1.0, 0.5))):
            self.table.things = {name: thing}
            self.table.posts.clear()
            res = P.find_and_grab(self.a, name)
            body = self.table.posts[0][1]
            self.assertEqual((body.get("guard"), body.get("feel_all_the_way"), res["touch"]), (True, True, True), name)
        for change in ({"no_touch": True}, {"close": 1.0}, {"no_feel": True}):      # each of these keeps to a height
            self.table.posts.clear()
            P.find_and_grab(P.options(arm_url="http://arm.invalid", **change), "tablet")
            self.assertNotIn("feel_all_the_way", self.table.posts[0][1], change)
        self.table.things = {"bottle cap": self.cap}                                 # 1.5 cm tall: planned height as before
        self.table.posts.clear()
        P.find_and_grab(self.a, "bottle cap")
        self.assertNotIn("feel_all_the_way", self.table.posts[0][1])

    def test_standing_or_lying_the_same_command_finds_the_way_down_by_itself(self):
        standing, lying = sim.cylinder(-6, 24, 0.9, 1.4), sim.box(-6, 24, 1.4, 0.9, 0.9)
        how = lambda: [("touch" if b.get("feel_all_the_way") else b["height"]) for n, b in self.table.posts if n == "grab"]
        self.table.things = {"small white cylinder": standing}
        self.assertTrue(P.find_and_grab(self.a, "small white cylinder")["holding"])
        first_standing = how()
        self.table.things, self.table.posts = {"small white cylinder": lying}, []
        self.assertTrue(P.find_and_grab(self.a, "small white cylinder")["holding"])
        self.assertEqual((first_standing, how()), ([0.8], ["touch"]))             # no height typed, no flag changed
        # misjudged or just missed: the second try goes down the other way, and what works is remembered
        self.table.miss_first, self.table.posts = 1, []
        self.table.miss_report = {"contact": True, "met_at_cm": 0.0, "jaws_gap_cm": 0.0}
        self.assertTrue(P.find_and_grab(self.a, "small white cylinder")["holding"])
        self.assertEqual(how(), ["touch", 0.5])
        self.table.posts = []
        P.find_and_grab(self.a, "small white cylinder")
        self.assertNotIn("touch", how())                                          # lying ones now start with what worked
        self.table.things, self.table.posts, self.table.miss_first = {"small white cylinder": standing}, [], 1
        self.table.miss_report = {"jaws_gap_cm": 0.0}
        P.find_and_grab(self.a, "small white cylinder")
        self.assertEqual(how(), [0.8, "touch"])                                   # a height that got nothing -> by touch

    def test_putting_down_lets_go_where_it_was_gripped_not_from_a_fixed_height(self):
        with patch.object(P, "_held", {}):
            P.find_and_grab(self.a, "bottle cap")                # gripped 0.8 cm up
            self.table.posts.clear()
            P.put_down(self.a, -12.0, 25.0)
            self.assertEqual(self.table.posts, [("place", {"x": -12.0, "y": 25.0, "height": 1.3, "above": 14.0})])
            self.table.posts.clear()
            P.put_down(self.a, -12.0, 25.0)                      # nothing held any more: the usual height
            self.assertEqual(self.table.posts[0][1]["height"], 5.0)
            P._held.update(close=4.0, above=17.0)                # a tall thing gripped 4 cm up: not pressed into the table
            self.table.posts.clear()
            P.put_down(self.a, 0.0, 20.0)
            self.assertEqual((self.table.posts[0][1]["height"], self.table.posts[0][1]["above"]), (4.5, 17.0))
            P._held.update(close=4.0, above=17.0)
            self.a.put_height_given, self.a.put_height = True, 2.0
            self.table.posts.clear()
            P.put_down(self.a, 0.0, 20.0)
            self.assertEqual(self.table.posts[0][1]["height"], 2.0)

    def test_without_a_park_pose_it_pulls_back_instead(self):
        self.table.miss_first, self.table.park_ok = 1, False
        P.find_and_grab(self.a, "bottle cap")
        self.assertEqual(self.table.names(), ["grab", "release", "park", "goto", "grab"])
        back = self.table.posts[3][1]
        self.assertLess(np.hypot(back["x"], back["y"] - 4.2), np.hypot(4, 30 - 4.2) - 10)

    def test_it_gives_up_after_the_allowed_tries_and_says_why(self):
        self.table.miss_first = 9
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"]), (False, 2))
        self.assertEqual(self.table.names().count("grab"), 2)
        self.assertIn("aim a little off", res["why"])
        self.a.tries = 1
        self.table.posts.clear()
        P.find_and_grab(self.a, "bottle cap")
        self.assertEqual(self.table.names(), ["grab"])

    def test_no_second_try_after_a_refusal_and_no_motion_in_dry_mode_or_out_of_reach(self):
        self.table.status["refuse"] = "Can't reach x=4.0, y=30.0"
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["ok"], res["holding"], res["error"]), (False, False, "Can't reach x=4.0, y=30.0"))
        self.assertEqual(self.table.names(), ["grab"])
        del self.table.status["refuse"]
        for change in ({"dry": True}, {}):
            self.table.posts.clear()
            P.find_and_grab(P.options(arm_url="http://arm.invalid", **change), "bottle cap" if change else "far cup")
            self.assertEqual(self.table.posts, [])
        self.table.status["read_only"] = True
        self.assertIsNone(P.find_and_grab(self.a, "bottle cap"))
        self.assertEqual(self.table.posts, [])

    def test_an_arm_that_stops_short_is_found_out_by_the_second_grab_and_remembered(self):
        self.table.arm_short = 1.0
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"], res["reach"]), (True, 2, 1.0))
        first, second = (body for name, body in self.table.posts if name == "grab")
        reach = lambda b: np.hypot(b["x"], b["y"] - 4.2)
        self.assertAlmostEqual(reach(second) - reach(first), 1.0, delta=0.05)
        self.assertIn("Got it with 1 cm extra reach", sys.stdout.getvalue())
        self.table.posts.clear()                             # the next grab starts with what worked
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"], self.table.names()), (True, 1, ["grab"]))
        self.table.arm_short = 0.0                           # ... and if that guess stops working, back to none
        self.table.posts.clear()
        res = P.find_and_grab(self.a, "bottle cap")
        self.assertEqual((res["holding"], res["attempts"], res["reach"]), (True, 2, 0.0))

    def test_a_reach_you_give_or_measured_is_never_second_guessed(self):
        self.table.arm_short, self.a.reach_out = 1.0, 0.0
        res = P.find_and_grab(self.a, "bottle cap")
        grabs = [body for name, body in self.table.posts if name == "grab"]
        self.assertEqual((res["holding"], res["reach"], len(grabs)), (False, None, 2))
        self.assertEqual((grabs[0]["x"], grabs[0]["y"]), (grabs[1]["x"], grabs[1]["y"]))

    def test_aimcheck_measures_the_offset_once_and_every_grab_then_uses_it(self):
        self.table.arm_short = 1.2
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        answers = iter(["", "out 1", "sideways 3", "out 9", "out 0.2", "left 0.3", "right 0.3", ""])
        with patch.object(P, "AIM_FILE", os.path.join(folder.name, "calib", "aim.json")), \
                patch("builtins.input", lambda *a: next(answers)):
            P.aim_check(self.a, "bottle cap")
            moves = [body for name, body in self.table.posts if name == "goto"]
            self.assertEqual(self.table.names()[0], "release")
            self.assertEqual([m["height"] for m in moves[:2]], [14.0, moves[1]["height"]])
            self.assertTrue(2.0 <= moves[1]["height"] <= 4.0)                 # open jaws just above the cap, not on it
            self.assertEqual(P.aim_checks(), [{"at": P.aim_checks()[0]["at"], "out": 1.2, "left": 0.0}])
            out, left = P.aim_offset(4.0, 30.0)
            self.assertAlmostEqual(out, 1.2, delta=0.01)
            self.table.posts.clear()
            res = P.find_and_grab(self.a, "bottle cap")                       # first grab, no guessing
            self.assertEqual((res["holding"], res["attempts"], res["reach"]), (True, 1, None))
            self.assertIn("aim check", sys.stdout.getvalue())
            P.aim_check(self.a, "clear")
            self.assertIsNone(P.aim_offset(4.0, 30.0))

    def test_reach_out_is_only_added_for_an_old_gripper_camera_setup_or_on_request(self):
        target = P.locate(self.a, "bottle cap")
        for hand, asked, extra in (({"used": True, "lean_aware": True}, None, 0.0),
                                   ({"used": True, "lean_aware": False}, None, 1.0),
                                   ({"used": False, "lean_aware": False}, None, 0.0),
                                   ({"used": True, "lean_aware": True}, -0.5, -0.5)):
            self.table.posts.clear()
            self.a.reach_out, self.a.no_hand = asked, False
            with patch.object(P, "hand_align", lambda a, text, x, y, h, w=None, hand=hand: (x, y, hand)):
                P.grab_target(self.a, target)
            body = self.table.posts[-1][1]
            moved = np.hypot(body["x"], body["y"] - 4.2) - np.hypot(target["x"], target["y"] - 4.2)
            self.assertAlmostEqual(moved, extra, delta=0.02)

    def test_what_went_wrong_is_read_from_where_the_arm_stopped(self):
        small = {"height": 2.0, "flat": False}
        self.assertIn("on top of the object", P.why_missed({"contact": True, "met_at_cm": 2.1}, small))
        self.assertIn("above the object", P.why_missed({"contact": True, "met_at_cm": 5.5}, small))
        plan = lambda r, reach=None: P.next_descent({"report": r, "close": 1.0, "reach": reach}, small)
        self.assertEqual(plan({"contact": True, "met_at_cm": 5.5}, 0.0), {"stop": False, "close": None, "reach": None})
        self.assertEqual(plan({"contact": False}, 0.0), {"stop": True, "close": None, "reach": None, "touch": True})
        self.assertEqual(P.next_descent({"report": {"contact": False}, "close": 4.0}, {"height": 8.0, "flat": False}),
                         {"stop": True, "close": 2.0, "reach": None})                  # a taller thing: half as high
        self.assertEqual(P.next_descent({"report": {"contact": True, "met_at_cm": 0.0}, "close": 0.8, "touch": True}, small),
                         {"stop": True, "close": 0.5, "reach": None, "touch": False})  # went to the table: now to a height
        self.assertEqual(plan({"contact": True, "met_at_cm": 2.1}, 0.0), {"stop": True, "close": None, "reach": 1.0})
        self.assertEqual(plan({"contact": True, "met_at_cm": 2.1}, 1.0), {"stop": True, "close": None, "reach": 0.0})
        self.assertEqual(plan({"contact": True, "met_at_cm": 0.2}, 1.0), {"stop": True, "close": None, "reach": None})   # beside it: same again
        self.assertEqual(plan({"contact": True, "met_at_cm": 2.1}), {"stop": True, "close": None, "reach": None})        # reach given or measured
        self.assertIn("beside it", P.why_missed({"contact": True, "met_at_cm": 0.2}, small))
        self.assertIn("without touching anything", P.why_missed({"contact": False}, small))
        self.assertIn("lost it", P.why_missed({"contact": True, "met_at_cm": 0.2, "slipped": True}, small))
        tall = {"height": 12.0, "flat": False}                # stopping on a tall thing's top is how it should be
        self.assertIn("beside it", P.why_missed({"contact": True, "met_at_cm": 6.0}, tall))


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
            sharp = P.sharpen(self.img, self.cand)
        self.assertEqual(seen["prompts"], ["cube"])
        self.assertGreater(sharp["fine"]["zoom"], 3.0)
        self.assertEqual(sharp["fine"]["offset"], self.crop[:2])
        self.assertNotIn("fine", self.cand)                  # the first outline is left as it was
        mapping = aim.TableMap(sim.boards(self.cam), sim.W, (sim.H, sim.W))
        fp = P.footprint_of(sharp, mapping, sim.W)
        self.assertLess(np.hypot(fp["center"][0] + 4, fp["center"][1] - 14), 0.12)

    def test_a_second_look_that_shows_something_else_is_ignored(self):
        x0, y0, x1, y1 = self.cand["box"]
        pad = max(40.0, 0.8 * max(x1 - x0, y1 - y0))
        self.crop = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
        sam, _ = self.second_look(sim.box(-1.5, 15, 1.2, 1.2, 1.2))          # a neighbour
        with patch.object(G.ts, "sam3_candidates", sam):
            self.assertIs(P.sharpen(self.img, self.cand), self.cand)
        with patch.object(G.ts, "sam3_candidates", Mock(side_effect=RuntimeError("out of memory"))):
            self.assertIs(P.sharpen(self.img, self.cand), self.cand)

    def test_big_things_and_no_zoom_skip_the_second_look(self):
        never = Mock(side_effect=AssertionError("no second look expected"))
        with patch.object(G.ts, "sam3_candidates", never):
            self.assertIs(P.sharpen(self.img, self.cand, argparse.Namespace(no_zoom=True)), self.cand)
            big = {"mask": np.ones((sim.H, sim.W), bool), "box": [100, 100, 700, 600], "prompts": {"box"}}
            self.assertIs(P.sharpen(self.img, big), big)


class CommandLineTests(unittest.TestCase):
    """On this branch the careful way is what grab_text.py does; --basic is the pickup as it was."""

    def run_main(self, *flags):
        with patch.object(G, "find_and_grab") as usual, patch.object(P, "find_and_grab") as careful, \
                patch.object(G, "HAND_FILE", self.hand_file), contextlib.redirect_stdout(io.StringIO()) as said:
            G.main(["--text", "bottle cap", *flags])
        return usual, careful, said.getvalue()

    hand_file = "no/such/hand.json"

    def test_the_gripper_camera_is_used_whenever_it_is_set_up(self):
        self.hand_file = __file__                            # any file that exists: it has been set up
        self.assertFalse(self.run_main()[1].call_args.args[0].no_hand)
        self.assertTrue(self.run_main("--no-hand")[1].call_args.args[0].no_hand)
        self.assertTrue(self.run_main("--basic")[0].call_args.args[0].no_hand)        # the basic pickup stays as it was
        self.hand_file = "no/such/hand.json"
        self.assertTrue(self.run_main()[1].call_args.args[0].no_hand)
        self.assertFalse(self.run_main("--hand")[1].call_args.args[0].no_hand)

    def test_basic_is_the_pickup_as_it_was(self):
        usual, careful, said = self.run_main("--basic")
        careful.assert_not_called()
        args, text = usual.call_args.args
        self.assertEqual((text, args.precise, args.reach_out, args.no_hand), ("bottle cap", False, 1.0, True))
        self.assertIn("careful aim is OFF", said)

    def test_default_is_the_careful_way_with_no_fixed_reach_out_unless_asked(self):
        for flags in ((), ("--precise",)):
            usual, careful, said = self.run_main(*flags)
            usual.assert_not_called()
            args, text = careful.call_args.args
            self.assertEqual((text, args.reach_out, args.no_hand, args.tries, args.min_close),
                             ("bottle cap", None, True, 2, 0.8))
            self.assertIn("Careful aim ON", said)
        for flags, reach in ((("--reach-out", "1.5"), 1.5), (("--reach-out=0",), 0.0)):
            _, careful, _ = self.run_main(*flags)
            self.assertEqual(careful.call_args.args[0].reach_out, reach)
        _, careful, _ = self.run_main("--hand", "--tries", "1", "--no-zoom", "--no-lens")
        args = careful.call_args.args[0]
        self.assertEqual((args.no_hand, args.tries, args.no_zoom, args.no_lens), (False, 1, True, True))


class ServerTests(unittest.TestCase):
    """dispense_server.py picks the careful way; with --basic, as before (test_dispense_server.py)."""

    def setUp(self):
        quiet(self)
        self.table = Table(self, {"small bottle": sim.cylinder(-6, 22, 2.5, 5), "bottle cap": sim.cylinder(4, 30, 3, 1.5)})
        self.a = argparse.Namespace(arm_url="http://arm.invalid", meds={"aspirin": "small bottle"}, read_labels=False,
                                    bowl=[-20., 15.], bowl_text="bowl", handover=None, listen_only=False,
                                    bowl_height=8., close=5., above=14., precise=True, tries=2)

    def ask(self, kind="dispense", **changes):
        body = {"schema": ds.SCHEMA, "message_type": kind, "request_id": "r1", "dry_run": False}
        body.update({"medicine": {"name": "Aspirin"}} if kind == "dispense" else {"object": {"label": "bottle cap"}})
        return ds.dispense(self.a, {**body, **changes}, kind)

    def test_medicine_is_picked_at_its_middle_at_a_height_that_fits_it_and_put_in_the_bowl(self):
        out = self.ask()
        self.assertEqual((out["ok"], out["final_state"], out["label"]), (True, "placed", "ASPIRIN"))
        self.assertEqual(self.table.names(), ["grab", "place", "park"])
        grab = self.table.posts[0][1]
        self.assertLess(np.hypot(grab["x"] + 6, grab["y"] - 22), 0.4)
        self.assertEqual(("guard" in grab, "feel_all_the_way" in grab), (False, False))    # small: no contact stop
        self.assertAlmostEqual(grab["height"], 2.5, delta=0.3)                          # half its 5 cm, not the fixed 5 cm
        self.assertEqual(self.table.posts[1][1]["x"], -20.)
        self.assertIn("looks_like:small bottle at x=-6", out["detect_detail"])

    def test_everyday_things_too_and_a_miss_is_tried_again(self):
        self.table.miss_first = 1
        out = self.ask("move_object")
        self.assertEqual((out["ok"], out["final_state"], out["label"]), (True, "placed", "bottle cap"))
        self.assertEqual(self.table.names(), ["grab", "release", "park", "grab", "place", "park"])
        self.assertIn("2 grabs", out["detect_detail"])

    def test_two_misses_are_reported_with_what_they_looked_like_and_nothing_is_placed(self):
        self.table.miss_first = 9
        out = self.ask()
        self.assertEqual((out["ok"], out["final_state"]), (False, "failed"))
        self.assertIn("pick failed: the jaws closed beside it", out["error"])
        self.assertNotIn("place", self.table.names())
        self.assertEqual(self.table.names()[-1], "park")

    def test_dry_run_and_not_found_never_move(self):
        out = self.ask(dry_run=True)
        self.assertEqual((out["ok"], out["final_state"], self.table.posts), (True, "aborted", []))
        self.a.meds = {"aspirin": "green vase"}
        out = self.ask()
        self.assertEqual((out["ok"], out["detect_detail"], self.table.posts), (False, "not_found", []))


if __name__ == "__main__":
    unittest.main()
