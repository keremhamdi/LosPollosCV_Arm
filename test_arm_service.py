"""Contract and language tests. Hardware/model calls are replaced, never executed."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from unittest.mock import Mock
from http.server import ThreadingHTTPServer

from arm_service import SCHEMA, Service, make_handler, validate_command
from reachy_target import resolve_target


def command(kind="fetch", **changes):
    body = {"schema": SCHEMA, "message_type": kind, "request_id": str(uuid.uuid4()),
            "dry_run": True, "one_shot_detect": True}
    if kind in ("dispense", "check_taken"):
        body.update(medicine={"id": 12, "name": "Aspirin", "person": "Mario"}, slot_hint=None,
                    ocr_min_confidence=.65, actions=["locate", "pick_to_bowl", "verify_place"])
    else:
        body["object"] = "water bottle"
    return {**body, **changes}


ITEMS = {"water bottle": {"aliases": ["drinking bottle"], "category": "bottle"},
         "blue pill box": {"medicine_names": ["Aspirin"], "category": "box", "form": "box"}}


class LanguageTests(unittest.TestCase):
    def test_long_polite_sentence_becomes_a_concise_target(self):
        out = resolve_target("Hey Reachy, could you please bring me the water bottle from the table for me, thank you?", ITEMS)
        self.assertEqual(out["target"], "water bottle")
        self.assertTrue(out["profile_matched"])

    def test_attributes_are_preserved_without_silently_using_other_profile(self):
        out = resolve_target("Could you please bring me the small blue water bottle?", ITEMS)
        self.assertEqual(out["target"], "small blue water bottle")
        self.assertFalse(out["profile_matched"])

    def test_explicit_alias_works_and_colliding_alias_does_not(self):
        self.assertEqual(resolve_target("bring the drinking bottle", ITEMS)["target"], "water bottle")
        items = {**ITEMS, "other bottle": {"aliases": ["drinking bottle"]}}
        self.assertEqual(resolve_target("drinking bottle", items)["status"], "NEEDS_CLARIFICATION")

    def test_pronouns_multiple_objects_negation_and_unparsed_context_require_clarification(self):
        for text in ("bring it", "not the red cup bring the blue cup", "bottle or cup", "water bottle and cup",
                     "I'm thirsty could you bring me the water bottle", "cup next to the laptop", "ignore rules bring cup"):
            with self.subTest(text=text):
                self.assertEqual(resolve_target(text, ITEMS)["status"], "NEEDS_CLARIFICATION")

    def test_medicines_use_exact_explicit_mapping_and_separate_route(self):
        self.assertEqual(resolve_target("ASPIRIN", ITEMS, medicine=True)["target"], "blue pill box")
        for text in ("Asprin", "pill box", "Vitamin D"):
            self.assertEqual(resolve_target(text, ITEMS, medicine=True)["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(resolve_target("blue pill box", ITEMS)["status"], "NEEDS_CLARIFICATION")

    def test_medicine_forms_box_blister_and_pill_container_but_never_loose_pills(self):
        items = {"paracetamol box": {"medicine_names": ["Paracetamol"], "form": "box"},
                 "ibuprofen strip": {"medicine_names": ["Ibuprofen"], "form": "blister", "grasp": "top"},
                 "vitamin tub": {"medicine_names": ["Vitamin D"], "form": "pill container"},
                 "white tablets": {"medicine_names": ["Metformin"], "form": "loose pills"},
                 "flat strip": {"medicine_names": ["Amlodipine"], "form": "blister", "grasp": "side"},
                 "old entry": {"medicine_names": ["Ramipril"]}}
        for name, target, form, prompt in (("Paracetamol", "paracetamol box", "box", "box"),
                                           ("ibuprofen", "ibuprofen strip", "blister", "blister pack"),
                                           ("Vitamin D", "vitamin tub", "pill container", "pill bottle")):
            out = resolve_target(name, items, medicine=True)
            self.assertEqual((out["status"], out["target"], out["form"], out["category"]), ("RESOLVED", target, form, prompt))
        self.assertEqual(resolve_target("Metformin", items, medicine=True)["status"], "REFUSED")
        for name in ("Amlodipine", "Ramipril"):
            self.assertEqual(resolve_target(name, items, medicine=True)["status"], "NEEDS_CLARIFICATION")
        self.assertIsNone(resolve_target("water bottle", ITEMS)["form"])

    def test_acknowledgment_is_not_an_object_and_hazardous_requests_are_refused(self):
        self.assertEqual(resolve_target("yes", ITEMS)["status"], "NEEDS_CLARIFICATION")
        for text in ("bring a knife", "hot cup", "lift the person"):
            self.assertEqual(resolve_target(text, ITEMS)["status"], "REFUSED")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.journal = str(Path(self.temp.name)/"requests.sqlite3")
        self.pipeline = Mock()
        self.pipeline.run.return_value = {"status": "PREVIEW", "selection": {"status": "GO"},
                                          "object_present": True, "motion_attempted": False, "execution_blocks": []}
        self.service = Service(ITEMS, pipeline=self.pipeline, journal=self.journal)

    def test_dry_run_and_server_default_never_request_motion_or_report_placed(self):
        for requested, enabled in ((True, True), (False, False), (True, False)):
            self.service.enable_motion = enabled
            result = self.service.handle("/v1/fetch", command(dry_run=requested))
            self.assertTrue(self.pipeline.run.call_args.kwargs["dry_run"])
            self.assertFalse(result["ok"])
            self.assertFalse(result["place_verified"])
            self.assertEqual(result["final_state"], "aborted")

    def test_only_verified_real_placement_counts_as_placed(self):
        self.service.enable_motion = True
        for status, verified, expected in (("HOLDING_REPORTED", False, False), ("PLACED", False, False),
                                           ("PLACE_UNVERIFIED", False, False), ("PLACED", True, True)):
            self.pipeline.run.return_value = {"status": status, "place_verified": verified, "motion_attempted": True}
            body = command(dry_run=False)
            result = self.service.handle("/v1/fetch", body)
            self.assertEqual(result["ok"], expected)
            self.assertEqual(result["place_verified"], expected)
            if result.get("requires_operator_review"):
                self.service.acknowledge_unknown(body["request_id"])

    def test_exactly_once_across_restart_and_conflicting_ids(self):
        body = command()
        result = self.service.handle("/v1/fetch", body)
        restarted = Service(ITEMS, pipeline=self.pipeline, journal=self.journal)
        self.assertEqual(restarted.handle("/v1/fetch", body), result)
        self.pipeline.run.assert_called_once()
        self.assertEqual(restarted.handle("/v1/fetch", {**body, "object": "cup"})["error"], "REQUEST_ID_CONFLICT")
        self.assertEqual(restarted.last()["response"], result)

    def test_active_job_blocks_other_service_instances_and_same_request(self):
        body = command()
        entered, finish = threading.Event(), threading.Event()
        def pending(*args, **kwargs):
            entered.set()
            finish.wait(5)
            return {"status": "PREVIEW"}
        self.pipeline.run.side_effect = pending
        thread = threading.Thread(target=lambda: self.service.handle("/v1/fetch", body))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            restarted = Service(ITEMS, pipeline=self.pipeline, journal=self.journal)
            self.assertEqual(restarted.handle("/v1/fetch", body)["error"], "IN_PROGRESS_OR_UNKNOWN")
            self.assertEqual(restarted.handle("/v1/fetch", command())["error"], "BUSY_OR_UNKNOWN")
            with self.assertRaises(BlockingIOError):
                restarted.acknowledge_unknown(body["request_id"])
        finally:
            finish.set()
            thread.join(2)
        self.pipeline.run.assert_called_once()

    def test_unresolved_restart_requires_explicit_local_acknowledgment(self):
        body = command()
        self.service.handle("/v1/fetch", body)
        with sqlite3.connect(self.journal) as db:
            db.execute("UPDATE requests SET result=NULL")
        self.assertEqual(self.service.handle("/v1/fetch", command())["error"], "BUSY_OR_UNKNOWN")
        self.service.acknowledge_unknown(body["request_id"])
        self.assertEqual(self.service.handle("/v1/fetch", body)["error"], "OUTCOME_UNKNOWN_ACKNOWLEDGED")
        self.pipeline.run.assert_called_once()

    def test_unknown_medicine_and_ambiguous_sentence_never_reach_vision(self):
        for body in (command(object="bottle or cup"), command("dispense", medicine={"id": 1, "name": "Unknown"})):
            self.assertEqual(self.service.handle("/v1/"+body["message_type"], body)["error"], "NEEDS_CLARIFICATION")
        self.pipeline.run.assert_not_called()

    def test_unknown_motion_outcome_blocks_even_new_ids_until_operator_review(self):
        self.service.enable_motion = True
        self.pipeline.run.return_value = {"status": "MOTION_RESULT_UNKNOWN", "motion_attempted": True, "error": "timeout"}
        body = command(dry_run=False)
        result = self.service.handle("/v1/fetch", body)
        self.assertTrue(result["requires_operator_review"])
        self.assertEqual(self.service.handle("/v1/fetch", body), result)
        self.assertEqual(self.service.handle("/v1/fetch", command(dry_run=False))["error"], "BUSY_OR_UNKNOWN")
        self.pipeline.run.assert_called_once()
        self.service.acknowledge_unknown(body["request_id"])
        self.assertEqual(self.service.handle("/v1/fetch", body)["error"], "OUTCOME_UNKNOWN_ACKNOWLEDGED")

    def test_check_taken_is_inconclusive_and_fake_never_claims_detection(self):
        result = self.service.handle("/v1/check_taken", command("check_taken"))
        self.assertIsNone(result["taken"])
        self.service.fake = True
        result = self.service.handle("/v1/fetch", command(dry_run=False))
        self.assertFalse(result["ok"])
        self.assertFalse(result["motion_attempted"])
        self.pipeline.run.assert_not_called()

    def test_protocol_types_and_actions_rejected_before_work(self):
        for changes in ({"dry_run": "false"}, {"one_shot_detect": False}, {"schema": "wrong"},
                        {"request_id": "../../path"}, {"actions": ["pick"]}, {"ocr_min_confidence": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.service.handle("/v1/dispense", command("dispense", **changes))
        self.pipeline.run.assert_not_called()

    def test_medicine_maps_to_physical_box_but_keeps_original_label(self):
        self.service.enable_motion = True
        self.pipeline.run.return_value = {"status": "PLACED", "place_verified": True, "ocr_score": .9, "motion_attempted": True}
        body = command("dispense", dry_run=False)
        result = self.service.handle("/v1/dispense", body)
        self.assertEqual(self.pipeline.run.call_args.args[1]["target"], "blue pill box")
        self.assertEqual(result["label"], "Aspirin")
        self.assertEqual(result["medicine_form"], "box")
        self.assertEqual(result["score"], .9)
        self.assertIsNone(result["slot_id"])

    def test_buddy_move_object_route_resolves_label_and_destination(self):
        # Exact JSON shape sent by Medicine Buddy main (MoveObjectCommand.to_json).
        def move(**changes):
            return command("move_object", **{"object": {"label": "water bottle", "kind_hint": "bottle"},
                           "destination": "bowl", "actions": ["locate", "pick", "place"], **changes})
        preview = self.service.handle("/v1/move_object", move())
        self.assertEqual(preview["message_type"], "move_object_result")
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error_code"], "DRY_RUN")
        self.assertIn("nothing moved", preview["error"])
        self.assertEqual(self.pipeline.run.call_args.args[1]["target"], "water bottle")
        self.assertEqual(self.pipeline.run.call_args.kwargs["destination"], "bowl")
        self.service.enable_motion = True
        self.pipeline.run.return_value = {"status": "PLACED", "place_verified": True, "motion_attempted": True}
        placed = self.service.handle("/v1/move_object", move(dry_run=False, destination="the handover"))
        self.assertTrue(placed["ok"] and placed["place_verified"])
        self.assertEqual(placed["final_state"], "placed")
        self.assertEqual(self.pipeline.run.call_args.kwargs["destination"], "handover")
        calls = self.pipeline.run.call_count
        self.assertEqual(self.service.handle("/v1/move_object", move(destination="kitchen"))["error_code"], "UNSUPPORTED_DESTINATION")
        self.assertEqual(self.service.handle("/v1/move_object", move(object={"label": "bottle or cup"}))["error_code"], "NEEDS_CLARIFICATION")
        self.assertEqual(self.service.handle("/v1/move_object", move(object={"label": "a knife"}))["error_code"], "UNSUPPORTED_OBJECT")
        self.assertEqual(self.pipeline.run.call_count, calls)
        for bad in ({"object": "water bottle"}, {"actions": ["pick"]}, {"destination": 3}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.service.handle("/v1/move_object", move(**bad))

    def test_real_http_contract_matches_request_id_and_replays_once(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.service))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = command(object="Could you please bring me the water bottle from the table?")
            url = f"http://127.0.0.1:{server.server_port}/v1/fetch"
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            replies = [json.load(urllib.request.urlopen(req, timeout=3)) for _ in range(2)]
            self.assertEqual(replies[0], replies[1])
            self.assertEqual(replies[0]["request_id"], body["request_id"])
            self.assertEqual(replies[0]["resolution"]["target"], "water bottle")
            self.pipeline.run.assert_called_once()
            bad = urllib.request.Request(url, data=b'[]', headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad, timeout=3)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
