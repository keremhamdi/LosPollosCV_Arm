"""Actual workflow orchestration with synthetic images and fake controller/OCR."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import dispense_pipeline as dp
from test_arm_service import command
from test_reliability import workspace, validation_record
from test_text_pick import fixture, live_fixture


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bridge, self.image, self.result, self.calib, profile, self.status, _ = live_fixture()
        profile.update(fetch_allowed=True, medicine_names=["Aspirin"])
        self.dest = {"schema_version": 1, "role": "stand", "camera_id": "stand", "image_size": [160, 120],
                     "polygon_px": [[105, 60], [150, 60], [150, 114], [105, 114]], "target_px": [125, 90],
                     "measurements_verified": True, "above_cm": 20., "release_height_cm": 3.}
        for name, value in (("items", {"test bottle": profile}), ("workspace", workspace()),
                            ("validation", validation_record(self.status, self.status["table_calibration"])),
                            ("destinations", {"handover": self.dest, "bowl": self.dest})):
            (self.root/f"{name}.json").write_text(json.dumps(value))
        self.ocr = Mock(return_value=.9)
        self.pipe = dp.Pipeline(arm_url="http://test.invalid", items=str(self.root/"items.json"),
            workspace=str(self.root/"workspace.json"), validation=str(self.root/"validation.json"),
            destinations=str(self.root/"destinations.json"), out=str(self.root/"output"), ocr=self.ocr)
        self.target = {"target": "test bottle", "category": "bottle"}
        self.after = copy.deepcopy(self.result)
        self.after["chosen"]["mask"][:] = False
        self.after["chosen"]["mask"][65:105, 110:145] = True
        self.bridge.place.return_value = {"ok": True, "released": True}

    def run_pipeline(self, *, dry_run=False, kind="fetch", after=None, hand=None, stand=None):
        images = {role: self.image for role in ("stand", "hand")}
        stages = [(images, self.status, {r: {"camera_id": r, "sequence": n} for r in images}) for n in range(1, 5)]
        with patch.object(dp.text_pick, "ArmBridge", return_value=self.bridge), \
             patch.object(dp.text_pick, "read_live", side_effect=stages), \
             patch.object(dp.text_pick.ts, "decide", side_effect=[stand or self.result, hand or self.result, after or self.after]), \
             redirect_stdout(io.StringIO()):
            return self.pipe.run(command(kind, dry_run=dry_run), self.target, dry_run=dry_run)

    def test_full_pipeline_preflights_picks_places_and_checks_new_image(self):
        report = self.run_pipeline()
        self.assertEqual(report["status"], "PLACED", report)
        self.assertTrue(report["place_verified"])
        self.assertEqual(self.bridge.place.call_count, 2)
        self.assertTrue(self.bridge.place.call_args_list[0].kwargs["preflight"])
        self.bridge.submit_grasp.assert_called_once()
        self.assertTrue((Path(report["output_dir"])/"placement_preview.jpg").exists())

    def test_dry_run_uses_same_sam_target_and_never_calls_motion_hooks(self):
        report = self.run_pipeline(dry_run=True)
        self.assertTrue(report["object_present"])
        self.bridge.submit_grasp.assert_not_called()
        self.bridge.place.assert_not_called()
        self.ocr.assert_not_called()

    def test_empty_grasp_never_places(self):
        self.bridge.submit_grasp.return_value = {"holding": False}
        report = self.run_pipeline()
        self.assertEqual(report["status"], "EMPTY_REPORTED")
        self.assertEqual(self.bridge.place.call_count, 1)  # preflight only

    def test_low_label_confidence_prevents_pick_and_place(self):
        self.ocr.return_value = .4
        report = self.run_pipeline(kind="dispense")
        self.assertEqual(report["status"], "EXECUTION_BLOCKED")
        self.assertIn("LABEL_UNVERIFIED", report["error"])
        self.bridge.submit_grasp.assert_not_called()
        self.bridge.place.assert_not_called()

    def test_verified_labels_required_before_and_after_medicine_placement(self):
        report = self.run_pipeline(kind="dispense")
        self.assertEqual(report["status"], "PLACED", report)
        self.assertEqual(self.ocr.call_count, 3)
        self.assertTrue(all(call.args[2] == "Aspirin" for call in self.ocr.call_args_list))

    def test_wrist_target_not_required_for_household_or_medicine_pick(self):
        for kind in ("fetch", "dispense"):
            with self.subTest(kind=kind):
                self.bridge.submit_grasp.reset_mock()
                self.ocr.reset_mock()
                report = self.run_pipeline(kind=kind, hand={"status": "NOT_FOUND", "candidates": [], "chosen": None})
                self.assertEqual(report["status"], "PLACED", report)
                self.assertEqual(report["hand_target_status"], "NOT_FOUND")
                self.bridge.submit_grasp.assert_called_once()
                self.assertEqual(self.ocr.call_count, 2 if kind == "dispense" else 0)

    def test_wrist_ambiguity_does_not_override_unique_stand_target(self):
        ambiguous = copy.deepcopy(self.result)
        ambiguous["chosen"] = None
        ambiguous["status"] = "AMBIGUOUS"
        ambiguous["candidates"].append(copy.deepcopy(ambiguous["candidates"][0]))
        report = self.run_pipeline(hand=ambiguous)
        self.assertEqual(report["status"], "PLACED", report)

    def test_wrist_detection_cannot_rescue_missing_stand_target(self):
        report = self.run_pipeline(stand={"status": "NOT_FOUND", "candidates": [], "chosen": None})
        self.assertEqual(report["status"], "EXECUTION_BLOCKED")
        self.bridge.submit_grasp.assert_not_called()

    def test_supplementary_wrist_ocr_cannot_veto_valid_stand_label(self):
        self.ocr.side_effect = [.9, .1, .9]
        report = self.run_pipeline(kind="dispense")
        self.assertEqual(report["status"], "PLACED", report)
        self.assertEqual(report["ocr_score"], .9)
        self.assertEqual(report["hand_ocr"]["score"], .1)

    def test_good_wrist_ocr_cannot_replace_failed_stand_label(self):
        self.ocr.side_effect = [.1, .99]
        report = self.run_pipeline(kind="dispense")
        self.assertEqual(report["status"], "EXECUTION_BLOCKED")
        self.bridge.submit_grasp.assert_not_called()
        self.assertEqual(self.ocr.call_count, 1)

    def test_unverified_destination_blocks_before_pick(self):
        self.dest["measurements_verified"] = False
        (self.root/"destinations.json").write_text(json.dumps({"handover": self.dest}))
        report = self.run_pipeline()
        self.assertEqual(report["status"], "EXECUTION_BLOCKED")
        self.bridge.submit_grasp.assert_not_called()

    def test_release_without_new_object_is_not_placement_success(self):
        report = self.run_pipeline(after={"status": "NOT_FOUND", "candidates": [], "chosen": None})
        self.assertEqual(report["status"], "PLACE_UNVERIFIED", report)
        self.assertFalse(report["place_verified"])

    def test_motion_timeout_never_places_or_retries(self):
        self.bridge.submit_grasp.side_effect = TimeoutError("response lost")
        report = self.run_pipeline()
        self.assertEqual(report["status"], "MOTION_RESULT_UNKNOWN")
        self.bridge.submit_grasp.assert_called_once()
        self.assertEqual(self.bridge.place.call_count, 1)

    def test_existing_target_at_destination_blocks_before_pick(self):
        self.dest["polygon_px"] = [[55, 20], [150, 20], [150, 114], [55, 114]]
        (self.root/"destinations.json").write_text(json.dumps({"handover": self.dest}))
        report = self.run_pipeline()
        self.assertEqual(report["status"], "EXECUTION_BLOCKED")
        self.assertIn("already overlaps", report["error"])
        self.bridge.submit_grasp.assert_not_called()


class OcrTests(unittest.TestCase):
    def tsv(self, words):
        return "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tconf\ttext\n" + "".join(
            f"5\t1\t1\t1\t{line}\t{i}\t{score}\t{word}\n" for i,(line,word,score) in enumerate(words))

    def test_exact_label_uses_minimum_word_confidence_on_same_line(self):
        data = self.tsv([(1, "VITAMIN", 95), (1, "D", 71)])
        self.assertEqual(dp.label_score(data, "Vitamin D"), .71)
        self.assertEqual(dp.label_score(data, "Aspirin"), 0)
        self.assertEqual(dp.label_score(self.tsv([(1,"VITAMIN",95),(2,"D",99)]), "Vitamin D"), 0)

    def test_missing_ocr_program_blocks_without_faking_a_score(self):
        image, result, *_ = fixture()
        with patch.object(dp.shutil, "which", return_value=None), self.assertRaisesRegex(ValueError, "OCR_UNAVAILABLE"):
            dp.read_label(image, result["chosen"]["mask"], "Aspirin")


if __name__ == "__main__":
    unittest.main()
