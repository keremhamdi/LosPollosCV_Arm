"""Wrong-region and identity regressions for wrist SAM outputs (inference mocked)."""
import io
import json
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch, mock_open

import numpy as np
from PIL import Image
import grab_text as grab


def candidate(x, y, score, size=(1280, 720)):
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[int(y)-5:int(y)+6, int(x)-5:int(x)+6] = True
    return {'mask': mask, 'target_score': score, 'box': [x-5, y-5, x+6, y+6]}


class WristSelectionTests(unittest.TestCase):
    def test_stale_wrist_frame_is_not_used_for_alignment(self):
        response = {'frames': {'hand': {'jpeg_base64':'not-decoded', 'receive_age_s':3.}}}
        with patch.object(grab.requests,'get',return_value=Mock(json=lambda:response)):
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                grab.hand_frame(types.SimpleNamespace(arm_url='http://offline-test'))

    def find(self, responses, tries=2):
        img = Image.new('RGB', (1280, 720))
        with patch.object(grab, 'hand_frame', return_value=img), \
                patch.object(grab.ts, 'decide', side_effect=responses) as detect, \
                patch('time.sleep'), redirect_stdout(io.StringIO()):
            result = grab.find_in_hand(types.SimpleNamespace(), 'small bottle', near=(620.,425.), tries=tries)
        return result, detect

    def test_distant_high_score_does_not_suppress_local_bottle(self):
        # Matches the bundle geometry: highlighted upper-right region, bottle below jaws.
        (point, _, _), detect = self.find([{'candidates': [
            candidate(1205, 20, .99), candidate(620, 575, .3)]}])
        self.assertEqual(point, (620., 575.))
        self.assertEqual(detect.call_count, 1)

    def test_no_local_match_never_switches_to_generic_small_white_object(self):
        (point, _, _), detect = self.find([
            {'candidates': [candidate(1205,20,.99)]}, {'candidates': []}])
        self.assertIsNone(point)
        self.assertEqual(detect.call_count, 2)
        for call in detect.call_args_list:
            self.assertEqual(call.args[1:], ('small bottle', None))

    def test_cropped_retry_restores_full_image_mask_and_coordinates(self):
        # Crop around (620,425): left=300, top=105, right=940, bottom=720.
        (point, img, best), detect = self.find([
            {'candidates': []}, {'candidates': [candidate(320,470,.4,(640,615))]}])
        self.assertEqual(point, (620.,575.))
        self.assertEqual(best['mask'].shape, (720,1280))
        self.assertEqual(detect.call_args_list[1].args[0].size, (640,615))

    def align(self, points):
        args = types.SimpleNamespace(no_hand=False, arm_url='http://offline-test')
        cal = {'target_px': [620.,425.], 'px_per_cm': [[17.25,0.],[0.,17.25]],
               'at': [0.,25.], 'look_above': 13.}
        img = Image.new('RGB',(1280,720))
        with patch('os.path.exists',return_value=True), \
                patch('builtins.open',mock_open(read_data=json.dumps(cal))), \
                patch.object(grab.requests,'post',return_value=Mock()), \
                patch.object(grab,'goto') as move, patch.object(grab,'save_hand_view'), \
                patch.object(grab,'find_in_hand',side_effect=[(p,img,None) for p in points]), \
                redirect_stdout(io.StringIO()):
            xy = grab.hand_correct(args, 'small bottle', 0.,25.,4.)
        return xy,args,move

    def test_large_valid_correction_uses_small_steps_and_observes_final_position(self):
        xy,args,move = self.align([(568.25,425.),(602.75,425.),(620.,425.)])
        self.assertEqual(xy,(3.,25.))
        self.assertTrue(args.wrist_alignment_used)
        self.assertEqual([c.args[1:] for c in move.call_args_list],
                         [(0.,25.,17.),(2.,25.,17.),(3.,25.,17.)])

    def test_lost_target_does_not_return_unobserved_corrected_position(self):
        xy,args,move = self.align([(568.25,425.),None])
        self.assertEqual(xy,(0.,25.))
        self.assertFalse(args.wrist_alignment_used)

    def test_no_unverified_extra_move_after_last_look(self):
        xy,args,move = self.align([(602.75,425.)]*5)
        self.assertEqual(xy,(0.,25.))
        self.assertFalse(args.wrist_alignment_used)
        self.assertEqual(move.call_count,5)


if __name__ == '__main__':
    unittest.main()
