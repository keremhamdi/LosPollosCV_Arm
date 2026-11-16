"""Completed requests and actual holding evidence must not be conflated."""
import io
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from PIL import Image
import grab_text as grab


class PickupOutcomeTests(unittest.TestCase):
    def test_successful_http_attempt_can_still_be_a_miss(self):
        self.assertEqual(grab.pickup_outcome({'ok':True,'holding':False})[0],'MISSED')
        self.assertEqual(grab.pickup_outcome({'ok':True,'report':{'holding':False}})[0],'MISSED')

    def test_hold_is_labelled_as_feedback_not_visual_verification(self):
        state,message=grab.pickup_outcome({'ok':True,'holding':True,'report':{'holding':True}})
        self.assertEqual(state,'HOLD DETECTED')
        self.assertIn('feedback',message)
        self.assertIn('Check',message)

    def test_slip_overrides_an_earlier_hold(self):
        self.assertEqual(grab.pickup_outcome({'ok':True,'holding':True,'report':{'slipped':True}})[0],
                         'SLIPPED')

    def test_missing_invalid_or_conflicting_evidence_is_unconfirmed(self):
        for response in ({'ok':True}, {'ok':True,'holding':'true'},
                         {'ok':True,'holding':True,'report':{'holding':False}}):
            with self.subTest(response=response):
                self.assertEqual(grab.pickup_outcome(response)[0],'UNCONFIRMED')

    def test_failure_is_not_hidden_by_hold_or_slip_fields(self):
        for response in ({'ok':False,'error':'move failed','holding':True},
                         {'ok':True,'report':{'error':'lift failed','slipped':True}}):
            with self.subTest(response=response):
                self.assertEqual(grab.pickup_outcome(response)[0],'FAILED')

    def test_actual_pickup_output_uses_response_without_a_status_poll(self):
        args=types.SimpleNamespace(arm_url='http://offline-test',last_text='test object',close=1.,object_height=None,above=14.,
                                   side=False,no_feel=False,dry=False,no_hand=True,reach_out=0.)
        for holding,expected in ((True,'HOLD DETECTED:'),(False,'MISSED:')):
            response={'ok':True,'holding':holding,'report':{'holding':holding}}
            output=io.StringIO()
            with self.subTest(holding=holding), \
                    patch.object(grab.requests,'post',return_value=Mock(json=lambda:response)) as post, \
                    patch.object(grab.requests,'get',side_effect=AssertionError('Result must not depend on status poll')), \
                    redirect_stdout(output):
                grab.go(args,Image.new('RGB',(1280,720)),500,500,'test object',.8,
                        close=1.,above=14.,aim={'xy_cm':[0.,25.]})
            self.assertIn(expected,output.getvalue())
            self.assertNotIn('OK:',output.getvalue())
            self.assertEqual(post.call_count,1)
            self.assertEqual(post.call_args.args[0].rsplit('/',1)[1],'grab')


if __name__ == '__main__':
    unittest.main()
