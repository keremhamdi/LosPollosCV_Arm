"""Automatic named-object profiles: geometry and routing, without SAM/hardware."""
import io
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

import grab_text as grab
import small_object_grasp as small


def geometry():
    centre = np.array([0.,-35.,45.])
    forward = np.array([0.,25.,0.])-centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0.,0.,1.]); right /= np.linalg.norm(right)
    down = np.cross(forward,right)
    rotation = np.array([right,down,forward])
    intrinsics = np.array([[900.,0.,640.],[0.,900.,360.],[0.,0.,1.]])
    return intrinsics,rotation,-rotation@centre,centre,np.array([0.,0.,1.])


def cylinder_mask(orientation, yaw=0., xy=(3.,25.)):
    points = small.cylinder_points(small.CYLINDER,orientation,np.radians(yaw)) + [*xy,0.]
    pixels = small.project(points,geometry())
    mask = np.zeros((720,1280),np.uint8)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype(np.int32)),1)
    return mask.astype(bool)


def candidate(mask, score=.8):
    ys,xs = np.nonzero(mask)
    return {'mask':mask,'target_score':score,'box':[xs.min(),ys.min(),xs.max()+1,ys.max()+1]}


class SmallObjectTests(unittest.TestCase):
    def test_profiles_match_names_without_changing_command_for_orientation(self):
        for text in ('small white cylinder','white cylinder lying sideways','pick up the white cylinder'):
            self.assertEqual(small.profile_for(text)['name'],'white cylinder')
        self.assertEqual(small.profile_for('small black cap')['name'],'black cap')
        for text in ('white bottle','black cylinder','small object','white cap'):
            self.assertIsNone(small.profile_for(text))

    def test_same_automatic_height_for_cylinder_in_both_orientations(self):
        args = types.SimpleNamespace(last_text='small white cylinder',close=None,object_height=None,above=14.)
        with patch.object(grab,'estimate_height',side_effect=AssertionError('upright-only height guess forbidden')):
            for orientation in ('upright','sideways'):
                close,above,est = grab.grab_heights(args,cylinder_mask(orientation),(1280,720))
                self.assertEqual(close,.5)
                self.assertEqual(above,14.)
                self.assertEqual(est[1],1.)

    def test_cap_uses_low_profile_instead_of_15mm_fallback(self):
        args = types.SimpleNamespace(last_text='small black cap',close=None,object_height=None,above=14.)
        with patch.object(grab,'estimate_height',side_effect=AssertionError('generic height guess forbidden')):
            close,above,est = grab.grab_heights(args,None,(1280,720))
        self.assertEqual(close,.5)
        self.assertEqual(est[0],.75)  # explicit range midpoint hint, not a measured height
        self.assertEqual(small.CAP['height_range_cm'],[.5,1.])

    def test_projected_cylinder_centres_across_orientations_and_positions(self):
        for xy in ((3.,25.),(-8.,23.),(7.,30.)):
            for orientation,yaw in [('upright',0),('sideways',0),('sideways',45),('sideways',90),('sideways',135)]:
                with self.subTest(xy=xy,orientation=orientation,yaw=yaw):
                    fit = small.aim_from_mask(small.CYLINDER,cylinder_mask(orientation,yaw,xy),geometry())
                    self.assertLess(np.linalg.norm(np.array(fit['xy_cm'])-xy),.4)
                    self.assertGreater(fit['fit_iou'],.75)

    def test_missing_default_phrase_retries_sideways_phrase(self):
        c = candidate(cylinder_mask('sideways'))
        detector = Mock(side_effect=[{'candidates':[]},{'candidates':[c]}])
        result = small.detect(Image.new('RGB',(1280,720)),'small white cylinder',detector)
        self.assertEqual(detector.call_args_list[1].args[1],'white cylinder lying on its side')
        self.assertEqual(len(result['candidates']),1)

    def test_named_prompt_retries_do_not_fall_back_to_any_small_object(self):
        detector = Mock(return_value={'candidates':[]})
        small.detect(Image.new('RGB',(1280,720)),'small black cap',detector)
        self.assertGreater(detector.call_count,1)
        for call in detector.call_args_list:
            self.assertIn('black',call.args[1])
            self.assertIn('cap',call.args[1])

    def test_duplicate_masks_are_not_separate_objects(self):
        c = candidate(cylinder_mask('upright'),.4)
        detector = Mock(return_value={'candidates':[c]})
        result = small.detect(Image.new('RGB',(1280,720)),'small white cylinder',detector,exhaustive=True)
        self.assertEqual(len(result['candidates']),1)
        self.assertGreater(detector.call_count,1)

    def test_background_match_does_not_prevent_table_zoom_retry(self):
        image = Image.new('RGB',(1280,720)); encoded=io.BytesIO(); image.save(encoded,format='JPEG')
        c = candidate(cylinder_mask('sideways'))
        args = types.SimpleNamespace(arm_url='http://offline-test',close=None,object_height=None,above=14.)
        response = Mock(json=lambda:{'capabilities':{'controller_build':'descent-taper-v1'}},content=encoded.getvalue())
        with patch.object(grab.requests,'get',return_value=response), \
                patch.object(Image.Image,'save'), patch('pathlib.Path.mkdir'), \
                patch.object(grab,'detect_fixed',side_effect=[{'candidates':[dict(c)]},{'candidates':[dict(c)]}]) as detect, \
                patch.object(grab,'in_reach',side_effect=[(None,False),((3.,25.),True)]), \
                patch.object(grab,'camera_geometry',return_value=None), patch.object(grab,'show_view'), \
                patch.object(grab,'go') as go, redirect_stdout(io.StringIO()):
            grab.find_and_grab(args,'small white cylinder')
        self.assertTrue(detect.call_args_list[1].kwargs['force_zoom'])
        self.assertEqual(go.call_args.args[6],.5)

    def test_fit_returns_table_centre_not_lowest_silhouette_pixel(self):
        mask = cylinder_mask('sideways',45)
        fit = small.aim_from_mask(small.CYLINDER,mask,geometry())
        self.assertLess(np.linalg.norm(np.array(fit['xy_cm'])-[3.,25.]),.1)
        low = grab.base_pixel(mask)
        centre_pixel = small.project(np.array([[*fit['xy_cm'],0.]]),geometry())[0]
        self.assertGreater(np.linalg.norm(np.array(low)-centre_pixel),1.)

    def test_profile_preview_never_moves_or_uses_wrist(self):
        args=types.SimpleNamespace(last_text='small white cylinder',close=None,object_height=None,
                                   above=14.,side=False,no_feel=False,dry=True)
        fit=small.aim_from_mask(small.CYLINDER,cylinder_mask('sideways'),geometry())
        with patch.object(grab.requests,'post',side_effect=AssertionError('No POST in fitted preview')), \
                patch.object(grab,'hand_correct',side_effect=AssertionError('No wrist movement in preview')), \
                redirect_stdout(io.StringIO()):
            grab.go(args,Image.new('RGB',(1280,720)),600,600,'small white cylinder',.8,
                    close=.5,above=14.,est=(1.5,1.,None),aim=fit)


if __name__ == '__main__':
    unittest.main()
