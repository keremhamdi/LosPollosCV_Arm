"""Synthetic feedback regressions. No Station, camera or servo connection."""
import asyncio
import math
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from controller.arm.descent_contact import DescentContact
import test_integration as fixtures


class ContactEvidenceTests(unittest.TestCase):
    def test_increasing_lag_while_still_descending_is_not_contact(self):
        detector = DescentContact(2.5)
        history, old_over, old_triggered = [], 0, False
        for i in range(76):
            t = i * .04
            target = 7 - 2.5*t
            gap = .12 + 1.5 * max(0., min(.8, t-.8))
            measured = target + gap  # still moves down >=1 cm/s throughout
            outcome = detector.update(t, target, measured)
            self.assertFalse(outcome['contact'])
            history.append((t, gap))
            usual = [g for tt, g in history if tt >= .25 and t-.7 <= tt <= t-.12]
            old_over = old_over + 1 if t > .45 and usual and gap-min(usual) > .6 else 0
            old_triggered |= old_over >= 2
        self.assertTrue(old_triggered, 'Fixture must reproduce the old false stop')

    def test_real_stationary_surface_after_free_motion_is_detected(self):
        detector = DescentContact(2.5)
        for i in range(100):
            t = i * .04
            target = 7 - 2.5*t
            measured = max(.2, target + .2)
            result = detector.update(t, target, measured)
            if result['contact']:
                self.assertEqual(measured, .2)
                self.assertLess(t, 3.2)
                self.assertGreater(result['free_travel_cm'], .25)
                break
        else:
            self.fail('Missed sustained surface stop')

    def test_brief_pause_does_not_count_as_contact(self):
        detector = DescentContact(2.5)
        for i in range(70):
            t = i*.04
            measured = 7-2.5*t+.15
            if 1.0 <= t <= 1.08:
                measured = 4.65
            self.assertFalse(detector.update(t, 7-2.5*t, measured)['contact'])

    def test_blocked_from_start_is_not_an_object_contact(self):
        detector = DescentContact(2.5)
        reasons = [detector.update(i*.04, 7-2.5*i*.04, 7)['reason'] for i in range(30)]
        self.assertIn('blocked_before_descent', reasons)
        self.assertNotIn('contact_inferred', reasons)

    def test_repeated_or_nonfinite_samples_rejected(self):
        detector = DescentContact(2.5)
        detector.update(0., 7, 7)
        for sample in ((0., 6, 7), (.04, 6., float('nan'))):
            with self.assertRaises(ValueError):
                detector.update(*sample)


class DescentLoopTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ControllerRoutingTests.setUpClass()
        cls.mod = fixtures.ControllerRoutingTests.reach_module

    def make_reacher(self, surface=.2, frozen=False, stale_motor=None, lag=False):
        self.clock = 100.
        names = self.mod.MOTOR_JOINTS
        reach = self.mod.Reacher.__new__(self.mod.Reacher)
        joints = {m: types.SimpleNamespace(raw=1700, goal=1700, range_min=0, range_max=4095,
                                           torque=True, calibrated=True) for m in range(1, 8)}
        arm = types.SimpleNamespace(joints=joints, sag={}, joint_updated_at={m:self.clock for m in joints},
                                    hold=AsyncMock(), _settle_and_ease=AsyncMock(), writes=[])
        async def write(writes):
            goals = {m:int.from_bytes(value, 'little') for m, _, value in writes}
            arm.writes.append(goals)
            for m, goal in goals.items():
                joints[m].goal = goal
        async def sleep(dt):
            self.clock += dt
            t = self.clock - 100.
            gap = .12 + (1.5*max(0., min(.8, t-.8)) if lag else 0.)
            h = 7. if frozen else min(7., max(surface, (joints[1].goal-1000)/100 + gap))
            for m in joints:
                if m != stale_motor:
                    joints[m].raw = round(1000+100*h)
                    arm.joint_updated_at[m] = self.clock
        arm._write = write
        reach.arm = arm
        reach.table = types.SimpleNamespace(height=lambda p:p[2])
        reach._raw = lambda m, q: round(1000+100*q)
        reach.measured_angles = lambda:{n:(joints[m].raw-1000)/100 for m,n in names.items()}
        reach.tip = lambda q:((0., .25, q[names[1]]/100), (0,0,-1), [])
        reach.state = lambda:(0., 25., (joints[1].raw-1000)/100, 0.)
        def path(x, y, h, **kw):
            start = reach.state()[2]
            count = max(4, math.ceil(abs(start-h)/.5))
            return [{n:start+(h-start)*i/count for n in names.values()} for i in range(1,count+1)]
        reach.plan_line = Mock(side_effect=path)
        async def follow(path, seconds, **kw):
            for m,n in names.items():
                joints[m].raw = reach._raw(m,path[-1][n])
        reach.follow = AsyncMock(side_effect=follow)
        self.patch = patch.dict(self.mod.Reacher.descend_guarded.__globals__, {
            'time':types.SimpleNamespace(monotonic=lambda:self.clock),
            'asyncio':types.SimpleNamespace(sleep=sleep), 'roll_raw':lambda:None})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        return reach

    async def test_free_moving_lag_reaches_floor_without_false_contact(self):
        reach = self.make_reacher(surface=-99., lag=True)
        report = {}
        result = await reach.descend_guarded(0,25,0.,report=report)
        self.assertFalse(result['contact'])
        self.assertEqual(report['stop_reason'],'floor_reached_without_contact')
        reach.follow.assert_not_awaited()
        self.assertTrue(reach.plan_line.call_args.kwargs['from_measured'])

    async def test_surface_stop_backs_off_from_actual_height_without_sag_push(self):
        reach = self.make_reacher()
        result = await reach.descend_guarded(0,25,-2.5)
        self.assertTrue(result['contact'])
        self.assertAlmostEqual(result['height_cm'],.2)
        self.assertAlmostEqual(result['final_cm'],.5)
        reach.follow.assert_awaited_once()
        self.assertFalse(reach.follow.call_args.kwargs['correct_sag'])
        self.assertTrue(reach.plan_line.call_args.kwargs['from_measured'])
        # wrist roll was not commanded, matching the existing controller contract.
        self.assertTrue(all(7 not in w for w in reach.arm.writes))

    async def test_one_stale_joint_stops_even_when_other_joints_update(self):
        reach = self.make_reacher(stale_motor=2)
        with self.assertRaisesRegex(RuntimeError,'joint 2'):
            await reach.descend_guarded(0,25,-2.5)
        self.assertEqual(reach.arm.writes[-1], {m:reach.arm.joints[m].raw for m in range(1,7)})
        reach.follow.assert_not_awaited()

    async def test_initial_stall_never_becomes_valid_contact(self):
        reach = self.make_reacher(frozen=True)
        with self.assertRaisesRegex(RuntimeError,'DESCENT_BLOCKED'):
            await reach.descend_guarded(0,25,-2.5)
        reach.follow.assert_not_awaited()

    async def test_unknown_height_no_contact_never_closes_jaws(self):
        reach = self.mod.Reacher.__new__(self.mod.Reacher)
        reach.arm = types.SimpleNamespace(move=AsyncMock(), close_gripper=AsyncMock())
        reach.pose_for = Mock(return_value=(({},{}),0.))
        reach.plan_swing = Mock(return_value=[{}])
        reach.duration = Mock(return_value=1.)
        reach.follow = AsyncMock()
        reach.plan_line = Mock(return_value=[{}])
        reach.descend_guarded = AsyncMock(return_value={'contact':False,'height_cm':-2.5,'final_cm':-2.5})
        with self.assertRaisesRegex(RuntimeError,'DESCENT_NO_CONTACT'):
            await reach.grab(0,25,None,14,guard=True)
        reach.arm.close_gripper.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
