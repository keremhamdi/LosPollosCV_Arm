# Improvements to prioritise

## Evidence from the SAM-only check

On 2026-11-15, the default `text_pick.py` command was run against the supplied
standing/hand photos for `white cup`. Cached SAM 3 weights were used with
Hugging Face offline mode and all Requests HTTP calls forbidden. The command
completed without a checker service or robot connection.

- Hand image: one selected cup mask.
- Standing image: AMBIGUOUS because both the target cup and a background cup
  passed the SAM score threshold. No pickup request was produced.
- Hand silhouette posture estimate: `flat`, although the cup is upright. Image
  aspect ratio changes with camera viewpoint and is not a reliable 3D posture.

This is one diagnostic example, not a measured recognition success rate. The
80 offline regression tests pass, including a default-command test that forbids
checker and controller HTTP calls. The earlier checker-enabled photo results
should not be treated as SAM-only benchmark results.

## Recommended implementation order

1. **Remeasure geometry.** Correct joint offsets/fingertip geometry, table plane,
   camera intrinsics/distortion and fixed-camera alignment. Verify unused physical
   points across the pickup area. Choose a position tolerance from the actual
   gripper/contact clearance; a low fitting residual alone is insufficient.
2. **Limit target selection to the reachable workspace.** Use measured table
   bounds to reject background candidates before ambiguity checking. Keep raised
   object silhouettes intact: their projected upper parts may extend beyond a
   tabletop polygon. Do not solve ambiguity by blindly taking the largest score.
3. **Use measured object and gripper geometry.** Start with a small set of known,
   rigid objects, their dimensions and suitable grasp methods. Do not infer
   upright/flat posture solely from image aspect ratio. Measure opening, finger
   thickness, fingertip offsets and grip contact. Some existing example objects
   are wider than the nominal usable gripper opening.
4. **Execute the planned orientation.** Add and validate wrist-roll-aware IK and
   approach clearance before enabling top grasps. Test on the actual calibrated
   arm model; a 2D jaw line is not a verified robot orientation.
5. **Use the hand camera for feedback.** After hand-eye calibration and exposure /
   joint timestamp alignment, add bounded final-approach corrections, target-loss
   stops and visual lift verification. Camera feeds alone are not a control loop.
6. **Measure results on the Linux robot.** Record missing/duplicate-object errors,
   position/angle/width error, latency, successful lifts and slips across varied
   positions and lighting. Use those results to tune thresholds and decide whether
   another vision model would actually help.

References: [SAM 3 concepts and masks](https://huggingface.co/docs/transformers/model_doc/sam3),
[OpenCV camera and hand-eye calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
