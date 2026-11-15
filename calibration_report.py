"""Read-only calibration diagnostics. Never opens cameras or commands the arm."""
import argparse
import json
from pathlib import Path

import numpy as np
import requests

from pick_v5 import Calib
import calibration_validation
from controller.arm.kinematics import TIP_OFFSET


def summarize_target(table, x_cm, y_cm, height_cm):
    """Expose the same plane-normal fingertip target used by Reacher._target."""
    a, b, c = [table[k] for k in ("a", "b", "c")]
    values = [a, b, c, x_cm, y_cm, height_cm]
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in values):
        raise ValueError("Plane and target coordinates must be finite numbers")
    if height_cm < 0:
        raise ValueError("Target height must be nonnegative")
    contact_cm = np.array([x_cm, y_cm, a*x_cm + b*y_cm + c*100])
    normal = np.array([-a, -b, 1.]) / np.sqrt(a*a + b*b + 1)
    return {"table_contact_base_cm": contact_cm.tolist(),
            "tip_target_base_cm": (contact_cm + normal*height_cm).tolist(),
            "height_above_plane_cm": height_cm, "table_normal_base": normal.tolist(),
            "tip_offset_gripper_m": list(TIP_OFFSET),
            "reference": "point between closed fingertips, not wrist or camera origin",
            "physical_accuracy_validated": False}


def summarize_tip(table, frame, measured_height_cm=None):
    """Read encoder FK at frame receipt. Not image-derived depth or metrology."""
    pose = frame.get("pose", {})
    age = frame.get("receive_age_s")
    if (pose.get("pose_valid") is not True or pose.get("translation_units") != "m" or
        type(age) not in (float, int) or not np.isfinite(age) or not 0 <= age <= 1):
        raise ValueError("Fresh valid measured gripper pose required for a fingertip height audit")
    matrix = np.asarray(pose.get("T_base_gripper"), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid measured gripper transform")
    p = (matrix @ np.array([*TIP_OFFSET, 1.]))[:3]
    a, b, c = [float(table[k]) for k in ("a", "b", "c")]
    height = float((p[2] - a*p[0] - b*p[1] - c) / np.sqrt(a*a + b*b + 1) * 100)
    report = {"tip_base_cm": (p*100).tolist(), "encoder_height_above_plane_cm": height,
              "tip_offset_gripper_m": list(TIP_OFFSET), "physical_accuracy_validated": False,
              "timestamp_basis": "host_receive; hold the arm still for a ruler comparison"}
    if measured_height_cm is not None:
        if not np.isfinite(measured_height_cm) or measured_height_cm < 0:
            raise ValueError("Measured fingertip height must be finite and nonnegative")
        report.update(measured_height_cm=measured_height_cm,
                      encoder_minus_measured_cm=height-measured_height_cm)
    return report


def summarize_camera(data):
    pairs = np.asarray(data["pairs"], dtype=float)
    size = float(data["size"])
    if pairs.ndim != 2 or pairs.shape[1] != 4 or len(pairs) < 4 or not np.isfinite(pairs).all():
        raise ValueError("Need at least four finite [u,v,x_cm,y_cm] marks")
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Calibration image width must be finite and positive")
    calib = Calib(pairs.tolist(), size)
    fit = [float(np.linalg.norm(np.array(calib.to_table(u, v)) - [x, y])) for u, v, x, y in pairs]
    loo = calib.loo
    if not np.isfinite(fit + loo).all():
        raise ValueError("Degenerate calibration produced nonfinite errors")
    return {"marks": len(pairs), "image_width_px": size,
            "fit_rms_cm": float(np.sqrt(np.mean(np.square(fit)))),
            "fit_max_cm": max(fit),
            "leave_one_out_max_cm": max(loo) if loo else None,
            "camera_error_gate_passed": bool(len(pairs) >= 6 and loo and max(loo) <= 1.0),
            "physical_accuracy_validated": False,
            "note": "Residuals from stored marks, not independent robot landing measurements. Target coverage must also be checked."}


def summarize_table(data):
    a, b, c = [float(data[key]) for key in ("a", "b", "c")]
    if not np.isfinite([a, b, c]).all():
        raise ValueError("Table coefficients must be finite")
    report = {"equation": "z = a*x + b*y + c", "units": "m", "a": a, "b": b, "c": c,
              "physical_accuracy_validated": False,
              "note": "Touch residuals cannot detect shared joint/FK/fingertip error"}
    points = np.asarray(data.get("touches_m", []), dtype=float)
    if points.size:
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Table touches must be finite xyz points in metres")
        errors = (points[:, 2] - a * points[:, 0] - b * points[:, 1] - c) / np.sqrt(a*a + b*b + 1) * 1000
        report.update(touches=len(points), fit_rms_mm=float(np.sqrt(np.mean(errors**2))),
                      fit_max_mm=float(np.abs(errors).max()))
    else:
        report.update(touches=0, fit_rms_mm=None, fit_max_mm=None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", help="measured camera JSON file")
    source.add_argument("--arm-url", help="read camera pairs from controller GET /status")
    parser.add_argument("--table", help="measured table JSON file on this laptop")
    parser.add_argument("--validation", help="independent measurement record to evaluate")
    parser.add_argument("--write-validation-template", help="write an empty record bound to the loaded geometry")
    parser.add_argument("--camera-id", help="exact stand camera ID from /observation")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--target-cm", nargs=2, type=float, metavar=("X", "Y"), help="inspect a table XY target; no motion")
    parser.add_argument("--height-cm", type=float, help="requested fingertip height above the table; use with --target-cm")
    parser.add_argument("--measured-tip-height-cm", type=float, help="ruler height of stationary fingertips to compare with live encoder FK")
    args = parser.parse_args(argv)
    if not any((args.camera, args.arm_url, args.table)):
        parser.error("Provide --camera, --arm-url or --table")
    if (args.target_cm is None) != (args.height_cm is None):
        parser.error("Use --target-cm and --height-cm together")
    if args.measured_tip_height_cm is not None and not args.arm_url:
        parser.error("--measured-tip-height-cm requires --arm-url")
    report = {"motion_allowed": False}
    try:
        camera, table, status = None, None, {}
        if args.camera:
            camera = json.loads(Path(args.camera).read_text())
            report["camera"] = summarize_camera(camera)
        elif args.arm_url:
            response = requests.get(args.arm_url.rstrip("/") + "/status", timeout=5)
            response.raise_for_status()
            status = response.json()
            camera, table = status, status.get("table_calibration")
            report["camera"] = summarize_camera(status)
            report["camera"]["selected_camera_role"] = status.get("camera_role")
        if args.table:
            table = json.loads(Path(args.table).read_text())
        if table:
            report["table"] = summarize_table(table)
        if args.target_cm is not None:
            if not table:
                raise ValueError("Table calibration required to inspect a target height")
            report["target_height"] = summarize_target(table, *args.target_cm, args.height_cm)
        if args.measured_tip_height_cm is not None:
            if not table:
                raise ValueError("Table calibration required for a fingertip height comparison")
            frame = status.get("observations", {}).get("frames", {}).get("stand", {})
            report["fingertip_comparison"] = summarize_tip(table, frame, args.measured_tip_height_cm)
        if args.validation or args.write_validation_template:
            if args.arm_url and status.get("camera_role") != "stand":
                raise ValueError("Select the fixed standing camera before recording validation")
            frame = status.get("observations", {}).get("frames", {}).get("stand", {})
            size = args.image_size or frame.get("image_size")
            camera_id = args.camera_id or frame.get("camera_id")
            if not camera or not table or not size or not camera_id:
                raise ValueError("Validation needs camera and table geometry, camera ID and full image size")
            if args.write_validation_template:
                path = Path(args.write_validation_template)
                # Never overwrite existing real measurements with a blank template.
                with path.open("x") as file:
                    json.dump(calibration_validation.template(camera, table, size, camera_id), file, indent=2)
                report["template"] = str(path)
            if args.validation:
                record = json.loads(Path(args.validation).read_text())
                report["independent_validation"] = calibration_validation.validate(record, camera, table, size, camera_id)
    except (ValueError, KeyError, TypeError, OSError, requests.RequestException, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"Calibration audit failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
