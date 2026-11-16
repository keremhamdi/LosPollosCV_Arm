"""A pretend table, for testing the aiming without the robot (used by test_aim.py).

  FixedCamera   a wide-angle camera looking at the table from the side opposite the arm
  boards()      checkerboard calibration marks as that camera sees them
  cylinder(), box(), outline()   simple objects and the outline the camera sees of them
  HandWorld     a gripper with a camera fixed to it, which leans out the further it reaches
  Reply         what a pretend arm page answers

Nothing here talks to an arm, a camera or a model.
"""
import numpy as np
import cv2

W, H = 1280, 720
BASE = np.array([0.0, 4.2])      # the arm's turning axis on the table (cm)


class FixedCamera:
    def __init__(self, k1=-0.28, k2=0.07, f=640.0, at=(3.0, 78.0, 46.0), looks_at=(0.0, 24.0, 0.0)):
        at, looks_at = np.array(at, float), np.array(looks_at, float)
        z = looks_at - at
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        self.R, self.C, self.k1, self.k2, self.f = np.vstack([x, np.cross(z, x), z]), at, k1, k2, f

    def project(self, points):
        """Pixels of 3D points (cm, table coordinates), with the lens's bending."""
        p = (np.atleast_2d(points) - self.C) @ self.R.T
        n = p[:, :2] / p[:, 2:3]
        r2 = (n ** 2).sum(axis=1, keepdims=True)
        return n * (1 + self.k1 * r2 + self.k2 * r2 ** 2) * self.f + [W / 2, H / 2]


BOARDS = [((-16, 12), 0.1), ((2, 16), -0.2), ((-8, 28), 0.05)]   # where each 9 x 6 board lies: corner, turn


def boards(cam, where=BOARDS, square=2.5, shifts=None):
    """Calibration marks [u, v, x_cm, y_cm]. shifts: per board, how far off the arm thought it lay."""
    out = []
    for n, (origin, turn) in enumerate(where):
        c, s = np.cos(turn), np.sin(turn)
        pts = np.array([[origin[0] + c * i * square - s * j * square, origin[1] + s * i * square + c * j * square, 0]
                        for j in range(6) for i in range(9)])
        dx, dy = shifts[n] if shifts else (0.0, 0.0)
        out += [[u, v, x + dx, y + dy] for (u, v), (x, y, _) in zip(cam.project(pts), pts)]
    return out


def cylinder(x, y, across, tall, n=80):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[x + across / 2 * np.cos(t), y + across / 2 * np.sin(t)]
    return np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, tall)]])


def box(x, y, wide, deep, tall, turn=0.0):
    c, s = np.cos(turn), np.sin(turn)
    corners = np.array([[x + c * a - s * b, y + s * a + c * b]
                        for a, b in ((sx * wide / 2, sy * deep / 2) for sx in (-1, 1) for sy in (-1, 1))])
    return np.vstack([np.c_[corners, np.zeros(4)], np.c_[corners, np.full(4, tall)]])


def outline(cam, points, zoom=1.0, offset=(0, 0), size=(W, H)):
    """The filled outline the camera sees of a (convex) object; optionally an enlarged cut-out of it."""
    px = (cam.project(points) + 0.5 - np.array(offset, float)) * zoom - 0.5
    m = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(m, np.round(cv2.convexHull(px.astype(np.float32))).astype(np.int32), 1)
    return m.astype(bool)


def ragged(mask, rng):
    """The same outline with its edge pixels randomly on or off, like a real segmentation."""
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    edge = cv2.dilate(m, k).astype(bool) & ~cv2.erode(m, k).astype(bool)
    out = mask.copy()
    out[edge] = rng.random(mask.shape)[edge] < 0.5
    return out


class HandWorld:
    """A gripper with a camera fixed to it, above a table with one object (x, y, height of its top).
    Like the real arm it points straight down near the base and leans outwards further away, in
    steps of 5 degrees; the fingertips are what is sent to x, y, height."""

    def __init__(self, obj, cam_back=6.0, cam_in=-3.5, cam_side=0.5, mount_deg=12.0, f=600.0, lean_from=18.0,
                 says_lean=True):
        self.obj = np.array(obj, float)
        self.cam = (cam_back, cam_in, cam_side, np.radians(mount_deg), f)
        self.lean_from, self.says_lean = lean_from, says_lean
        self.tip, self.tilt, self.moves = None, 0.0, []

    def needed(self, x, y, h):
        r = np.hypot(x - BASE[0], y - BASE[1])
        return max(0.0, (r - self.lean_from) * 2.2 + (h - 12.0) * 0.8)

    def goto(self, x, y, h, tilt=None):
        need = self.needed(x, y, h)
        if tilt is None:
            tilt = 5.0 * np.ceil(need / 5.0 - 1e-9) + 0.0
        elif not need - 1e-9 <= tilt <= need + 30:
            return {"ok": False, "error": f"Can't reach x={x:.1f}, y={y:.1f}, height=[{h}] with tilt {tilt}."}
        self.tip, self.tilt = np.array([x, y, h], float), float(tilt)
        self.moves.append((x, y, h, float(tilt)))
        return {"ok": True, "tilt": float(tilt)} if self.says_lean else {"ok": True}

    def pixel(self, point=None):
        """Where the gripper camera sees the object (or `point`), or None if it is out of the picture."""
        P = self.obj if point is None else np.array(point, float)
        back, c_in, c_side, mount, f = self.cam
        out = self.tip[:2] - BASE
        out = np.append(out / np.linalg.norm(out), 0.0)
        up, th = np.array([0, 0, 1.0]), np.radians(self.tilt)
        a = -np.cos(th) * up + np.sin(th) * out          # where the gripper points
        e_r = np.cos(th) * out + np.sin(th) * up         # across it, in the arm's plane
        e_t = np.cross(up, out)                          # sideways
        C = self.tip - back * a + c_in * e_r + c_side * e_t
        z = np.cos(mount) * a + np.sin(mount) * e_r
        d = P - C
        if d @ z <= 0.5:
            return None
        u, v = 640 + f * (d @ e_t) / (d @ z), 360 + f * (d @ np.cross(z, e_t)) / (d @ z)
        return (float(u), float(v)) if 0 <= u < 1280 and 0 <= v < 720 else None


class Reply:
    def __init__(self, data, content=b""):
        self._data, self.content = data, content

    def json(self):
        return self._data
