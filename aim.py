"""From an outline in the fixed camera's picture to the table point the arm should go to.

Two things decide whether the jaws come down around a small object or beside it:

1. Pixel -> table. The arm page turns a pixel into table x, y with a flat mapping (a homography).
   A wide-angle lens bends the picture, so that mapping is off by up to a few cm away from the
   calibration marks. TableMap does what the page does, but first straightens the picture with a
   lens correction it works out from the same calibration marks (a checkerboard's rows must come
   out straight). The correction is only used if it clearly predicts held-out marks better;
   otherwise TableMap gives exactly the page's numbers.

2. Which point of the outline. The camera looks at the table at a slant, so an outline is the
   object's footprint plus its "shadow": everything above the table shows further away from the
   camera. The lowest part of the outline is only the footprint's near EDGE. footprint() gives the
   MIDDLE of where the object stands, its width, and a height estimate; flat things (a blister
   pack, a card) are recognised and aimed at their centre.

No arm, camera or model is needed here: numbers in, numbers out (see test_aim.py).
"""
import numpy as np
import cv2

import height as cam_geometry

BOARD_MARKS = 54        # one checkerboard position (9 x 6 corners), as in controller/arm/click.py
BLEND_PX = 18.0         # ... and how its per-board mappings are blended there
MIN_LENS_MARKS = 24     # fewer marks than this can't tell a bent picture from loose clicks
MIN_CAMERA_MARKS = 8    # fewer marks than this can't tell where the camera sits (for heights)
LENS_GAIN = 0.85        # a lens correction must cut the held-out error to this fraction to be used ...
LENS_MAX_ERR_CM = 0.5   # ... and leave no more than this: loose marks can't say how a lens bends
FLAT_CM = 1.0           # estimated height below this: a flat thing, aim at the middle of its outline
MAX_HEIGHT_CM = 30.0    # taller than this isn't believable on this table
GRIP_DEPTH_CM = 4.5     # what the jaws can take: a graspable thing is at most about this deep


# ---------- pixel -> table ----------

def fit_h(src, dst):
    """3x3 matrix mapping src points to dst points: the same least squares as the arm page."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    A = np.zeros((2 * n, 8))
    A[0::2, 0:2], A[0::2, 2] = src, 1
    A[0::2, 6:8] = -dst[:, :1] * src
    A[1::2, 3:5], A[1::2, 5] = src, 1
    A[1::2, 6:8] = -dst[:, 1:] * src
    h = np.linalg.lstsq(A, dst.reshape(-1), rcond=None)[0]
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def straighten(px, k, centre, size):
    """Pixels as a lens without bending would show them. k = (k1, k2); all zero = unchanged."""
    d = np.asarray(px, float) - centre
    r2 = (d ** 2).sum(axis=1, keepdims=True) / size ** 2
    return centre + d * (1 + k[0] * r2 + k[1] * r2 ** 2)


def _groups(n):
    """Marks that were laid down together: one slice per checkerboard position, else all in one."""
    if n >= 2 * BOARD_MARKS and n % BOARD_MARKS == 0:
        return [slice(i, i + BOARD_MARKS) for i in range(0, n, BOARD_MARKS)]
    return [slice(0, n)]


def _bend(k, px, xy, groups, centre, size):
    """How far the marks are from a flat mapping after straightening with k (cm, per coordinate).
    Each board gets its own mapping, so where the arm thought each board lay doesn't count: only
    bending of the picture is left."""
    u = straighten(px, k, centre, size) / size
    return np.concatenate([(apply_h(fit_h(u[g], xy[g]), u[g]) - xy[g]).ravel() for g in groups])


def fit_lens(px, xy, groups, centre, size, terms):
    """Lens bending (k1, k2) that makes the marks fit flat mappings best. terms: 1 or 2 numbers used."""
    k = np.zeros(2)
    r = _bend(k, px, xy, groups, centre, size)
    cost, lam, eps = float(r @ r), 1e-3, 1e-4
    for _ in range(60):
        J = np.column_stack([(_bend(k + eps * np.eye(2)[i], px, xy, groups, centre, size) - r) / eps
                             for i in range(terms)])
        A = J.T @ J
        try:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(terms), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        trial = k.copy()
        trial[:terms] += step
        r_new = _bend(trial, px, xy, groups, centre, size)
        new = float(r_new @ r_new)
        if not np.isfinite(new) or new >= cost:
            lam *= 10
            if lam > 1e8:
                break
            continue
        done = cost - new < 1e-10 * max(cost, 1e-12)
        k, r, cost, lam = trial, r_new, new, lam * 0.3
        if done:
            break
    return k


def _believable(k, r_max=0.62):
    """A real lens: the straightened picture doesn't fold over itself anywhere in the image, and
    nothing is moved to less than half or more than three times its distance from the middle."""
    r2 = np.linspace(0, r_max, 40) ** 2
    grow = 1 + k[0] * r2 + k[1] * r2 ** 2
    return bool(np.all(np.isfinite(k)) and np.all(1 + 3 * k[0] * r2 + 5 * k[1] * r2 ** 2 > 0.2)
                and grow.min() > 0.5 and grow.max() < 3.0)


class TableMap:
    """Pixel -> table x, y (cm) from the arm page's calibration marks [u, v, x_cm, y_cm].

    Pixels are in the calibration's own scale (the image width the marks were made at = `size`).
    lens=False gives the arm page's mapping exactly."""

    def __init__(self, pairs, size, image_hw=None, lens=True):
        P = np.asarray(pairs, float).reshape(-1, 4)
        if len(P) < 4:
            raise ValueError("needs at least 4 calibration marks")
        self.size = float(size)
        aspect = (image_hw[0] / image_hw[1]) if image_hw else 9 / 16
        self.height_px = self.size * aspect
        self.centre = np.array([self.size / 2, self.height_px / 2])
        self.px, self.xy = P[:, :2], P[:, 2:]
        self.groups = _groups(len(P))
        self.k = np.zeros(2)
        self.lens_note = "no lens correction"
        if lens and len(P) >= MIN_LENS_MARKS:
            self._choose_lens()
        self._fit()

    # ----- lens -----

    def _held_out(self, terms):
        """Error (cm, RMS) on marks that were left out when fitting: every other mark, both ways."""
        errs = []
        idx = np.arange(len(self.px))
        for fold in (0, 1):
            train, test = [], []
            for g in self.groups:
                i = idx[g]
                train.append(i[(i - i[0]) % 2 == fold])
                test.append(i[(i - i[0]) % 2 != fold])
            tr = np.concatenate(train)
            starts = np.cumsum([0] + [len(t) for t in train])
            tr_groups = [slice(starts[j], starts[j + 1]) for j in range(len(train))]
            k = (fit_lens(self.px[tr], self.xy[tr], tr_groups, self.centre, self.size, terms)
                 if terms else np.zeros(2))
            if not _believable(k):
                return np.inf
            for a, b in zip(train, test):
                H = fit_h(straighten(self.px[a], k, self.centre, self.size) / self.size, self.xy[a])
                got = apply_h(H, straighten(self.px[b], k, self.centre, self.size) / self.size)
                errs.append(np.linalg.norm(got - self.xy[b], axis=1))
        e = np.concatenate(errs)
        return float(np.sqrt((e ** 2).mean()))

    def _choose_lens(self):
        try:
            best, used = self._held_out(0), 0
            flat = best
            for terms in (1, 2):
                e = self._held_out(terms)
                if e < LENS_GAIN * best and e <= LENS_MAX_ERR_CM:
                    best, used = e, terms
            if used:
                k = fit_lens(self.px, self.xy, self.groups, self.centre, self.size, used)
                if _believable(k):
                    self.k = k
                    self.lens_note = (f"lens correction on: left-out marks land within {best:.2f} cm "
                                      f"(flat mapping: {flat:.2f} cm)")
                    return
            self.lens_note = (f"no lens correction (left-out marks land within {flat:.2f} cm"
                              + (")" if flat <= LENS_MAX_ERR_CM else "; the marks are too loose to tell)"))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            self.k = np.zeros(2)
            self.lens_note = "no lens correction (couldn't work one out from these marks)"

    @property
    def lens_on(self):
        return bool(np.any(self.k != 0))

    # ----- mapping -----

    def straight(self, px):
        return straighten(np.asarray(px, float).reshape(-1, 2), self.k, self.centre, self.size)

    def _fit(self):
        s = self.straight(self.px)
        self.H = fit_h(s / self.size, self.xy)
        self.boards = []
        if len(self.groups) > 1:
            for g in self.groups:
                b = s[g]
                self.boards.append((fit_h(b / self.size, self.xy[g]),
                                    b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()))
        self.hull = cv2.convexHull(self.px.astype(np.float32))
        self.cam = None   # where the camera sits, for heights (None if these marks can't tell)
        if len(self.px) >= MIN_CAMERA_MARKS:
            try:
                cam = cam_geometry.camera_from_pairs(np.c_[s, self.xy], (self.size, self.height_px))
                f, C = cam[0][0, 0], cam[3]
                if np.all(np.isfinite(C)) and 5 < abs(C[2]) < 300 and 0.25 * self.size < f < 4 * self.size:
                    self.cam = cam
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                pass

    def locate(self, px):
        """Table x, y (cm) of pixels (N x 2, or one pair)."""
        px = np.asarray(px, float)
        one = px.ndim == 1
        s = self.straight(px)
        n = s / self.size
        if not self.boards:
            out = apply_h(self.H, n)
        else:   # each board's own mapping where it lies, shared smoothly in between (as the page does)
            w = []
            for _, u0, u1, v0, v1 in self.boards:
                du = np.maximum(np.maximum(u0 - s[:, 0], 0), s[:, 0] - u1)
                dv = np.maximum(np.maximum(v0 - s[:, 1], 0), s[:, 1] - v1)
                w.append(1.0 / (du * du + dv * dv + BLEND_PX ** 2) ** 2)
            w = np.array(w)
            w /= w.sum(axis=0)
            out = sum(wk[:, None] * apply_h(H, n) for wk, (H, *_) in zip(w, self.boards))
        return out[0] if one else out

    def pixel_of(self, xy, start=None):
        """The pixel that shows table point xy (for drawing): locate() run backwards."""
        xy = np.asarray(xy, float)
        p = np.array(start if start is not None else self.centre, float)
        for _ in range(25):
            here = self.locate(p)
            J = np.column_stack([(self.locate(p + d) - here) / 2.0 for d in ((2.0, 0), (0, 2.0))])
            try:
                step = np.linalg.solve(J, xy - here)
            except np.linalg.LinAlgError:
                break
            n = np.linalg.norm(step)
            p = p + step * min(1.0, 200.0 / max(n, 1e-9))
            if n < 0.01:
                break
        return float(p[0]), float(p[1])

    def covers(self, px, margin_px):
        """Is the pixel on the part of the table the marks cover (plus a margin)? Outside it the
        mapping is a guess: something in the background can land 'within reach'."""
        return cv2.pointPolygonTest(self.hull, (float(px[0]), float(px[1])), True) >= -margin_px


# ---------- outline -> where it stands ----------

WIDENING_FROM_CM = 5.0   # the "does it get wider towards its top?" check: only for things this tall ...
WIDENING_MIN_PX = 24     # ... and wide enough in the picture to measure
WIDENING_LYING = 0.10    # less than this share of the widening its height would need: it's lying down


def _widest(a, s, a_near, depth, slices=10):
    """Widest place of the outline in its near half and in its far half (cm), or None."""
    edges = np.linspace(a_near, a_near + depth, slices + 1)
    w = []
    for i in range(slices):
        sel = (a >= edges[i]) & (a <= edges[i + 1])
        w.append(float(np.ptp(s[sel])) if sel.sum() >= 4 else 0.0)
    near, far = max(w[:slices // 2]), max(w[slices // 2:])
    return (near, far) if near > 0 else None

def _main_part(mask):
    """The biggest connected piece of the outline (drops specks)."""
    m = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _polygon_centre(poly):
    """Middle (by area) of a closed outline given as N x 2 points."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * area)


def footprint(mask, tmap, image_width, offset=(0.0, 0.0), zoom=1.0):
    """Where the object in `mask` stands on the table.

    `mask` is in the picture's own pixels, or (offset, zoom given) an enlarged cut-out of it:
    cut-out pixel c shows picture pixel offset + c / zoom.

    Returns a dict (all cm, table coordinates):
      center   the point to aim at: the middle of where it stands (flat things: middle of the outline)
      near     the middle of its edge nearest the camera
      width    how wide it is, seen from the camera
      depth    how deep its outline reaches on the table (footprint + what its height adds)
      height   estimated height, or None if it can't be told (flat things: None)
      flat     True for something lying flat (under about 1 cm)
      how      one of "stands", "flat", "long" (lying, pointing towards the camera), "unknown height"
    or None if the outline is empty.
    """
    m = _main_part(mask)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    scale = tmap.size / float(image_width)
    off = np.asarray(offset, float)

    def calib_px(cols_rows):                         # mask pixels -> the calibration's pixels
        return (off + (np.asarray(cols_rows, float) + 0.5) / zoom - 0.5) * scale
    if len(xs) > 8000:                               # plenty for the numbers below
        pick = np.linspace(0, len(xs) - 1, 8000).astype(int)
        xs, ys = xs[pick], ys[pick]
    P = tmap.locate(calib_px(np.column_stack([xs, ys])))
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(cnts, key=len).reshape(-1, 2).astype(float)
    middle = _polygon_centre(tmap.locate(calib_px(outline))) if len(outline) >= 3 else P.mean(axis=0)

    # which way is "away from the camera" on the table, at this object
    if tmap.cam is not None:
        C = tmap.cam[3]
        away = P.mean(axis=0) - C[:2]
    else:   # camera position unknown: up in the picture is away
        u0, v0 = calib_px([xs.mean(), ys.mean()])
        away = tmap.locate([u0, v0 - 4.0]) - tmap.locate([u0, v0 + 4.0])
    if not np.all(np.isfinite(away)) or np.linalg.norm(away) < 1e-9:
        return None
    away = away / np.linalg.norm(away)
    side = np.array([-away[1], away[0]])
    a, s = P @ away, P @ side
    a_near, a_far = np.percentile(a, [0.5, 99.5])
    depth = float(a_far - a_near)

    # its width: measured across the part nearest the camera (further back, its height widens the outline)
    lo, hi = np.percentile(s, [0.5, 99.5])
    width = float(hi - lo)
    for _ in range(2):
        band = a <= a_near + max(0.5 * width, 0.25 * depth)
        if band.sum() < 12:
            break
        lo, hi = np.percentile(s[band], [0.5, 99.5])
        width = float(hi - lo)
    stand_depth = min(width, GRIP_DEPTH_CM, depth)   # how deep the footprint itself is, at a guess
    near = away * a_near + side * (lo + hi) / 2

    h = h_round = grows = None
    if tmap.cam is not None:
        cz = abs(float(C[2]))
        d_near = a_near - float(C[:2] @ away)
        if d_near > 1 and cz > 1:
            # Its top's far edge shows (d_near + depth) from the camera's foot instead of
            # (d_near + footprint depth): that stretch is what its height adds.
            h_round = cz * (depth - width) / (d_near + depth)                     # if it's as deep as wide
            h = cz * (depth - min(width, GRIP_DEPTH_CM)) / (d_near + depth)       # if it's at most jaw-deep
            if h >= WIDENING_FROM_CM and np.ptp(xs) / zoom >= WIDENING_MIN_PX:
                # Something that stands shows wider at its top (nearer the camera) than at its base;
                # something lying on the table doesn't. `grows`: seen widening / widening its height needs.
                w = _widest(a, s, a_near, depth)
                if w is not None:
                    grows = (w[1] / w[0] - 1) / max(1e-6, cz / max(cz - h, 1e-6) - 1)
    out = {"near": (float(near[0]), float(near[1])), "width": width, "depth": depth,
           "away": (float(away[0]), float(away[1]))}
    lying = h is not None and (h > MAX_HEIGHT_CM or (grows is not None and grows < WIDENING_LYING))
    if h is None:
        c = near + away * stand_depth / 2
        out.update(center=(float(c[0]), float(c[1])), height=None, flat=False, how="unknown height")
    elif h_round < FLAT_CM or lying:
        # lying flat (or lying and pointing at the camera): what the camera sees is where it is
        out.update(center=(float(middle[0]), float(middle[1])), height=None, flat=True,
                   how="long" if lying and h_round >= FLAT_CM else "flat")
    else:
        c = near + away * stand_depth / 2
        out.update(center=(float(c[0]), float(c[1])), height=float(h), flat=False, how="stands")
    return out
