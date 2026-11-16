"""Encoder-based descent stop evidence; this is not a force/contact sensor."""
from collections import deque
import math
from statistics import median


class DescentContact:
    """Require sustained stationary feedback, not just a changing tracking lag.

    A free downward segment must precede a contact inference. A stall before
    that is indistinguishable from a blocked/disabled arm and must not close jaws.
    Times and heights refer to distinct measured samples (seconds and cm).
    """

    def __init__(self, speed_cm_s, threshold_cm=.6, blank_s=.45):
        if not all(math.isfinite(v) and v > 0 for v in (speed_cm_s, threshold_cm, blank_s)):
            raise ValueError('Invalid descent detector settings')
        self.threshold = threshold_cm
        self.blank = blank_s
        self.window = max(.24, .35 / speed_cm_s)
        self.history = deque()
        self.free_gaps = deque(maxlen=16)
        self.free_travel = 0.
        self.over = 0
        self.initial = None

    def update(self, elapsed_s, target_cm, measured_cm):
        if not all(math.isfinite(v) for v in (elapsed_s, target_cm, measured_cm)):
            raise ValueError('Nonfinite descent feedback')
        if self.history and elapsed_s <= self.history[-1][0]:
            raise ValueError('Descent samples must have increasing timestamps')
        if self.initial is None:
            self.initial = (elapsed_s, target_cm, measured_cm)
        self.history.append((elapsed_s, target_cm, measured_cm))
        # Keep the last sample at/before the window boundary as its anchor.
        while len(self.history) > 2 and self.history[1][0] <= elapsed_s - self.window:
            self.history.popleft()
        first = self.history[0]
        gap = measured_cm - target_cm
        ready = elapsed_s - first[0] >= self.window - 1e-9 and len(self.history) >= 4
        advance = first[1] - target_cm
        travel = first[2] - measured_cm
        span = max(p[2] for p in self.history) - min(p[2] for p in self.history)
        if ready and travel > .12 and advance > .1:
            # Learn lag only while actually descending, never from stalled data.
            self.free_gaps.append(gap)
            self.free_travel = max(self.free_travel, self.initial[2] - measured_cm)
        baseline = median(self.free_gaps) if self.free_gaps else self.initial[2] - self.initial[1]
        stationary = ready and span <= .10 and advance >= .30
        candidate = elapsed_s >= self.blank and stationary and gap - baseline > self.threshold
        self.over = self.over + 1 if candidate else 0
        reason = 'descending' if ready and travel > .12 else 'observing'
        if self.over >= 2:
            reason = 'contact_inferred' if self.free_travel >= .25 and self.free_gaps else 'blocked_before_descent'
        return {'reason': reason, 'contact': reason == 'contact_inferred',
                'gap_cm': gap, 'free_lag_cm': baseline, 'window_travel_cm': travel,
                'window_span_cm': span, 'command_advance_cm': advance,
                'free_travel_cm': self.free_travel, 'stationary': stationary}
