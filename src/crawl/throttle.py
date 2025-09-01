import time


class Throttle:
    def __init__(self, min_interval_sec: float = 1.0):
        self.min_interval = float(min_interval_sec)
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()
