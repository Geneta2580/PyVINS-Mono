import time


class PerformanceStats:
    """Tracks per-frame timing and reports average / instantaneous FPS."""

    def __init__(self, name="system"):
        self.name = name
        self.frame_count = 0
        self._start_time = None
        self._last_tick_time = None
        self._instant_fps = 0.0
        self._sum_instant_fps = 0.0

    def tick(self):
        """Call once after each processed frame (e.g. one image)."""
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now

        if self._last_tick_time is not None:
            dt = now - self._last_tick_time
            if dt > 1e-9:
                self._instant_fps = 1.0 / dt
                self._sum_instant_fps += self._instant_fps

        self._last_tick_time = now
        self.frame_count += 1

    @property
    def instant_fps(self):
        return self._instant_fps

    def get_average_fps(self):
        """Overall throughput: total frames / total wall time."""
        if self.frame_count < 2 or self._start_time is None or self._last_tick_time is None:
            return 0.0
        elapsed = self._last_tick_time - self._start_time
        if elapsed <= 1e-9:
            return 0.0
        return (self.frame_count - 1) / elapsed

    def get_mean_instant_fps(self):
        """Arithmetic mean of per-frame instantaneous FPS (excludes first frame)."""
        n = self.frame_count - 1
        if n <= 0:
            return 0.0
        return self._sum_instant_fps / n

    def get_elapsed_seconds(self):
        if self._start_time is None or self._last_tick_time is None:
            return 0.0
        return self._last_tick_time - self._start_time

    def summary(self):
        return {
            "name": self.name,
            "frame_count": self.frame_count,
            "elapsed_sec": self.get_elapsed_seconds(),
            "average_fps": self.get_average_fps(),
            "mean_instant_fps": self.get_mean_instant_fps(),
            "last_instant_fps": self._instant_fps,
        }

    def format_summary(self):
        s = self.summary()
        return (
            f"【Performance】【{s['name']}】"
            f" frames={s['frame_count']}, "
            f"elapsed={s['elapsed_sec']:.2f}s, "
            f"avg_fps={s['average_fps']:.2f}, "
            f"mean_instant_fps={s['mean_instant_fps']:.2f}"
        )

    def reset(self):
        self.frame_count = 0
        self._start_time = None
        self._last_tick_time = None
        self._instant_fps = 0.0
        self._sum_instant_fps = 0.0
