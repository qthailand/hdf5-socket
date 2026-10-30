"""Per-stage timing of the GPU backends, taken only on profiler-sampled frames.

A backend asks :func:`stage_timer` for a timer once per frame and calls
``mark(i)`` at the stage boundaries 0-6 (before upload, after upload, dark,
hot pixel, Smile, binning, download), then ``report(profiler)``.  The timer is
a no-op unless ``profiler.want_detail()`` is true and no Grab was requested
(a Grab adds a blocking download in the middle of the pipeline).
"""
import time

#: Span names recorded, in pipeline order.
STAGES = ('gpu_upload', 'gpu_dark', 'gpu_hot_pixel', 'gpu_smile', 'gpu_bin', 'gpu_download')


class CudaEventTimer:
    """CUDA events between stages; read after the final download, so no extra synchronisation."""

    def __init__(self, cp):
        """Args: cp: the ``cupy`` module."""
        self.cp = cp
        self.events = [cp.cuda.Event() for _ in range(len(STAGES) + 1)]

    def mark(self, index):
        """Record boundary ``index`` on the current stream."""
        self.events[index].record()

    def report(self, profiler):
        """Record each stage's device time (ms) through ``profiler._record``."""
        self.events[-1].synchronize()
        for name, start, end in zip(STAGES, self.events, self.events[1:]):
            profiler._record(name, float(self.cp.cuda.get_elapsed_time(start, end)))


class HostClockTimer:
    """Host timestamps between stages (OpenCV CUDA).

    With asynchronous stream work, pass ``sync`` (e.g. ``stream.waitForCompletion``)
    so each boundary waits for the queued work; only sampled frames pay for it.
    """

    def __init__(self, sync=None):
        """Args: sync: Optional zero-argument callable run before each timestamp."""
        self.sync = sync
        self.times = [0.0] * (len(STAGES) + 1)

    def mark(self, index):
        """Wait for queued work (if ``sync`` is set), then take the timestamp of boundary ``index``."""
        if self.sync is not None:
            self.sync()
        self.times[index] = time.perf_counter()

    def report(self, profiler):
        """Record each stage's wall time (ms) through ``profiler._record``."""
        for name, start, end in zip(STAGES, self.times, self.times[1:]):
            profiler._record(name, (end - start) * 1000.0)


class NoTimer:
    """Timer used on frames that are not sampled."""

    def mark(self, index):
        """Do nothing."""

    def report(self, profiler):
        """Do nothing."""


NO_TIMER = NoTimer()


def stage_timer(profiler, snapshot, make):
    """Return ``make()`` on a sampled frame without Grab, else :data:`NO_TIMER`.

    Args:
        profiler: Profiler passed to the backend; sampled when its optional
            ``want_detail()`` returns true.
        snapshot: Whether this frame also produces a Grab.
        make: Zero-argument factory of the real timer.
    """
    want = getattr(profiler, 'want_detail', None)
    if snapshot or want is None or not want():
        return NO_TIMER
    return make()
