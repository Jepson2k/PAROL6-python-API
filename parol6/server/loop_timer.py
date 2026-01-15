"""Loop timing with hybrid sleep + busy-loop for precise deadline scheduling."""

import time
from dataclasses import dataclass


@dataclass
class LoopMetrics:
    """Metrics tracked by the loop timer."""

    loop_count: int = 0
    overrun_count: int = 0
    last_period_s: float = 0.0
    ema_period_s: float = 0.0


class LoopTimer:
    """Deadline-based loop timing with hybrid sleep + busy-loop.

    Uses time.sleep() for most of the wait time to reduce CPU usage,
    then switches to a busy-loop for the final portion to achieve
    precise timing without OS scheduling jitter.
    """

    def __init__(self, interval_s: float, busy_threshold_s: float = 0.001):
        """Initialize the loop timer.

        Args:
            interval_s: Target loop interval in seconds.
            busy_threshold_s: Time before deadline to switch from sleep to busy-loop.
                             Default 1ms provides good precision without excessive CPU.
        """
        self._interval = interval_s
        self._busy_threshold = busy_threshold_s
        self._next_deadline = 0.0
        self._prev_t = 0.0
        self.metrics = LoopMetrics()

    @property
    def interval(self) -> float:
        """Target loop interval in seconds."""
        return self._interval

    def start(self) -> None:
        """Initialize timing at loop start. Call once before entering the loop."""
        now = time.perf_counter()
        self._next_deadline = now
        self._prev_t = now

    def wait_for_next_tick(self) -> None:
        """Wait until next deadline using hybrid sleep + busy-loop.

        Updates metrics (loop_count, period, EMA) and handles overruns.
        Call at the end of each loop iteration.
        """
        # Update metrics
        now = time.perf_counter()
        period = now - self._prev_t
        self._prev_t = now
        self.metrics.loop_count += 1
        self.metrics.last_period_s = period

        # EMA with alpha=0.1
        if self.metrics.ema_period_s <= 0.0:
            self.metrics.ema_period_s = period
        else:
            self.metrics.ema_period_s = 0.1 * period + 0.9 * self.metrics.ema_period_s

        # Advance deadline
        self._next_deadline += self._interval
        sleep_time = self._next_deadline - time.perf_counter()

        if sleep_time > self._busy_threshold:
            # Sleep for most of the time, leaving headroom for busy-loop
            time.sleep(sleep_time - self._busy_threshold)

        if sleep_time > 0:
            # Busy-loop for remaining time (precise timing)
            while time.perf_counter() < self._next_deadline:
                pass
        else:
            # Overrun - reset deadline to avoid perpetual catch-up
            self.metrics.overrun_count += 1
            self._next_deadline = time.perf_counter()
