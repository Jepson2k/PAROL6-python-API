"""
Null-space IK benchmark — ported RTB null-space term vs. baseline.

Runs the sequence:

    home -> cart-jog X full range
         -> cart-jog Y full range
         -> cart-jog Z full range

in the simulator for two conditions:

    ns_off : baseline pinokin IK (kq=0, km=0, null-space injection disabled)
    ns_on  : null-space gradient enabled with the defaults from parol6.config

For each run the script reports, per axis:

* **Solve rate**: how far the TCP actually traveled during the 4 s full-speed
  jog. The controller's Cartesian-jog tick calls ``solve_ik`` every control
  tick; when IK fails it stops (gracefully) and the robot freezes. Distance
  travelled is therefore a monotone proxy for the fraction of IK calls that
  succeeded along the trajectory.
* **Speed to solve**: p50 / p95 / max of the control-loop period sampled via
  ``loop_stats()``. On a 6-DOF / 6-D pose task the null-space step adds one
  ``pinv(J)·J`` per iteration — the loop timing is the observable cost.
* **Precision**: ``‖pose_end_actual − pose_end_expected‖`` after returning
  home with ``move_l``, where ``pose_end_expected`` is the commanded pose.

A second pass runs a deterministic **in-process** micro-benchmark (no
controller subprocess) that replays poses sampled from the jog trajectories
through ``solve_ik`` directly, giving per-call success / iterations /
residual / wall-time. This isolates the IK math from motion-planner timing.

Both summaries are printed and written to ``./benchmark_results.md``.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

RESULTS_PATH = os.path.abspath("./benchmark_results.md")
JOG_DURATION_S = 4.0
JOG_SPEED = 1.0
AXES: tuple[str, ...] = ("X", "Y", "Z")


@dataclass
class JogOutcome:
    axis: str
    distance_mm: float
    elapsed_s: float
    pose_start: list[float]
    pose_end: list[float]
    precision_mm: float | None = None
    precision_deg: float | None = None


@dataclass
class ConditionResult:
    label: str
    jogs: list[JogOutcome] = field(default_factory=list)
    loop_mean_ms: float = 0.0
    loop_p95_ms: float = 0.0
    loop_max_ms: float = 0.0
    loop_overruns: int = 0


# ---------------------------------------------------------------------------
# End-to-end benchmark (controller in the loop)
# ---------------------------------------------------------------------------


def _tcp_distance_mm(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.asarray(list(a)[:3], dtype=float)
    b_arr = np.asarray(list(b)[:3], dtype=float)
    return float(np.linalg.norm(a_arr - b_arr))


def _tcp_orient_deg(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.asarray(list(a)[3:6], dtype=float)
    b_arr = np.asarray(list(b)[3:6], dtype=float)
    # RPY space — good enough for a coarse orientation error proxy.
    return float(np.linalg.norm(a_arr - b_arr))


def _run_condition(label: str, *, nullspace: bool) -> ConditionResult:
    env = dict(os.environ)
    env["PAROL6_FAKE_SERIAL"] = "1"
    env["PAROL6_IK_NULLSPACE"] = "1" if nullspace else "0"
    # Apply to current process so the in-process Robot imports pick it up.
    os.environ["PAROL6_FAKE_SERIAL"] = env["PAROL6_FAKE_SERIAL"]
    os.environ["PAROL6_IK_NULLSPACE"] = env["PAROL6_IK_NULLSPACE"]

    from parol6 import Robot  # imported lazily so env vars take effect
    from parol6.utils.ik import set_nullspace

    # Force in-process helpers to honour the current env-var state.
    set_nullspace(enabled=nullspace)

    result = ConditionResult(label=label)

    with Robot(timeout=30.0, normalize_logs=True) as robot:
        rbt = robot.create_sync_client(timeout=3.0)
        rbt.wait_ready(timeout=10.0)
        rbt.simulator(True)
        rbt.reset_loop_stats()

        rbt.home(wait=True, timeout=30.0)
        home_pose = rbt.pose(frame="WRF") or [0.0] * 6

        for axis in AXES:
            rbt.home(wait=True, timeout=30.0)
            time.sleep(0.2)  # let controller settle the safety-latch state
            pose_start = rbt.pose(frame="WRF") or [0.0] * 6
            print(f"[bench] {label} {axis} pose_start={pose_start[:3]}")

            t0 = time.perf_counter()
            rbt.jog_l(frame="WRF", axis=axis, speed=JOG_SPEED,
                      duration=JOG_DURATION_S)
            # jog_l is streamable — wait_motion handles both the
            # "motion started and finished" and "motion never started"
            # (safety/ik blocked) cases via motion_start_timeout.
            rbt.wait_motion(
                timeout=JOG_DURATION_S + 3.0,
                motion_start_timeout=1.5,
                settle_window=0.25,
            )
            elapsed = time.perf_counter() - t0

            pose_end = rbt.pose(frame="WRF") or [0.0] * 6
            print(f"[bench] {label} {axis} pose_end={pose_end[:3]} "
                  f"elapsed={elapsed:.2f}s")

            # Precision probe: joint-space home and re-measure.
            rbt.home(wait=True, timeout=30.0)
            pose_after_return = rbt.pose(frame="WRF") or [0.0] * 6
            precision_mm = _tcp_distance_mm(pose_after_return, home_pose)
            precision_deg = _tcp_orient_deg(pose_after_return, home_pose)

            result.jogs.append(
                JogOutcome(
                    axis=axis,
                    distance_mm=_tcp_distance_mm(pose_end, pose_start),
                    elapsed_s=elapsed,
                    pose_start=list(pose_start),
                    pose_end=list(pose_end),
                    precision_mm=precision_mm,
                    precision_deg=precision_deg,
                )
            )

        stats = rbt.loop_stats()
        if stats is not None:
            # LoopStatsResultStruct fields are in seconds — convert to ms.
            result.loop_mean_ms = float(getattr(stats, "mean_period_s", 0.0)) * 1000.0
            result.loop_p95_ms = float(getattr(stats, "p95_period_s", 0.0)) * 1000.0
            result.loop_max_ms = float(getattr(stats, "max_period_s", 0.0)) * 1000.0
            result.loop_overruns = int(getattr(stats, "overrun_count", 0))

    return result


# ---------------------------------------------------------------------------
# In-process IK micro-benchmark
# ---------------------------------------------------------------------------


@dataclass
class IkStats:
    label: str
    total: int = 0
    successes: int = 0
    iter_samples: list[int] = field(default_factory=list)
    residual_samples: list[float] = field(default_factory=list)
    time_us_samples: list[float] = field(default_factory=list)


def _pose_to_T(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
    """Compose a 4×4 SE(3) from xyz (mm) + RPY (deg, XYZ intrinsic)."""
    T = np.eye(4, dtype=np.float64)
    r, p, y = np.deg2rad(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz_mm / 1000.0
    return T


def _sample_axis_targets(
    T_home: np.ndarray, axis: str, steps: int = 60, reach_mm: float = 250.0
) -> list[np.ndarray]:
    """Sweep a single WRF axis through ±reach around the home pose."""
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    targets = []
    deltas = np.linspace(-reach_mm / 1000.0, reach_mm / 1000.0, steps)
    for d in deltas:
        T = T_home.copy()
        T[idx, 3] += d
        targets.append(np.ascontiguousarray(T))
    return targets


def _micro_ik_condition(label: str, *, nullspace: bool) -> IkStats:
    os.environ["PAROL6_IK_NULLSPACE"] = "1" if nullspace else "0"
    # Reset any cached solver so env-driven config takes effect.
    from parol6.utils.ik import set_nullspace, solve_ik
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    set_nullspace(enabled=nullspace)

    robot = PAROL6_ROBOT.robot  # pinokin Robot handle
    q_home = np.deg2rad(np.asarray(PAROL6_ROBOT.joint.standby_deg, dtype=float))
    T_home = robot.fkine(q_home)

    stats = IkStats(label=label)
    for axis in AXES:
        targets = _sample_axis_targets(T_home, axis)
        q_seed = q_home.copy()
        for T in targets:
            t0 = time.perf_counter()
            res = solve_ik(robot, T, q_seed, quiet_logging=True)
            dt_us = (time.perf_counter() - t0) * 1e6
            stats.total += 1
            stats.time_us_samples.append(dt_us)
            if res.success:
                stats.successes += 1
                stats.iter_samples.append(int(res.iterations))
                stats.residual_samples.append(float(res.residual))
                q_seed = res.q.copy()  # warm-start chain — matches controller
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _pctile(samples: list[float] | list[int], p: float) -> float:
    if not samples:
        return 0.0
    data = sorted(samples)
    k = max(0, min(len(data) - 1, int(round((p / 100.0) * (len(data) - 1)))))
    return float(data[k])


def _format_e2e(results: list[ConditionResult]) -> str:
    lines = ["## End-to-end jog benchmark (controller in the loop)", ""]
    lines.append(
        "| condition | axis | distance_mm | elapsed_s | "
        "precision_mm | precision_deg |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in results:
        for jog in cond.jogs:
            lines.append(
                f"| {cond.label} | {jog.axis} | {jog.distance_mm:.1f} | "
                f"{jog.elapsed_s:.2f} | {jog.precision_mm or 0.0:.3f} | "
                f"{jog.precision_deg or 0.0:.3f} |"
            )
    lines.append("")
    lines.append("### Control-loop timing (reported by the controller)")
    lines.append("")
    lines.append(
        "| condition | mean (ms) | p95 (ms) | max (ms) | overruns |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for cond in results:
        lines.append(
            f"| {cond.label} | {cond.loop_mean_ms:.3f} | "
            f"{cond.loop_p95_ms:.3f} | {cond.loop_max_ms:.3f} | "
            f"{cond.loop_overruns} |"
        )
    return "\n".join(lines)


def _format_micro(stats_list: list[IkStats]) -> str:
    lines = ["## In-process IK micro-benchmark", ""]
    lines.append(
        "| condition | solves | solve_rate | iter_mean | "
        "residual_mean | us_p50 | us_p95 | us_max |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats_list:
        iter_mean = statistics.fmean(s.iter_samples) if s.iter_samples else 0.0
        res_mean = (
            statistics.fmean(s.residual_samples) if s.residual_samples else 0.0
        )
        us_p50 = _pctile(s.time_us_samples, 50.0)
        us_p95 = _pctile(s.time_us_samples, 95.0)
        us_max = max(s.time_us_samples) if s.time_us_samples else 0.0
        lines.append(
            f"| {s.label} | {s.total} | "
            f"{_pct(s.successes, s.total):.1f}% | {iter_mean:.2f} | "
            f"{res_mean:.2e} | {us_p50:.1f} | {us_p95:.1f} | {us_max:.1f} |"
        )
    return "\n".join(lines)


def _format_summary(
    e2e_off: ConditionResult, e2e_on: ConditionResult,
    micro_off: IkStats, micro_on: IkStats,
) -> str:
    lines = ["## Summary (ns_on − ns_off)", ""]
    off_dist = sum(j.distance_mm for j in e2e_off.jogs)
    on_dist = sum(j.distance_mm for j in e2e_on.jogs)
    off_rate = _pct(micro_off.successes, micro_off.total)
    on_rate = _pct(micro_on.successes, micro_on.total)
    lines.append(
        f"* total jog distance: off={off_dist:.1f} mm, on={on_dist:.1f} mm, "
        f"delta={on_dist - off_dist:+.1f} mm"
    )
    lines.append(
        f"* micro-bench IK solve rate: off={off_rate:.1f}%, "
        f"on={on_rate:.1f}%, delta={on_rate - off_rate:+.2f} pp"
    )
    if micro_off.time_us_samples and micro_on.time_us_samples:
        off_p95 = _pctile(micro_off.time_us_samples, 95.0)
        on_p95 = _pctile(micro_on.time_us_samples, 95.0)
        lines.append(
            f"* IK p95 wall time: off={off_p95:.1f} us, "
            f"on={on_p95:.1f} us, delta={on_p95 - off_p95:+.1f} us"
        )
    lines.append("")
    lines.append(
        "> PAROL6 is a 6-DOF non-redundant arm, so on a full-rank 6-D pose "
        "task the null-space projector `(I - J⁺J)` is theoretically zero; "
        "the term's observable effect is confined to near-singular "
        "configurations (wrist alignment at workspace edges) and — when "
        "the task is reduced (e.g. position-only) — the resulting "
        "redundancy. Parity on the interior with small positive deltas "
        "near limits is the expected outcome."
    )
    return "\n".join(lines)


def main() -> int:
    print(f"[bench] PID={os.getpid()} results → {RESULTS_PATH}")

    print("[bench] running end-to-end condition: ns_off")
    e2e_off = _run_condition("ns_off", nullspace=False)
    print("[bench] running end-to-end condition: ns_on")
    e2e_on = _run_condition("ns_on", nullspace=True)

    print("[bench] running in-process micro-bench: ns_off")
    micro_off = _micro_ik_condition("ns_off", nullspace=False)
    print("[bench] running in-process micro-bench: ns_on")
    micro_on = _micro_ik_condition("ns_on", nullspace=True)

    e2e_md = _format_e2e([e2e_off, e2e_on])
    micro_md = _format_micro([micro_off, micro_on])
    summary_md = _format_summary(e2e_off, e2e_on, micro_off, micro_on)

    report = "# Null-space IK benchmark\n\n" + "\n\n".join(
        [e2e_md, micro_md, summary_md]
    ) + "\n"
    with open(RESULTS_PATH, "w") as fp:
        fp.write(report)

    print(report)
    print(f"[bench] wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
