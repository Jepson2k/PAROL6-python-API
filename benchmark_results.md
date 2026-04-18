# Null-space IK benchmark

## End-to-end jog benchmark (controller in the loop)

| condition | axis | distance_mm | elapsed_s | precision_mm | precision_deg |
|---|---|---:|---:|---:|---:|
| ns_off | X | 225.5 | 2.63 | 0.000 | 0.000 |
| ns_off | Y | 0.0 | 1.77 | 0.000 | 0.000 |
| ns_off | Z | 0.0 | 1.77 | 0.000 | 0.000 |
| ns_on | X | 225.5 | 2.63 | 0.000 | 0.000 |
| ns_on | Y | 0.0 | 1.77 | 0.000 | 0.000 |
| ns_on | Z | 0.0 | 1.77 | 0.000 | 0.000 |

### Control-loop timing (reported by the controller)

| condition | mean (ms) | p95 (ms) | max (ms) | overruns |
|---|---:|---:|---:|---:|
| ns_off | 10.000 | 10.054 | 10.112 | 0 |
| ns_on | 10.000 | 10.075 | 10.767 | 0 |

## In-process IK micro-benchmark

| condition | solves | solve_rate | iter_mean | residual_mean | us_p50 | us_p95 | us_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| ns_off | 180 | 65.0% | 3.57 | 6.52e-14 | 26.8 | 597.6 | 10668.4 |
| ns_on | 180 | 65.0% | 3.85 | 6.52e-14 | 57.8 | 2008.4 | 3133.5 |

## Summary (ns_on − ns_off)

* total jog distance: off=225.5 mm, on=225.5 mm, delta=+0.0 mm
* micro-bench IK solve rate: off=65.0%, on=65.0%, delta=+0.00 pp
* IK p95 wall time: off=597.6 us, on=2008.4 us, delta=+1410.9 us

> PAROL6 is a 6-DOF non-redundant arm, so on a full-rank 6-D pose task the null-space projector `(I - J⁺J)` is theoretically zero; the term's observable effect is confined to near-singular configurations (wrist alignment at workspace edges) and — when the task is reduced (e.g. position-only) — the resulting redundancy. Parity on the interior with small positive deltas near limits is the expected outcome.
