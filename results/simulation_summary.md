# Simulation results

Held-out test set: 100 scenes × 120 s (seed 2026); parameters were tuned only on the dev set (40 scenes, seed 7). Match tolerance ±3.0 s; 95% CIs from 2000 scenario bootstraps.

## Baseline vs SentryScribe (test set)

| Noise | System | Precision | Recall | F1 [95% CI] | Start error (s) |
|---|---|---|---|---|---|
| clean | baseline | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 0.00 |
| clean | sentryscribe | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 0.18 |
| low | baseline | 0.284 | 0.746 | 0.411 [0.392, 0.429] | 0.31 |
| low | sentryscribe | 0.998 | 1.000 | 0.999 [0.997, 1.000] | 0.20 |
| medium | baseline | 0.224 | 0.590 | 0.325 [0.308, 0.340] | 0.42 |
| medium | sentryscribe | 0.993 | 1.000 | 0.997 [0.993, 0.999] | 0.24 |
| high | baseline | 0.185 | 0.431 | 0.259 [0.242, 0.276] | 0.27 |
| high | sentryscribe | 0.963 | 0.970 | 0.967 [0.952, 0.980] | 0.33 |

## SentryScribe F1 per event type (test set)

| Event type | clean | low | medium | high |
|---|---|---|---|---|
| ABANDONED_OBJECT | 1.000 | 1.000 | 1.000 | 0.909 |
| CROWDING | 1.000 | 1.000 | 1.000 | 0.800 |
| LINE_CROSSING | 1.000 | 0.998 | 0.993 | 0.988 |
| LOITERING | 1.000 | 1.000 | 1.000 | 0.933 |
| VEHICLE_IN_PED_ZONE | 1.000 | 1.000 | 1.000 | 1.000 |
| ZONE_INTRUSION | 1.000 | 1.000 | 0.997 | 0.991 |

## Ablations (test set, medium noise)

| Removed | Precision | Recall | F1 [95% CI] |
|---|---|---|---|
| nothing (full system) | 0.993 | 1.000 | 0.997 [0.993, 0.999] |
| no hysteresis (cooldown 0 s) | 0.291 | 0.592 | 0.390 [0.371, 0.409] |
| no ID stitching | 0.797 | 0.872 | 0.833 [0.813, 0.851] |
| no position smoothing | 0.981 | 0.998 | 0.990 [0.982, 0.996] |
| no line dead-band | 0.855 | 1.000 | 0.922 [0.908, 0.936] |
| no line debounce | 0.993 | 1.000 | 0.997 [0.993, 0.999] |
| no minimum track age | 0.993 | 1.000 | 0.997 [0.993, 0.999] |
| no bag-attendance gate | 0.979 | 0.986 | 0.983 [0.973, 0.991] |
