"""Simulation study for the SentryScribe event engine.

Generates synthetic CCTV scenes (intruders, loiterers, line crossers, abandoned bags, crowds,
vehicles in the walkway, plus background pedestrians) with exactly known ground truth, degrades
the perfect tracks with tracker-like noise (box jitter, missed detections, ID switches, false
detections), runs the *same* event engine as notebooks/colab_demo.ipynb on the result, and scores
it with the notebook's evaluate().

Ground truth comes from an independent labeller that applies the event definitions to the clean
trajectories, so the study measures how well the engine copes with perception noise.

Usage:  python scripts/simulate.py [--scenarios 40] [--seed 7]
Writes: results/simulation_metrics.json, results/simulation_summary.md, results/*.png
"""
import argparse
import json
import math
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_demo.ipynb"
RESULTS = ROOT / "results"

W, H, FPS, T_END = 1280, 720, 10, 120.0
SIZES = {"person": (40, 110), "car": (160, 90), "backpack": (30, 30)}

NOISE_LEVELS = {
    #          jitter px   miss prob   ID switches / track / s   false tracks / frame
    "clean":  dict(jitter=0.0, miss=0.00, idsw=0.000, fp=0.00),
    "low":    dict(jitter=2.0, miss=0.05, idsw=0.005, fp=0.05),
    "medium": dict(jitter=5.0, miss=0.15, idsw=0.020, fp=0.20),
    "high":   dict(jitter=10.0, miss=0.30, idsw=0.050, fp=0.50),
}


def load_notebook_code():
    # Execute the notebook's CONFIG, engine and eval cells so the study runs the exact shipped code.
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    ns = {}
    for head in ("CONFIG = dict(", "# [engine]", "# [eval]"):
        exec(next(s for s in cells if s.startswith(head)), ns)
    return ns


# ------------------------------------------------------------------ scene actors

class Actor:
    def __init__(self, cls, wps):
        self.cls = cls
        self.wps = [w for w in wps if w[0] <= T_END + 1]
        self.t = np.array([w[0] for w in self.wps])
        self.x = np.array([w[1] for w in self.wps])
        self.y = np.array([w[2] for w in self.wps])

    def present(self, t):
        return self.t[0] <= t <= self.t[-1]

    def foot(self, t):
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y))


def walk(points, t0, speed, pauses=None):
    # Waypoints for moving through `points` at `speed` px/s; pauses[i] = seconds to wait at points[i].
    pauses = pauses or {}
    t, wps = t0, [(t0, *points[0])]
    for i in range(1, len(points)):
        t += math.dist(points[i - 1], points[i]) / speed
        wps.append((t, *points[i]))
        if pauses.get(i):
            t += pauses[i]
            wps.append((t, *points[i]))
    return wps


def make_scenario(rng):
    u = rng.uniform
    actors = []
    for _ in range(rng.choice([1, 2])):                       # intruders from the right edge
        y, tx = u(420, 650), u(800, 1150)
        actors.append(Actor("person", walk([(1270, y), (tx, y + u(-20, 20)), (1270, y)], u(0, 80),
                                           u(45, 70), {1: u(3, 25)})))
    for _ in range(rng.choice([1, 2, 3])):                    # line crossers in the gap between zones
        x = u(610, 730)
        pts = [(x, 200), (x + u(-20, 20), 715)]
        if rng.random() < 0.3:
            pts.reverse()                                     # walking up = wrong way
        actors.append(Actor("person", walk(pts, u(0, 100), u(45, 70))))
    if rng.random() < 0.5:                                    # loiterer wandering in the walkway
        t0, stay = u(0, 15), u(65, 100)
        pts = [(10, u(450, 690))] + [(u(120, 520), u(450, 690)) for _ in range(int(stay / 6))] + [(10, u(450, 690))]
        actors.append(Actor("person", walk(pts, t0, u(15, 25), {i: 1.5 for i in range(1, len(pts) - 1)})))
    if rng.random() < 0.6:                                    # bag drop in the upper band
        y = u(230, 320)
        bx = u(250, 480) if rng.random() < 0.5 else u(860, 1050)
        owner = walk([(10, y), (bx, y), (1270, y + u(-10, 10))], u(0, 60), u(45, 70), {1: u(1, 2)})
        actors.append(Actor("person", owner))
        actors.append(Actor("backpack", [(t, x + 25, yy) for t, x, yy in owner[:2]] + [(T_END, bx + 25, y)]))
    if rng.random() < 0.35:                                   # crowd gathering in the walkway
        n, t0 = 10 + rng.randint(0, 3), u(10, 60)
        arrivals = sorted(t0 + u(0, 10) for _ in range(n))
        leave = arrivals[-1] + u(10, 20)
        for ta in arrivals:
            start, spot = (10, u(450, 690)), (u(120, 520), u(450, 690))
            wps = walk([start, spot], ta, u(45, 70))
            wps.append((leave, *spot))
            wps += walk([spot, start], leave, u(45, 70))[1:]
            actors.append(Actor("person", wps))
    if rng.random() < 0.5:                                    # car reversing into the walkway
        y = u(600, 690)
        actors.append(Actor("car", walk([(20, y), (u(250, 450), y), (20, y)], u(0, 90), 150, {1: u(3, 8)})))
    for _ in range(rng.randint(2, 6)):                        # background pedestrians, far from everything
        y, pts = u(40, 110), [(10, 0), (1270, 0)]
        pts = [(x, y) for x, _ in (pts if rng.random() < 0.5 else pts[::-1])]
        actors.append(Actor("person", walk(pts, u(0, 100), u(45, 70))))
    return actors


# ------------------------------------------------------------------ ground truth (clean geometry)

def segments(times):
    # Group consecutive frame times into (start, end) runs.
    runs, dt = [], 1.0 / FPS + 1e-6
    for t in times:
        if runs and t - runs[-1][1] <= dt:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return runs


def label_ground_truth(actors, zones, lines, cfg, video_id):
    frames = np.round(np.arange(0, T_END, 1.0 / FPS), 3)
    gt = []
    radius = cfg["abandon_radius"] * W
    zone_hits = defaultdict(list)            # (actor, zone) -> frame times inside
    count_hits = defaultdict(list)           # zone -> frame times with >= crowd_n people
    bag_alone = defaultdict(list)
    for t in frames:
        live = [(i, a, a.foot(t)) for i, a in enumerate(actors) if a.present(t)]
        persons = [f for _, a, f in live if a.cls == "person"]
        per_zone = defaultdict(int)
        for i, a, f in live:
            for z in zones:
                if z["poly"].contains(Point(f)):
                    zone_hits[(i, z["name"])].append(t)
                    per_zone[z["name"]] += a.cls == "person"
            if a.cls == "backpack":
                prev = a.foot(t - 1.0 / FPS)
                still = math.dist(prev, f) < 1e-6 and a.present(t - 1.0 / FPS)
                if still and all(math.dist(p, f) > radius for p in persons):
                    bag_alone[i].append(t)
        for zname, n in per_zone.items():
            if n >= cfg["crowd_n"]:
                count_hits[zname].append(t)

    def add(etype, zone, start, end):
        gt.append(dict(video=f"{video_id}.mp4", type=etype, zone=zone or "", start_s=float(start), end_s=float(end)))

    zone_type = {z["name"]: z["type"] for z in zones}
    for (i, zname), ts in zone_hits.items():
        cls = actors[i].cls
        for s, e in segments(ts):
            if zone_type[zname] == "restricted" and cls in ("person", "car") and e - s >= cfg["intrusion_s"]:
                add("ZONE_INTRUSION", zname, s, e)
            if zone_type[zname] == "pedestrian_only" and cls == "car" and e - s >= cfg["veh_ped_s"]:
                add("VEHICLE_IN_PED_ZONE", zname, s, e)
            if cls == "person" and e - s >= cfg["loiter_s"]:
                add("LOITERING", zname, s, e)
    for zname, ts in count_hits.items():
        for s, e in segments(ts):
            if e - s >= cfg["crowd_s"]:
                add("CROWDING", zname, s, e)
    for i, ts in bag_alone.items():
        for s, e in segments(ts):
            if e - s >= cfg["abandon_s"]:
                add("ABANDONED_OBJECT", None, s, e)
    for a in actors:
        if a.cls not in ("person", "car"):
            continue
        prev = None
        for t in frames:
            if not a.present(t):
                prev = None
                continue
            f = a.foot(t)
            if prev is not None and prev != f:
                for ln in lines:
                    if LineString([prev, f]).intersects(ln["geom"]):
                        add("LINE_CROSSING", ln["name"], t, t)
            prev = f
    return gt


# ------------------------------------------------------------------ noisy perception

def perceive(actors, noise, rng, cfg):
    # Yield (t, dets) per frame, imitating detector + tracker errors.
    next_id = [1]

    def new_id():
        next_id[0] += 1
        return next_id[0] - 1

    tid = {i: new_id() for i in range(len(actors))}
    false_tracks = []
    for t in np.round(np.arange(0, T_END, 1.0 / FPS), 3):
        dets = []
        for i, a in enumerate(actors):
            if not a.present(t):
                continue
            if rng.random() < noise["idsw"] / FPS:
                tid[i] = new_id()
            if rng.random() < noise["miss"] * (2 if a.cls == "backpack" else 1):
                continue
            fx, fy = a.foot(t)
            fx += rng.gauss(0, noise["jitter"])
            fy += rng.gauss(0, noise["jitter"])
            bw, bh = SIZES[a.cls]
            s = 1 + rng.gauss(0, 0.03 if noise["jitter"] else 0)
            dets.append((tid[i], a.cls, rng.uniform(0.5, 0.95), (fx - bw * s / 2, fy - bh * s, fx + bw * s / 2, fy)))
        for _ in range(np.random.default_rng(rng.randrange(1 << 30)).poisson(noise["fp"])):
            cls = rng.choices(["person", "car", "backpack"], [0.7, 0.15, 0.15])[0]
            false_tracks.append(dict(id=new_id(), cls=cls, left=rng.randint(1, 5),
                                     x=rng.uniform(0, W), y=rng.uniform(100, H), conf=rng.uniform(0.3, 0.6)))
        for ft in false_tracks:
            bw, bh = SIZES[ft["cls"]]
            ft["x"] += rng.gauss(0, 8)
            ft["y"] += rng.gauss(0, 8)
            dets.append((ft["id"], ft["cls"], ft["conf"], (ft["x"] - bw / 2, ft["y"] - bh, ft["x"] + bw / 2, ft["y"])))
            ft["left"] -= 1
        false_tracks = [ft for ft in false_tracks if ft["left"] > 0]
        # same confidence gate the notebook applies before the engine
        dets = [d for d in dets if d[1] == "backpack" or d[2] >= cfg["person_vehicle_conf"]]
        yield float(t), dets




def run_engine(ns, actors, noise, rng, cfg, video_id, snap_dir, record=None, aliases=None):
    zones, lines = ns["scene_to_px"](ns["DEFAULT_SCENE"], W, H)
    engine = ns["EventEngine"](video_id, snap_dir, cfg["cooldown_s"])
    book = ns["TrackBook"](maxlen=int(cfg["target_fps"] * max(cfg["stationary_window_s"], 5)) + 5)
    frames = 0
    for t, dets in perceive(actors, noise, rng, cfg):
        ns["analyse_frame"](t, dets, zones, lines, book, engine, None, cfg, W)
        if record is not None:
            record.append((t, dets))
        frames += 1
    engine.finish()
    if aliases is not None:
        aliases.update(book.alias)   # raw tracker ID -> stitched identity
    return engine.events, frames


# Naive temporal logic: every rule fires on raw per-frame tracker output.
BASELINE = dict(cooldown_s=0.0, stitch_gap_s=0.0, line_debounce_s=0.0, min_track_obs=1, smooth_alpha=1.0, line_margin=0.0,
                attend_min_obs=0)
ABLATIONS = {
    "no hysteresis (cooldown 0 s)": dict(cooldown_s=0.0),
    "no ID stitching": dict(stitch_gap_s=0.0),
    "no position smoothing": dict(smooth_alpha=1.0),
    "no line dead-band": dict(line_margin=0.0),
    "no line debounce": dict(line_debounce_s=0.0),
    "no minimum track age": dict(min_track_obs=1),
    "no bag-attendance gate": dict(attend_min_obs=0),
}
_SCENES = {}


def scene(ns, cfg, seed, k):
    # Scenario and its ground truth, cached: they depend only on the seed (and fixed event thresholds).
    if (seed, k) not in _SCENES:
        zones, lines = ns["scene_to_px"](ns["DEFAULT_SCENE"], W, H)
        actors = make_scenario(random.Random(seed * 1000 + k))
        _SCENES[(seed, k)] = (actors, label_ground_truth(actors, zones, lines, cfg, f"s{seed}_{k:03d}"))
    return _SCENES[(seed, k)]


def bootstrap_ci(counts, n_boot=2000, seed=0):
    # 95% CI for precision, recall and F1 by resampling scenarios.
    c = np.array(counts, dtype=float)               # rows: (tp, fp, fn) per scenario
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    tp, fp, fn = c[idx].sum(axis=1).T
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r = tp / (tp + fp), tp / (tp + fn)
        f = 2 * p * r / (p + r)

    def q(a):
        return [round(float(np.nanpercentile(a, 2.5)), 3), round(float(np.nanpercentile(a, 97.5)), 3)]
    return dict(precision=q(p), recall=q(r), f1=q(f))


def study(ns, cfg, noise, n_scenarios, seed, snap_dir):
    events, gt, counts, frames, elapsed = [], [], [], 0, 0.0
    for k in range(n_scenarios):
        actors, g = scene(ns, cfg, seed, k)
        t0 = time.time()
        ev, n = run_engine(ns, actors, noise, random.Random(seed * 7919 + k), cfg, f"s{seed}_{k:03d}", snap_dir)
        elapsed += time.time() - t0
        o = ns["evaluate"](ev, g, tol_s=3.0)["overall"]
        counts.append((o["tp"], o["fp"], o["fn"]))
        events += ev
        gt += g
        frames += n
    m = ns["evaluate"](events, gt, tol_s=3.0)
    m["ci95"] = bootstrap_ci(counts)
    m["n_gt_events"], m["n_pred_events"] = len(gt), len(events)
    m["engine_fps"] = round(frames / elapsed, 1)
    return m


def show(tag, m):
    o, ci = m["overall"], m["ci95"]["f1"]
    print(f"{tag:34s} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
          f"  err={o['mean_abs_start_err_s']}s  n_gt={m['n_gt_events']} n_pred={m['n_pred_events']}", flush=True)


def export_replay(ns, cfg, seed, n_candidates, snap_dir):
    # One rich test scene, replayed at 5 fps for the showcase page.
    def richness(k):
        g = scene(ns, cfg, seed, k)[1]
        return len({x["type"] for x in g}), len(g)
    best = max(range(n_candidates), key=richness)
    actors, gt = scene(ns, cfg, seed, best)
    zones, lines = ns["scene_to_px"](ns["DEFAULT_SCENE"], W, H)
    classes = ["person", "car", "backpack"]
    out = dict(scene=f"test scene {best}", frame=[W, H], fps=5, duration_s=T_END, classes=classes,
               zones=[dict(name=z["name"], type=z["type"],
                           polygon=[[round(x), round(y)] for x, y in z["poly"].exterior.coords][:-1]) for z in zones],
               lines=[dict(name=l["name"], p1=[round(v) for v in l["p1"]], p2=[round(v) for v in l["p2"]],
                           wrong_way=l.get("wrong_way")) for l in lines],
               ground_truth=[dict(type=g["type"], zone=g["zone"], start=round(g["start_s"], 1), end=round(g["end_s"], 1))
                             for g in gt],
               variants={})
    for level in ("clean", "medium", "high"):
        # Both systems see the identical noisy tracks (same RNG seed), so only the event logic differs.
        systems, frames = {}, None
        for name, c in (("baseline", dict(cfg, **BASELINE)), ("sentryscribe", cfg)):
            rec, alias = [], {}
            ev, _ = run_engine(ns, actors, NOISE_LEVELS[level], random.Random(seed * 7919 + best), c,
                               f"replay_{level}", snap_dir, record=rec, aliases=alias)
            m = ns["evaluate"](ev, [dict(g, video=f"replay_{level}.mp4") for g in gt], tol_s=3.0)["overall"]
            if frames is None:
                frames = [[[d[0], classes.index(d[1]), *[round(v) for v in d[3]]] for d in dets]
                          for i, (t, dets) in enumerate(rec) if i % 2 == 1]
            systems[name] = dict(
                score=dict(tp=m["tp"], fp=m["fp"], fn=m["fn"], f1=m["f1"]), alias=alias,
                events=[dict(id=e["event_id"], type=e["type"], zone=e["zone"], object=e["object_class"], tracks=e["track_ids"],
                             start=e["start_ts"], end=e["end_ts"],
                             details={k: v for k, v in e["details"].items() if k in ("direction", "wrong_way", "max_count")})
                        for e in ev])
        out["variants"][level] = dict(frames=frames, systems=systems)
    (RESULTS / "replay.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"replay: scene {best}, {len(gt)} GT events, types {sorted({g['type'] for g in gt})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-scenarios", type=int, default=40)
    ap.add_argument("--test-scenarios", type=int, default=100)
    ap.add_argument("--dev-seed", type=int, default=7)
    ap.add_argument("--test-seed", type=int, default=2026)
    ap.add_argument("--dev-only", action="store_true", help="tuning run: skip the held-out test set")
    args = ap.parse_args()

    ns = load_notebook_code()
    full = dict(ns["CONFIG"])
    base = dict(full, **BASELINE)
    snap_dir = Path(tempfile.mkdtemp())
    out = dict(setup=dict(dev_seed=args.dev_seed, dev_scenarios=args.dev_scenarios, test_seed=args.test_seed,
                          test_scenarios=args.test_scenarios, scenario_length_s=T_END, fps=FPS, frame=[W, H],
                          tolerance_s=3.0, noise_levels=NOISE_LEVELS, config=full, baseline_changes=BASELINE),
               dev={}, test=dict(by_noise={}, ablations={}))

    print("== dev set (used for tuning) ==")
    for level, noise in NOISE_LEVELS.items():
        m = study(ns, full, noise, args.dev_scenarios, args.dev_seed, snap_dir)
        out["dev"][level] = dict(overall=m["overall"], ci95=m["ci95"], per_type=m["per_type"])
        show(f"dev   {level}", m)
    if args.dev_only:
        for level in ("medium", "high"):
            print(level, {t: (v["tp"], v["fp"], v["fn"]) for t, v in out["dev"][level]["per_type"].items()})
        return

    print("== held-out test set ==")
    for level, noise in NOISE_LEVELS.items():
        row = {}
        for name, cfg in (("baseline", base), ("sentryscribe", full)):
            row[name] = study(ns, cfg, noise, args.test_scenarios, args.test_seed, snap_dir)
            show(f"test  {level:6s} {name}", row[name])
        out["test"]["by_noise"][level] = row
    for label, change in ABLATIONS.items():
        m = study(ns, dict(full, **change), NOISE_LEVELS["medium"], args.test_scenarios, args.test_seed, snap_dir)
        out["test"]["ablations"][label] = dict(change=change, overall=m["overall"], ci95=m["ci95"])
        show(f"ablate {label}", m)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "simulation_metrics.json").write_text(json.dumps(out, indent=2))
    export_replay(ns, full, args.test_seed, 30, snap_dir)
    write_summary(out)
    plot(out)
    print("wrote", RESULTS)


def fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_summary(out):
    st, tb = out["setup"], out["test"]["by_noise"]
    types = sorted({t for row in tb.values() for t in row["sentryscribe"]["per_type"]})
    L = ["# Simulation results", "",
         f"Held-out test set: {st['test_scenarios']} scenes × {int(T_END)} s (seed {st['test_seed']}); "
         f"parameters were tuned only on the dev set ({st['dev_scenarios']} scenes, seed {st['dev_seed']}). "
         f"Match tolerance ±{st['tolerance_s']} s; 95% CIs from 2000 scenario bootstraps.", "",
         "## Baseline vs SentryScribe (test set)", "",
         "| Noise | System | Precision | Recall | F1 [95% CI] | Start error (s) |", "|---|---|---|---|---|---|"]
    for level, row in tb.items():
        for name in ("baseline", "sentryscribe"):
            o, ci = row[name]["overall"], row[name]["ci95"]["f1"]
            L.append(f"| {level} | {name} | {fmt(o['precision'])} | {fmt(o['recall'])} | "
                     f"{fmt(o['f1'])} [{ci[0]:.3f}, {ci[1]:.3f}] | {fmt(o['mean_abs_start_err_s'], 2)} |")
    L += ["", "## SentryScribe F1 per event type (test set)", "", "| Event type | " + " | ".join(tb) + " |",
          "|---|" + "---|" * len(tb)]
    for t in types:
        L.append(f"| {t} | " + " | ".join(fmt(row["sentryscribe"]["per_type"].get(t, {}).get("f1")) for row in tb.values()) + " |")
    L += ["", "## Ablations (test set, medium noise)", "", "| Removed | Precision | Recall | F1 [95% CI] |", "|---|---|---|---|"]
    o, ci = tb["medium"]["sentryscribe"]["overall"], tb["medium"]["sentryscribe"]["ci95"]["f1"]
    L.append(f"| nothing (full system) | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    for label, a in out["test"]["ablations"].items():
        o, ci = a["overall"], a["ci95"]["f1"]
        L.append(f"| {label} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    (RESULTS / "simulation_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb = out["test"]["by_noise"]
    levels = list(tb)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=170)
    x = np.arange(len(levels))
    for j, (name, color) in enumerate((("baseline", "#b7bec8"), ("sentryscribe", "#0f6e6e"))):
        f1 = [tb[l][name]["overall"]["f1"] for l in levels]
        lo = [f - tb[l][name]["ci95"]["f1"][0] for f, l in zip(f1, levels)]
        hi = [tb[l][name]["ci95"]["f1"][1] - f for f, l in zip(f1, levels)]
        bars = ax.bar(x + (j - 0.5) * 0.38, f1, 0.36, color=color, label="naive baseline" if j == 0 else "SentryScribe",
                      yerr=[lo, hi], capsize=3, error_kw=dict(ecolor="#3b4654", lw=0.9))
        for b, v in zip(bars, f1):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=8, color="#1b2430")
    ax.set_xticks(x, [f"{l} noise" for l in levels])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("event F1 (95% CI)")
    ax.set_title("Held-out test set: naive per-frame rules vs SentryScribe", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "sim_baseline_vs_full.png")
    plt.close(fig)

    types = sorted({t for row in tb.values() for t in row["sentryscribe"]["per_type"]})
    colors = ["#cfe3e3", "#7fb8b8", "#2f8f8f", "#0f5454"]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    bw = 0.8 / len(levels)
    for j, lvl in enumerate(levels):
        vals = [tb[lvl]["sentryscribe"]["per_type"].get(t, {}).get("f1") or 0 for t in types]
        ax.bar(np.arange(len(types)) + (j - (len(levels) - 1) / 2) * bw, vals, bw * 0.92, label=lvl, color=colors[j])
    ax.set_xticks(np.arange(len(types)), [t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("SentryScribe F1 per event type, held-out test set", fontsize=11, loc="left")
    ax.legend(title="noise", fontsize=8, title_fontsize=8, frameon=False, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "sim_f1_by_type.png")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
