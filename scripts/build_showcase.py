"""Build showcase/index.html from showcase/template.html and the simulation results.

    python scripts/simulate.py        # writes results/simulation_metrics.json and results/replay.json
    python scripts/build_showcase.py  # writes showcase/index.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SHOWCASE = ROOT / "showcase"

# Dev-set scores recorded while the engine was being fixed (40 scenes, seed 7). The test set was not used.
HISTORY = [
    dict(v="v1 · first engine", title="Rules work on clean tracks, fall apart on noisy ones",
         text="Perfect on clean tracks, but at high noise the engine raised 396 events for 222 true ones. "
              "Each ID switch restarted the loitering timer and opened a duplicate intrusion. "
              "Phantom one-frame tracks crossed the line. Abandoned-bag alerts started about 8 s late.",
         delta="dev F1 · low 0.965 · medium 0.869 · high 0.667"),
    dict(v="v2 · track hygiene", title="Stitch broken IDs, ignore newborn tracks, backdate starts",
         text="A new ID that appears near a recently lost track of the same class inherits its identity "
              "(only if that track lived at least 10 frames, so phantoms cannot chain). "
              "Line crossings need a track at least 3 observations old. Events carry a 'since' time, so a bag's alert starts when it was put down.",
         delta="dev F1 · medium 0.986 · high 0.937"),
    dict(v="v3 · geometry under jitter", title="Smooth positions, add a dead-band, check who is really nearby",
         text="Foot points are smoothed with an exponential moving average. A crossing counts only once the track is clearly on the other side of the line. "
              "A bag counts as attended only by a person track at least 10 frames old, so phantom people near a bag cannot reset its timer.",
         delta="dev F1 · medium 1.000 · high 0.980 · config frozen here"),
]


def main():
    metrics = json.loads((RESULTS / "simulation_metrics.json").read_text())
    replay = json.loads((RESULTS / "replay.json").read_text())
    data = dict(metrics=metrics, replay=replay, history=HISTORY)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (SHOWCASE / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    (SHOWCASE / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {SHOWCASE / 'index.html'} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
