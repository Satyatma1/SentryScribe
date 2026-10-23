"""Local checks for colab_demo.ipynb: syntax of every cell, rule engine on synthetic tracks,
evaluation maths, LLM validator/agent loop with a fake client, and (optionally) a real
YOLO+ByteTrack run on a synthetic video."""
import json
import sys
from pathlib import Path

NB = Path(sys.argv[1])
WORK = Path(sys.argv[2])
E2E = "--e2e" in sys.argv
WORK.mkdir(parents=True, exist_ok=True)

cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
srcs = ["".join(c["source"]) for c in cells]

# 1. every cell compiles once shell (!) lines are removed
for i, s in enumerate(srcs):
    py = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("!"))
    compile(py, f"cell{i}", "exec")
print(f"[ok] {len(srcs)} code cells compile")

tagged = lambda tag: next(s for s in srcs if s.startswith(f"# [{tag}]"))
config_src = next(s for s in srcs if "\nCONFIG = dict(" in s)
try:
    import torch
except ImportError:                 # the config cell only asks torch how big the GPU is
    import types
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
G: dict = {"torch": torch}
exec(config_src, G)
G["OUT_DIR"] = str(WORK / "outputs")
exec(tagged("engine"), G)
exec(tagged("eval"), G)

# 2. synthetic scenario on a 1000x1000 frame, 10 fps
cfg = dict(G["CONFIG"], loiter_s=20.0, crowd_n=3, crowd_s=2.0, abandon_s=10.0)
zones, lines = G["scene_to_px"](G["DEFAULT_SCENE"], 1000, 1000)
eng = G["EventEngine"]("synthetic", WORK / "snaps", cfg["cooldown_s"])
book = G["TrackBook"](maxlen=200)
box = lambda cx, foot_y, w=40, h=120: (cx - w / 2, foot_y - h, cx + w / 2, foot_y)
for k in range(0, 600):            # 60 s
    t = k / 10
    dets = []
    # track 1: walks down across the entry line (y=500) at t~5s, into Restricted (x 600-950, y 450-950) and stays until t=30
    if t < 30:
        y = min(300 + 40 * t, 800)
        x = 700 if t > 2 else 300 + 200 * t
        dets.append((1, "person", 0.9, box(x, y)))
    # track 2: walks up across the line at t~10 (wrong way)
    if 5 <= t < 15:
        dets.append((2, "person", 0.8, box(100, 700 - 40 * (t - 5))))
    # track 3: a backpack placed at (300, 900) in Walkway, alone from t=20
    if t >= 15:
        dets.append((3, "backpack", 0.5, box(300, 900, 30, 30)))
    if 15 <= t < 20:                # owner standing next to it
        dets.append((4, "person", 0.9, box(320, 905)))
    # tracks 5-7: three people standing in Walkway 40-50 s -> crowding
    if 40 <= t < 50:
        for j, tid in enumerate((5, 6, 7)):
            dets.append((tid, "person", 0.85, box(150 + 60 * j, 700)))
    # track 8: a car in Walkway 52-58 s
    if 52 <= t < 58:
        dets.append((8, "car", 0.9, box(250, 800, 160, 100)))
    G["analyse_frame"](t, dets, zones, lines, book, eng, None, cfg, 1000)
eng.finish()
evs = eng.events
by = {}
for e in evs:
    by.setdefault(e["type"], []).append(e)
for e in evs:
    print("   ", e["event_id"], e["type"], e["zone"], e["track_ids"], e["start_ts"], "->", e["end_ts"], e["details"])

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("[ok]", msg)

check(len(by.get("ZONE_INTRUSION", [])) == 1 and by["ZONE_INTRUSION"][0]["track_ids"] == [1], "one intrusion by track 1")
lc = by.get("LINE_CROSSING", [])
check(sorted((e["track_ids"][0], e["details"]["direction"], e["details"]["wrong_way"]) for e in lc)
      == [(1, "down", False), (2, "up", True)], "two line crossings, track 2 flagged wrong-way")
ab = by.get("ABANDONED_OBJECT", [])
check(len(ab) == 1 and abs(ab[0]["start_ts"] - 20.0) < 0.3 and ab[0]["end_ts"] > 59, "bag abandoned from t=20 until end")
cr = by.get("CROWDING", [])
check(len(cr) == 1 and cr[0]["details"]["max_count"] == 3 and abs(cr[0]["start_ts"] - 40) < 0.2, "crowding in Walkway at t=40")
check(len(by.get("VEHICLE_IN_PED_ZONE", [])) == 1, "car in walkway flagged once")
lo = by.get("LOITERING", [])
check(any(e["track_ids"] == [1] and e["zone"] == "Restricted" for e in lo), "track 1 loitered in Restricted (20 s threshold)")
check(not any(e["track_ids"] == [2] for e in lo), "track 2 (passing through) did not loiter")
# person 4 standing 5 s next to the bag must not loiter at 20 s threshold
check(not any(e["track_ids"] == [4] for e in lo), "short stay does not loiter")

# 3. evaluation maths
gt = [dict(video="synthetic.mp4", type="ZONE_INTRUSION", zone="Restricted", start_s=by["ZONE_INTRUSION"][0]["start_ts"] + 1, end_s=30),
      dict(video="synthetic.mp4", type="ABANDONED_OBJECT", zone=float("nan"), start_s=21, end_s=60),
      dict(video="synthetic.mp4", type="CROWDING", zone="Walkway", start_s=48, end_s=50)]   # 8 s off -> miss
m = G["evaluate"](evs, gt, tol_s=3.0)
check(m["per_type"]["ZONE_INTRUSION"]["tp"] == 1 and m["per_type"]["ABANDONED_OBJECT"]["tp"] == 1, "eval matches within tolerance")
check(m["per_type"]["CROWDING"]["tp"] == 0 and m["per_type"]["CROWDING"]["fn"] == 1 and m["per_type"]["CROWDING"]["fp"] == 1,
      "eval rejects start error > tolerance")
check(m["overall"]["tp"] == 2 and m["overall"]["fn"] == 1 and m["overall"]["fp"] == len(evs) - 2, "overall counts add up")
print("    overall:", m["overall"])
check(G["evaluate"](evs, [], videos=["synthetic"])["overall"]["fp"] == len(evs), "an annotated clip with no events still counts false alarms")

# 4. LLM layer with a fake client (no API calls)
L = dict(G)
L["ALL_EVENTS"] = evs
L["ALL_TRACKS"] = G["track_summaries"](book, "synthetic")
L["SUMMARIES"] = [dict(video_id="synthetic", file="synthetic.mp4", duration_s=60.0,
                       unique_tracks_by_class={"person": 6}, events_by_type={})]
calls = []


def fake_generate(chat, tools=None, max_tokens=1200):
    # Stands in for the Hugging Face model: same chat in, raw text (with <tool_call> blocks) out.
    calls.append(dict(chat=chat, tools=tools))
    if tools:
        if len(chat) == 2:          # system + question: ask for two tools at once
            return ('Searching.\n<tool_call>\n{"name": "search_events", "arguments": {"type": "ZONE_INTRUSION"}}\n</tool_call>\n'
                    '<tool_call>\n{"name": "count_objects", "arguments": {"object_class": "person", "zone": "walkway"}}\n</tool_call>')
        return "Yes, at 00:00:09 [E1]."
    if len(chat) == 2:              # first report attempt cites a bogus id
        return "Intrusion [E99] at 00:00:07."
    return f"Intrusion [{by['ZONE_INTRUSION'][0]['event_id']}] at 00:00:07."


exec(tagged("llm"), L)
L["generate"] = fake_generate
text, chk = L["generate_report"](L["SUMMARIES"][0], evs)
check(chk["valid"] and chk.get("retried") and chk["unknown_ids"] == [], "report validator rejects [E99] and retry passes")
bad = L["validate_report"]("At 00:05:00 something [E1][E42].", evs)
check(bad["unknown_ids"] == ["E42"] and bad["times_outside_events"] == ["00:05:00"] and bad["citation_validity"] == 0.5,
      "validator flags unknown id and out-of-window time")
trace = []
ans = L["ask"]("Did anyone enter the restricted zone?", trace=trace)
last = calls[-1]["chat"]
tool_msg = [m for m in last if m["role"] == "tool"]
check(ans.startswith("Yes") and len(last[2]["tool_calls"]) == 2 and [m["name"] for m in tool_msg] == ["search_events", "count_objects"],
      "agent loop parses both tool calls and returns both results")
res = json.loads(tool_msg[0]["content"])
check(res["total"] == 1, "search_events tool filters by type")
check(json.loads(tool_msg[1]["content"])["unique_tracks"] >= 3, "count_objects counts people in walkway (case-insensitive zone)")
check([t["tool"] for t in trace] == ["search_events", "count_objects"] and trace[0]["total"] == 1, "ask records its tool calls in trace")
loose = L["search_events"](type="intrusion", zone="restricted zone", video_id="synthetic.mp4", t_from="00:00:00", t_to="00:01:00")
check(loose["total"] == 1, "search_events tolerates loose type, zone, video id and HH:MM:SS times")
veh = L["search_events"](object_class="vehicle")["events"]
check(veh and all(e["object"] == "car" for e in veh), "'vehicle' matches car/truck/bus")
check(L["off_script"]("ไม่มีการทิ้งกระเป๋า (unattended bag)") > 0.5 and L["off_script"]("No bag was left [E3].") == 0,
      "language check flags a Thai answer, not an English one")
check("Restricted" in L["query_context"]() and "synthetic" in L["query_context"](), "query prompt lists real zone and video names")

if not E2E:
    sys.exit(0)

# 5. real YOLO + ByteTrack run on a synthetic video: bus.jpg panned right to left
import cv2, numpy as np
from ultralytics.utils import ASSETS
img = cv2.imread(str(ASSETS / "bus.jpg"))          # 810x1080, 4 people + bus
H, W = img.shape[:2]
canvas_w = W + 600
vid_path = WORK / "pan_test.mp4"
vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in range(30 * 8):                             # 8 s
    big = np.full((H, canvas_w, 3), 90, np.uint8)
    big[:, 300:300 + W] = img
    off = int(600 * f / (30 * 8))
    vw.write(np.ascontiguousarray(big[:, off:off + W]))
vw.release()
G["SCENES"] = {"pan_test.mp4": {"zones": [{"name": "Left half", "type": "restricted",
                                           "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}],
                                "lines": [{"name": "Mid line", "p1": [0.5, 0], "p2": [0.5, 1], "wrong_way": "left"}]}}
cfg = dict(G["CONFIG"], model="yolov8n.pt", target_fps=5)
summary, events, tracks = G["process_video"](vid_path, cfg, WORK / "outputs", first_id=1)
print(json.dumps({k: summary[k] for k in ("frames_processed", "stride", "processing_fps", "unique_tracks_by_class",
                                           "events_by_type", "annotated_video", "device")}, indent=1))
check(summary["frames_processed"] == 240 // summary["stride"], "one result per stride group")
check(summary["unique_tracks_by_class"].get("person", 0) >= 3, "YOLO+ByteTrack tracked the people")
check(any(e["type"] == "LINE_CROSSING" and e["details"]["direction"] == "left" for e in events), "panning people cross the mid line leftwards")
check(all(Path(e["snapshot"]).exists() for e in events), "every event has a snapshot")
check(Path(summary["annotated_video"]).exists(), "annotated video written")
