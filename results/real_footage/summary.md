# Real-footage run: MOT15 + MOT16

The outputs of one run of `notebooks/colab_demo.ipynb` in benchmark mode on a Google Colab A100, on 23 October 2026.

## What was run

- **Detector:** YOLOv8x (COCO weights) at 1280 px, tracked by ByteTrack, with frames sampled at about 10 per second.
- **LLM:** Qwen2.5-14B-Instruct, run locally.
- **Footage:** all nine static-camera training sequences of MOT15 and MOT16. That is 248 s of video in which every person in every frame is boxed by hand.

| Sequence | Scene | Size, frame rate | Length |
|---|---|---|---|
| PETS09-S2L1 | Elevated surveillance camera over a campus walkway | 768×576, 7 fps | 114 s |
| Venice-2 | Busy public square | 1920×1080, 30 fps | 20 s |
| ADL-Rundle-6 | Pedestrian street | 1920×1080, 30 fps | 17.5 s |
| KITTI-17 | Street corner from a parked car | 1224×370, 10 fps | 14.5 s |
| TUD-Stadtmitte | Pedestrian zone, eye level | 640×480, 25 fps | 7 s |
| TUD-Campus | Campus path, eye level | 640×480, 25 fps | 3 s |
| MOT16-02 | Square with many people | 1920×1080, 30 fps | 20 s |
| MOT16-04 | Elevated camera over a busy pedestrian street at night | 1920×1080, 30 fps | 35 s |
| MOT16-09 | Shopping street entrance | 1920×1080, 30 fps | 17.5 s |

- **Zones:** placed automatically from where the annotated people walk.
  - A *Restricted* zone covers the least-visited part of the walked area that at least two people enter.
  - A *Plaza* watch zone covers the whole walked area.
  - An *Entry line* runs across the middle.
- **Ground truth:** the event definitions applied to the hand-annotated tracks, with no detector or tracker involved.

## Files

| File | Contents |
|---|---|
| `benchmark_scenes.json` | The zones, line and per-scene thresholds placed for each sequence |
| `ground_truth.csv` | Ground-truth events derived from the hand-annotated tracks |
| `all_events.json` | Every event the system raised, with times, zone, tracks and confidence |
| `all_summaries.json` | Per-video statistics: duration, frames processed, objects tracked, speed |
| `metrics.json` | The evaluation the notebook wrote in step 8 |
| `reports/` | The incident report for each video, plus its validator check (`*.check.json`) |
| `snapshot_*.jpg` | Two event snapshots written by the system |

## Observations

- **Speed:** every clip ran faster than real time on the A100. 1080p clips ran 1.6–1.9× faster and smaller clips 2.6–3.6× faster. The 248 s of video took 103 s in total.
- **Reports:** all nine incident reports passed the citation validator. TUD-Campus had no events; its first draft cited a made-up `E0`, and the retry passed.
- **Hardest case:** people hidden behind each other in dense crowds. The tracker gives them a new ID when they reappear, so long stays are split.
