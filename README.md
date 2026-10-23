# SentryScribe 🎥
### Turning CCTV Footage into Searchable Events and Intelligent Incident Reports

SentryScribe is a final-year B.Tech project that transforms CCTV footage into **timestamped, searchable incident logs and AI-generated reports** using computer vision, multi-object tracking, rule-based event detection, and a locally running Large Language Model (LLM).

Instead of manually reviewing hours of surveillance footage, security operators can identify important events, generate incident summaries, and ask natural-language questions about what happened and when.

**Core technologies:** Python · YOLOv8 · ByteTrack · OpenCV · Qwen2.5 · Hugging Face Transformers

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Incident Detection](#-incident-detection)
- [AI-Powered Reporting and Q&A](#-ai-powered-reporting-and-qa)
- [Technology Stack](#-technology-stack)
- [Evaluation and Results](#-evaluation-and-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Example Use Cases](#-example-use-cases)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Privacy and Responsible Use](#-privacy-and-responsible-use)

## 🚀 Overview

CCTV cameras continuously generate large volumes of video, but finding a specific incident often requires manually scrubbing through hours of footage. Important events may last only a few seconds, making manual monitoring inefficient and error-prone.

SentryScribe addresses this problem by combining pretrained computer vision models with a rule-based event engine and a grounded LLM.

The system processes uploaded video files, detects and tracks objects, identifies meaningful incidents, and stores structured event data. An LLM then uses this event data to generate readable incident reports and answer questions in natural language.

### The pipeline

**Video → Object Detection → Multi-Object Tracking → Event Detection → Structured Event Log → AI Reports and Question Answering**

The project is designed to run through a Google Colab notebook, with an interactive showcase page for exploring results.

## ✨ Key Features

- **Object detection:** Detects people, vehicles, bicycles, and bags using pretrained YOLOv8 models.
- **Multi-object tracking:** Uses ByteTrack to maintain object identities across video frames.
- **Configurable surveillance zones:** Supports camera-specific polygon zones and virtual counting lines.
- **Rule-based incident detection:** Identifies six categories of security-relevant events using spatial and temporal conditions.
- **Tracker-noise handling:** Applies hysteresis, ID stitching, position smoothing, and additional safeguards to reduce false alarms.
- **Automated incident reports:** Generates structured Markdown reports with event IDs and timestamps.
- **Natural-language question answering:** Answers questions by searching structured event data through predefined tools.
- **Citation validation:** Checks event references and timestamps before accepting generated reports.
- **Annotated video and snapshots:** Produces visual outputs showing detections, track IDs, zones, and incident snapshots.
- **Simulation-based evaluation:** Compares the event engine against a naive baseline using controlled tracker noise and held-out test scenes.

## 🏗️ System Architecture

SentryScribe separates video processing from language-based reasoning. The computer vision pipeline runs once per video, while report generation and question answering operate on the resulting event data.

### 1. Vision pipeline

The input video is sampled into frames. YOLOv8 detects relevant objects, and ByteTrack associates detections across frames to maintain track IDs.

### 2. Event engine

The system evaluates object trajectories against configured zones, virtual lines, and time thresholds. Noise-handling mechanisms reduce duplicate events and missed incidents caused by inconsistent tracks.

### 3. Event storage

Events, track summaries, and video-level statistics are stored in structured JSON files. Annotated videos and event snapshots are saved alongside them.

### 4. LLM layer

Qwen2.5 generates incident reports and answers natural-language questions using the event log. A set of typed search tools provides access to the stored information.

### 5. Application layer

The Colab notebook provides the main workflow for video processing, event inspection, report generation, and question answering. A static HTML showcase allows users to explore evaluation results.

## 🚨 Incident Detection

SentryScribe implements six event types using explainable, configurable rules rather than training a separate model for every incident.

| Event type | Detection logic |
|---|---|
| Zone intrusion | A person's or vehicle's foot point remains inside a restricted zone for at least 1 second. |
| Loitering | A person remains in a zone for approximately 60 seconds or longer while moving within the configured distance threshold. |
| Line crossing | A tracked object crosses a virtual counting line; movement in a prohibited direction can be flagged. |
| Abandoned bag | A bag remains stationary for 30 seconds without a sufficiently established person track nearby. |
| Crowding | At least 10 people remain inside a zone for 5 seconds. |
| Vehicle in pedestrian zone | A car, truck, bus, or motorcycle remains inside a pedestrian-only zone for 2 seconds. |

Thresholds can be adjusted through the camera and scene configuration.

### Handling tracking noise

Real-world tracking is imperfect. Bounding boxes can jitter, detections can disappear, and object IDs can change. SentryScribe addresses these problems through six mechanisms:

1. **Hysteresis:** Prevents short detection gaps from splitting one incident into multiple events.
2. **ID stitching:** Connects compatible tracks across short gaps when their positions and classes match.
3. **Position smoothing:** Uses an exponential moving average to reduce coordinate jitter.
4. **Line dead-band:** Requires an object to move sufficiently beyond a counting line before a crossing is registered.
5. **Bag-attendance gate:** Prevents short-lived phantom tracks from incorrectly resetting an abandoned-bag timer.
6. **Backdated starts:** Records an abandoned-bag incident from when the bag was left rather than when the detection threshold is finally reached.

Together, these mechanisms make the event engine more robust to imperfect tracking data.

## 🤖 AI-Powered Reporting and Q&A

SentryScribe uses Qwen2.5 through Hugging Face Transformers to generate incident reports and answer questions about processed footage.

A key design principle is that **the LLM reasons over structured event data, not raw video**.

### Automated incident reports

The report writer generates a structured report containing:

- Executive summary
- Incident timeline
- Notable incidents
- Event statistics
- Recommended follow-up actions

Claims in the report are expected to cite the event IDs that support them. A validator checks whether cited IDs exist and whether cited timestamps fall within the corresponding events. If validation fails, the system attempts one regeneration.

Reports and their validation results are saved as Markdown and JSON files.

### Natural-language question answering

The query agent uses four predefined tools to retrieve information from the event log:

| Tool | Purpose |
|---|---|
| `search_events()` | Filters incidents by event type, object class, zone, and time range. |
| `count_objects()` | Counts unique tracked objects, optionally within a zone or time range. |
| `get_event()` | Retrieves the details of a particular event. |
| `video_summary()` | Returns video duration, object totals, and incident statistics. |

Rather than allowing the LLM to generate arbitrary SQL queries, the system restricts data access to these typed functions.

**Example questions:**

- When did a truck enter Gate B?
- How many people entered the restricted zone?
- Were there any abandoned bags?
- Which incidents occurred near the pedestrian walkway?
- Summarize the significant events in this video.

The answers are grounded in retrieved event records, with event references used to make the results easier to verify.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.10+ |
| Object detection | Ultralytics YOLOv8 |
| Multi-object tracking | ByteTrack |
| Video processing | OpenCV |
| Geometric operations | Shapely |
| Language model | Qwen2.5-Instruct |
| LLM inference | Hugging Face Transformers |
| Data storage | JSON files |
| Interactive environment | Google Colab |
| Showcase | HTML, CSS, JavaScript |
| Evaluation and analysis | NumPy, pandas, matplotlib |
| Testing | Python test scripts |

The project uses pretrained models and does not train a new detector or language model. Model size and inference configuration can be selected according to available hardware.

## 📊 Evaluation and Results

The event engine was evaluated using simulated surveillance scenes with known ground truth and controlled levels of tracking noise.

The study used 40 development scenes for tuning and 100 held-out test scenes for final evaluation. A naive baseline applied the same event rules without the noise-handling mechanisms.

An event was considered a match when its type and zone agreed with the ground truth and its start time was within ±3 seconds.

### Event detection performance

| Noise level | Naive baseline F1 | SentryScribe F1 |
|---|---:|---:|
| Clean | 1.000 | 1.000 |
| Low | 0.411 | 0.999 |
| Medium | 0.325 | 0.997 |
| High | 0.259 | 0.967 |

<sub>Results reported in the project's held-out simulation evaluation.</sub>

### Key findings

- **0.967 F1 at high tracker noise**, with a reported 95% bootstrap confidence interval of 0.952–0.980.
- **0.997 F1 at medium noise**, compared with 0.325 for the naive baseline.
- **0.33 seconds** of mean event-start timing error at high noise.
- **Hysteresis was the most influential mechanism:** removing it reduced medium-noise F1 from 0.997 to approximately 0.390.
- **ID stitching was the second-largest contributor:** removing it reduced medium-noise F1 to approximately 0.833.

The evaluation also included runs on nine static-camera sequences from the MOT15 and MOT16 benchmarks. Those real-footage experiments exposed challenges associated with dense crowds and long occlusions.

**Important:** The headline F1 results measure event-logic robustness under simulated tracking noise. They should not be interpreted as a claim of 96.7% real-world CCTV incident accuracy.

## 📂 Project Structure

The repository is organized as follows:

```text
SentryScribe/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── colab_demo.ipynb
├── scripts/
│   ├── simulate.py
│   └── build_showcase.py
├── tests/
│   └── test_notebook.py
├── results/
│   ├── real_footage/
│   ├── simulation_metrics.json
│   ├── simulation_summary.md
│   ├── replay.json
│   ├── sim_baseline_vs_full.png
│   └── sim_f1_by_type.png
├── showcase/
│   ├── template.html
│   └── index.html
└── docs/
    ├── project_structure.pdf
    ├── viva_prep.pdf
    └── SentryScribe_CIE3.pptx
```

The `notebooks/colab_demo.ipynb` notebook contains the main pipeline. The simulation script evaluates event detection, the test script checks core logic and integration behaviour, and the showcase builder generates an interactive page from the evaluation results.

Keep the repository structure aligned with the files actually committed to GitHub.

## ▶️ Getting Started

SentryScribe is designed around Google Colab, so a local GPU installation is not required to explore the documented workflow.

### Prerequisites

- A Google account with access to Google Colab
- The project repository
- A supported input video in MP4, AVI, MOV, or MKV format
- Access to the required Python dependencies and model weights

For full-size LLM inference, a GPU with sufficient memory is recommended. The notebook can select smaller Qwen2.5 model variants for more limited hardware.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/SentryScribe.git
cd SentryScribe
```

Replace `<your-username>` with your GitHub username.

### 2. Open the notebook

Open `notebooks/colab_demo.ipynb` in Google Colab. Alternatively, upload the notebook to Colab directly.

### 3. Install dependencies

Use the notebook's dependency-installation cells and the provided `requirements.txt` to set up the environment.

For a local Python environment, you can create a virtual environment first:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the project's dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure the scene

Review the notebook configuration and the scene settings. Define surveillance zones as polygons and counting lines for the camera being analyzed. Adjust event thresholds when necessary.

### 5. Process a video

Upload a supported video or select a video from the configured input folder. Run the notebook cells in order to detect and track objects, evaluate event rules, and save the results.

### 6. Inspect the outputs

Review the generated annotated video, event table, snapshots, JSON event records, and incident report. Use the question-answering interface to query the processed event log.

### 7. Run the simulation evaluation

From the repository root, run:

```bash
python scripts/simulate.py
```

Generate the interactive results page with:

```bash
python scripts/build_showcase.py
```

Open `showcase/index.html` in a browser to explore the generated showcase. It is designed to work offline.

**Reproducibility note:** Exact runtime and memory requirements depend on the selected detector and LLM variants, input video, and available hardware. Follow the notebook's configuration and dependency instructions for the tested environment.

## 💡 Example Use Cases

- **Campus security:** Identify entries into restricted areas and activity near walkways.
- **Facility monitoring:** Detect prolonged loitering, line crossings, and unauthorized vehicle movement.
- **Public spaces:** Identify periods of high crowd density.
- **Parking and access monitoring:** Review vehicle movements through configured zones.
- **Incident investigation:** Search previously processed footage by event type, location, and timestamp.
- **Security reporting:** Generate structured, traceable incident summaries without manually reviewing every frame.

These are potential applications of the implemented capabilities; effectiveness in a particular deployment requires appropriate testing and configuration.

## ⚠️ Limitations

SentryScribe is an offline video-analysis prototype, not a production-ready surveillance platform.

- Simulation results do not fully reproduce real-world lighting, perspective, persistent false detections, or dense-crowd occlusion.
- Real-footage evaluation covers person-based events; abandoned-bag and vehicle-event performance is evaluated in simulation.
- Long occlusions can cause track IDs to change, which can affect loitering and other duration-based rules.
- Small or distant objects may be missed, depending on detector size, resolution, and hardware.
- LLM output can still be incomplete or incorrect. Citation validation improves traceability but does not guarantee semantic correctness.
- Automated question-answering quality requires careful evaluation beyond checking citations.
- Processing is designed as an offline job; live camera streams and real-time alert delivery are outside the current scope.
- Results depend on correctly configured zones, thresholds, video timestamps, and model settings.

## 🔮 Future Enhancements

Potential extensions include:

- Real-time RTSP camera processing
- Live intrusion and abandoned-object alerts via email or messaging services
- A Streamlit dashboard for monitoring multiple processed videos
- A database-backed event store for larger collections of recordings
- Appearance-based re-identification for improved tracking through occlusion
- Fine-tuning or adapting object detection for CCTV viewpoints and low-light conditions
- Multi-camera tracking and cross-camera event queries
- Vision-language captions for event snapshots
- REST API endpoints for video uploads, incident reports, and question answering

These are proposed extensions rather than claims about the current implementation.

## 🔒 Privacy and Responsible Use

SentryScribe is designed to reason about anonymous object track IDs, not to identify individuals. It does not implement face recognition.

The documented architecture uses a locally running LLM and sends structured event metadata to it rather than raw video. In the intended local inference workflow, this avoids sending footage to an external LLM service.

When using CCTV footage, obtain the necessary permissions, respect applicable privacy laws, and protect recordings and generated incident data. Review AI-generated reports before using them to make consequential security decisions.

## 👥 Project Information

**Project:** SentryScribe  
**Category:** Final-Year B.Tech Project  
**Domains:** Computer Vision · Multi-Object Tracking · Rule-Based Event Detection · Large Language Models  
**Status:** Prototype implementation and evaluation completed as documented.

### Acknowledgements

SentryScribe builds on pretrained computer vision models, open-weight language models, and open-source Python libraries. Refer to the respective projects and model cards for licensing, attribution, and usage requirements.

---

If you find this project useful, consider starring the repository ⭐

*Turning hours of CCTV footage into searchable, verifiable incident intelligence.*
