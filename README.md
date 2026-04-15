# 🏏 Cricket Intelligence Engine

**Ball-Level Video Understanding for Cricket — MVP**

Extract structured, ball-by-ball cricket intelligence from any match video using **Gemini Vision AI**. Feed a video → get line, length, shot type, bowler type, footwork, outcome, and confidence scores — automatically.

---

## 🏗️ Architecture

```
Video Input (YouTube URL or local .mp4)
   ↓
Video Ingestion  ─────────────────────────────────  src/ingestion/
   ↓
┌──────────────────────────────────────────────────────────────┐
│  PRIMARY MODE  — Gemini Batch Analysis (Recommended)         │
│  Upload full video → Gemini auto-detects all deliveries      │
│  Returns one JSON record per ball  (1 API call total)        │
└──────────────────────────────────────────────────────────────┘
   ─ OR ─
┌──────────────────────────────────────────────────────────────┐
│  SEGMENTED MODE  — ffmpeg split → Gemini per clip            │
│  Video split into N clips → Gemini analyses each clip        │
│  (N API calls, but allows clip-level video playback in UI)   │
└──────────────────────────────────────────────────────────────┘
   ↓
Validation + Normalization  ──────────────────────  src/validation/
   ↓
SQLite Database Storage  ─────────────────────────  src/storage/
   ↓
┌──────────────┬─────────────┐
│ Review UI    │  REST API   │
│ (Streamlit)  │  (FastAPI)  │
└──────────────┴─────────────┘
```

---

## 🚀 Quick Start

### 1. Setup

```bash
cd cricket-intelligence
python -m venv venv
source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
pip install yt-dlp             # for YouTube downloads
brew install ffmpeg            # macOS — required for video segmentation
```

### 2. Configure API Keys

Edit `.env`:
```env
GEMINI_API_KEY=your_key_here        # https://aistudio.google.com/apikey  (free)
ROBOFLOW_API_KEY=your_key_here      # only needed if using CV detection (see below)
```

---

## 🎬 Running the Pipeline

### Step 1 — Download a YouTube video

```bash
# Download to data/raw_videos/ and register metadata
python -m src.ingestion.downloader \
  --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --match-id my-match \
  --format T20 \
  --team-a "India" \
  --team-b "Australia"

# Video saved to: data/raw_videos/my-match.mp4
```

Or download + run the full pipeline in one command (see Step 2 below).

---

### Step 2 — Run Gemini Analysis

#### Recommended: Batch Mode (let Gemini count the balls)

Send the full video to Gemini once. Gemini automatically identifies each delivery and returns one JSON record per ball — no manual segmentation needed.

```bash
python run_pipeline.py \
  --video data/raw_videos/my-match.mp4 \
  --match-id my-match \
  --batch-mode
```
```
What would need to change to do full-video Gemini analysis:
Instead of:
Video → ffmpeg splits into N clips → Gemini analyzes each clip → N JSON records
You'd do:
Video → upload whole video to Gemini → Gemini returns JSON array with N ball records
This would:
Eliminate the segmentation step entirely
Let Gemini decide how many balls there are (it counted 4, not 15 in your case)
Use the existing BATCH_EXTRACTION_PROMPT
Cost 1 Gemini API call instead of 15
The downside: you lose clip-level video playback in the UI (no individual .mp4 per ball).
```

**Real example (Sri Lanka match — 4 deliveries detected correctly):**
```bash
python run_pipeline.py \
  --video data/raw_videos/srilanka-match.mp4 \
  --match-id srilanka-match \
  --batch-mode
```

Output:
```
✓ Gemini detected 4 ball deliveries
  Ball srilanka-match_1_1: pace | off_stump | yorker | defend → wicket (conf: 0.93)
  Ball srilanka-match_1_2: pace | middle | full  | defend → wicket (conf: 0.90)
  Ball srilanka-match_1_3: pace | middle | yorker | defend → wicket (conf: 0.93)
  Ball srilanka-match_1_4: pace | middle | full  | defend → wicket (conf: 0.90)
```

#### Download from YouTube + analyse in one command

```bash
python run_pipeline.py \
  --youtube-url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --match-id my-match \
  --batch-mode \
  --team-a "India" \
  --team-b "Australia"
```

#### Segmented Mode (fixed-duration clips, Gemini per clip)

Use this if your video is long and you want individual clip files saved per delivery:

```bash
python run_pipeline.py \
  --video data/raw_videos/my-match.mp4 \
  --match-id my-match \
  --uniform \
  --segment-duration 8 \
  --max-clips 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--segment-duration` | `8.0` | Duration of each clip in seconds |
| `--max-clips` | `30` | Maximum number of clips to process |

> **Tip:** A real cricket delivery is typically 6–10 seconds in broadcast footage. Use `--segment-duration 8` for highlights, `--segment-duration 3` for tight edits.

---

### Step 3 — Review Results

**Streamlit UI (recommended):**
```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**JSON export** (auto-saved after each pipeline run):
```
data/<match-id>_extracted.json
```

**SQLite database:**
```bash
sqlite3 data/cricket_intelligence.db
SELECT ball_id, line, length, shot_type, outcome FROM balls;
```

**Quick stats:**
```bash
python -m src.storage.db
```

---

## 📊 Ball Record Schema

Each delivery produces a structured record:

```json
{
  "ball_id": "srilanka-match_1_2",
  "bowler_type": "pace",
  "line": "middle",
  "length": "yorker",
  "variation": "none",
  "shot_type": "defend",
  "footwork": "front_foot",
  "contact_quality": "miss",
  "outcome": "wicket",
  "bounce_behavior": "normal",
  "movement": "seam",
  "raw_description": "Full-length delivery on middle stump, batsman attempted a defensive push but was beaten — clean bowled.",
  "confidence": {
    "line": 0.93,
    "length": 0.90,
    "shot_type": 0.88,
    "outcome": 0.95
  }
}
```

---

## 📁 Project Structure

```
cricket-intelligence/
├── data/
│   ├── raw_videos/          # Downloaded match videos + metadata
│   ├── ball_clips/          # Per-ball clips (segmented mode only)
│   ├── roboflow_dataset/    # YOLO training data (future use)
│   └── cricsheet/           # Ground truth data (future use)
├── models/                  # Custom model weights (future use)
├── src/
│   ├── ingestion/           # YouTube download + video registration
│   ├── segmentation/        # ffmpeg-based clip extraction
│   ├── detection/           # Roboflow CV + YOLO (see below)
│   ├── tracking/            # Ball trajectory tracking
│   ├── intelligence/        # Gemini AI extraction (primary)
│   ├── validation/          # Schema validation + normalization
│   ├── storage/             # SQLite via SQLAlchemy
│   └── api/                 # REST API (FastAPI)
├── ui/                      # Streamlit review + correction app
├── scripts/
│   ├── train_yolo.py        # Custom YOLO training (future use)
│   └── validate_cric360.py  # Model validation on Cric-360 dataset
├── run_pipeline.py          # Main pipeline orchestrator
├── requirements.txt
└── .env
```

---

## 🔬 Computer Vision Layer (Future / Optional)

The codebase includes a full **Roboflow + YOLO computer vision layer** built as groundwork for a more accurate Phase 2. It is **not required** to run the current Gemini-based pipeline.

> **Note:** The Roboflow API key bundled in `.env` is on a **19-day free trial expiring 28 April 2026**. After that date a new key will be required to use CV features.

### What's built

| File | Role |
|---|---|
| `src/detection/detect.py` | `CricketDetector`, `DualModelDetector`, `LineLengthEstimator` — detects players, ball, stumps with bounding boxes; computes geometric line/length from stump pixel positions |
| `run_pipeline.py` Step 2.5 | Roboflow CV pre-analysis — samples a frame from each clip, runs both models, feeds geometry to Gemini as grounding context |
| `scripts/validate_cric360.py` | Validates the Roboflow model on the Cric-360 broadcast dataset (HuggingFace: `sarimshahzad/Cric-360`) |
| `scripts/train_yolo.py` | Full YOLO fine-tuning pipeline on the Roboflow cricket dataset — ready to train a custom model once sufficient annotated data is available |
| `data/roboflowinference.py` | Quick test script to run a single image through the Roboflow model and print raw predictions |

### Two Roboflow models

| Model ID | Classes | Purpose |
|---|---|---|
| `cricket-oftm6/3` | ball, batsman, bowler, wicketkeeper, nonstriker, umpire | Scene understanding — who is where (mAP 96.2%) |
| `stumps/10` | ball, Batsman, Stumps | Geometric reference — stumps bbox → precise line/length calculation |

### How to enable CV pre-analysis (when API key is valid)

```bash
# Default pipeline with Roboflow geometry (Roboflow ON by default in segmented mode)
python run_pipeline.py \
  --video data/raw_videos/my-match.mp4 \
  --match-id my-match \
  --uniform \
  --segment-duration 8

# Disable Roboflow explicitly (Gemini estimates line/length from video)
python run_pipeline.py \
  --video data/raw_videos/my-match.mp4 \
  --match-id my-match \
  --uniform \
  --segment-duration 8 \
  --no-cv

# Test a single image through Roboflow
python src/detection/detect.py --image data/test_img/img1.jpg

# Dual model — scene + geometric line/length overlay on video
python src/detection/detect.py \
  --video data/raw_videos/my-match.mp4 \
  --output data/my-match-annotated.mp4 \
  --dual
```

### Custom YOLO training (future)

```bash
# Download Cric-360 validation dataset from HuggingFace (requires HUGGINGFACE_TOKEN)
python scripts/validate_cric360.py

# Fine-tune YOLO on cricket data
python scripts/train_yolo.py \
  --dataset data/roboflow_dataset/data.yaml \
  --epochs 50
```

---

## 🔑 API Keys

| Key | Required | Where to get |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free |
| `ROBOFLOW_API_KEY` | Only for CV features | [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api) — trial expires 28 Apr 2026 |
| `HUGGINGFACE_TOKEN` | Only for Cric-360 download | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free |

---

## 🔗 Data Sources

| Source | Purpose |
|---|---|
| YouTube (IPL / ICC highlights) | Match video input |
| Roboflow Universe (`cricket-oftm6`, `stumps`) | Pre-trained CV models for player/ball/stump detection |
| Cric-360 (HuggingFace: `sarimshahzad/Cric-360`) | 3,558 broadcast frames for YOLO validation |
| CricSheet | Ground truth ball-by-ball data for future evaluation |

## Default Values
|From the code we already saw, the defaults are:|
|--segment-duration → 8.0 seconds per clip|
|--max-clips → 30 clips|
|So if you run:
python run_pipeline.py \
  --video data/raw_videos/match-005.mp4 \
  --match-id my_match \
  --uniform
It will split the video into 8-second chunks, up to a maximum of 30 clips (= first 240 seconds / 4 minutes of the video).
For a typical cricket highlight video with 6-ball overs, 8 seconds per clip and 30 clips is a reasonable default — covers about half an over worth of deliveries if each ball clip is well-trimmed.|
|Example with a 30-second video:
|--segment-duration|	--max-clips|	Result|
|3|	1|	1 clip × 3s = analyzes first 3s only|
|8|	3|	3 clips × 8s = analyzes first 24s|
|5| 6|	6 clips × 5s = analyzes first 30s|

