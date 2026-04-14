# 🏏 Cricket Intelligence Engine

**Ball-Level Video Understanding for Cricket — POC**

Convert cricket match videos into structured, queryable ball-by-ball intelligence using Computer Vision (YOLO) and AI (Gemini).

## 🏗️ Architecture

```
Video Input
   ↓
Video Ingestion (YouTube / local file)
   ↓
Ball Segmentation (per-delivery clips)
   ↓
┌─────────────────────┬─────────────────────┐
│  YOLO Ball Tracking  │  Gemini AI Analysis  │
│  (trajectory + viz)  │  (shot/line/length)  │
└─────────────────────┴─────────────────────┘
   ↓
Validation + Normalization
   ↓
SQLite Database Storage
   ↓
┌──────────────┬─────────────┐
│ Review UI    │  REST API   │
│ (Streamlit)  │  (FastAPI)  │
└──────────────┴─────────────┘
```

## 🚀 Quick Start

### 1. Setup
```bash
# Clone/navigate to project
cd cricket-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com/apikey
```

### 2. Run the Pipeline

**Option A: With a local video + timestamps**
```bash
# Generate a timestamp template (fill in manually)
python -m src.segmentation.clip_extractor --template --match-id ipl_2024_mi_csk --overs 5

# Edit data/ipl_2024_mi_csk_timestamps.json with actual timestamps

# Run full pipeline
python run_pipeline.py \
  --video data/raw_videos/highlight.mp4 \
  --match-id ipl_2024_mi_csk \
  --team-a "MI" --team-b "CSK" \
  --timestamps data/ipl_2024_mi_csk_timestamps.json
```

**Option B: Quick test with uniform splitting**
```bash
python run_pipeline.py \
  --video data/raw_videos/highlight.mp4 \
  --match-id test_001 \
  --uniform --segment-duration 8 --max-clips 10
```

### 3. Review Results
```bash
# Launch review UI
streamlit run ui/app.py

# Start REST API
python -m src.api.main
# API docs at: http://localhost:8000/docs
```

### 4. Ball Tracking (CV)
```bash
# Track ball in a clip
python -m src.tracking.tracker \
  --video data/ball_clips/test_001/clip.mp4 \
  --output data/tracked_output.mp4 \
  --pitch-map

# Train YOLO on cricket data (after downloading from Roboflow)
python -m src.detection.detect --train --dataset data/roboflow_dataset/data.yaml
```

## 📁 Project Structure
```
cricket-intelligence/
├── data/                    # All data files
│   ├── raw_videos/          # Downloaded match videos
│   ├── ball_clips/          # Per-ball video clips
│   ├── roboflow_dataset/    # YOLO training data
│   └── cricsheet/           # Ground truth data
├── models/                  # Trained model weights
├── src/
│   ├── ingestion/           # Video download
│   ├── segmentation/        # Ball clip extraction
│   ├── detection/           # YOLO ball detection
│   ├── tracking/            # Ball trajectory tracking
│   ├── intelligence/        # Gemini AI extraction
│   ├── validation/          # Output normalization
│   ├── storage/             # Database layer
│   └── api/                 # REST API
├── ui/                      # Streamlit review app
├── run_pipeline.py          # Main orchestrator
├── requirements.txt
└── .env.example
```

## 📊 Ball Record Schema
Each delivery produces a structured JSON record:
```json
{
  "ball_id": "ipl_2024_mi_csk_14_3",
  "bowler_type": "pace",
  "line": "outside_off",
  "length": "good",
  "shot_type": "drive",
  "contact_quality": "clean",
  "outcome": "4",
  "confidence": { "line": 0.85, "length": 0.78, "shot_type": 0.72 }
}
```

## 🔑 Data Sources
| Source | Purpose | Link |
|--------|---------|------|
| Roboflow Universe | YOLO training datasets | [universe.roboflow.com](https://universe.roboflow.com) |
| CricSheet | Ground truth ball-by-ball | [cricsheet.org](https://cricsheet.org) |
| YouTube (IPL/ICC) | Match video clips | Various channels |
| Cric-360 | Annotated broadcast frames | HuggingFace |
