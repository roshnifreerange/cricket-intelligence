# 🏏 Cricket Intelligence Engine — Architecture & Roadmap

> **Mindset:** Modularity · Iteration · Data-first  
> Goal: *"Build a continuously improving dataset of ball-level intelligence from video."*

---

## Table of Contents

1. [What's Built — Phase 1 Status](#phase-1-status)
2. [Every File Explained](#every-file-explained)
3. [Data Flow End-to-End](#data-flow)
4. [Phase 2 — Player & Pitch Insights](#phase-2)
5. [Phase 3 — Strategy & AI Coach](#phase-3)
6. [Continuous Learning Loop](#continuous-learning-loop)
7. [Success Metrics](#success-metrics)

---

## Phase 1 Status — What's Built ✅ {#phase-1-status}

| Module | Status | Notes |
|---|---|---|
| Video ingestion (YouTube + local) | ✅ Done | `src/ingestion/downloader.py` |
| Ball segmentation — uniform split | ✅ Done | `src/segmentation/clip_extractor.py` |
| Ball segmentation — timestamp-based | ✅ Done | Same file, separate method |
| **Batch mode** — Gemini auto-detects balls | ✅ Done | `run_pipeline.py --batch-mode` |
| Gemini vision extraction | ✅ Done | `src/intelligence/extractor.py` |
| Structured JSON schema | ✅ Done | `src/intelligence/schema.py` |
| Validation + normalization | ✅ Done | `src/validation/normalizer.py` |
| SQLite storage | ✅ Done | `src/storage/db.py` |
| Human review UI (Streamlit) | ✅ Done | `ui/app.py` |
| REST API | ✅ Done | `src/api/main.py` |
| Ball trajectory tracking (YOLO) | ✅ Built, not active | `src/tracking/tracker.py` |
| Roboflow CV — player/stump detection | ✅ Built, optional | `src/detection/detect.py` |
| Custom YOLO training pipeline | ✅ Built, future use | `scripts/train_yolo.py` |
| Cric-360 validation | ✅ Built, future use | `scripts/validate_cric360.py` |

**Phase 1 is complete.** You can run the full pipeline today with `--batch-mode` for best results.

---

## Every File Explained {#every-file-explained}

### `run_pipeline.py` — Main Orchestrator
**Purpose:** The single entry point that runs the full pipeline end-to-end.  
**Modes:**
- `--batch-mode` → upload full video to Gemini, auto-detect all deliveries (recommended)
- `--uniform` → split video into fixed-length clips, Gemini analyses each clip
- `--timestamps` → split video using manually provided delivery timestamps

**Key flags:** `--no-cv` (skip Roboflow), `--batch-mode`, `--segment-duration`, `--max-clips`

---

### `src/ingestion/downloader.py` — Video Ingestion
**Purpose:** Download YouTube videos or register local video files.  
**Uses:** `yt-dlp` for YouTube, saves metadata JSON alongside the video.  
**Output:** `data/raw_videos/<match-id>.mp4` + `<match-id>_meta.json`

```bash
python -m src.ingestion.downloader --url "https://youtube.com/..." --match-id my-match
```

---

### `src/segmentation/clip_extractor.py` — Ball Clip Cutter
**Purpose:** Split a full match video into individual delivery clips using `ffmpeg`.  
**Two modes:**
1. `extract_uniform_segments()` — cuts every N seconds blindly (fast, imprecise)
2. `extract_from_timestamps()` — cuts using exact start/end times per ball (precise, needs manual JSON)

**Output:** `data/ball_clips/<match-id>/<match-id>_inn1_ov1_b1.mp4` per delivery  
**Future:** Will be replaced/augmented by YOLO-based automatic timestamp detection

---

### `src/intelligence/schema.py` — Ball Record Data Model
**Purpose:** Defines the complete schema for one ball delivery. Single source of truth.  
**Contains:**
- `BallRecord` — Pydantic model with all ball fields
- Enums: `Line`, `Length`, `ShotType`, `BowlerType`, `Outcome`, `Footwork`, etc.
- `GEMINI_JSON_SCHEMA` — JSON schema passed to Gemini's `response_schema` to force structured output
- `ConfidenceScores` — per-field confidence (0.0–1.0)

**Every other module imports from here. Do not change field names without updating all modules.**

---

### `src/intelligence/prompt.py` — Gemini Prompts
**Purpose:** All prompts sent to Gemini. Kept separate so prompts can be improved without touching logic.  
**Three prompts:**
- `SYSTEM_PROMPT` — "You are an elite cricket analyst with 20+ years of experience..."
- `EXTRACTION_PROMPT` — single ball clip analysis instructions
- `BATCH_EXTRACTION_PROMPT` — full video, return array of deliveries (used by `--batch-mode`)
- `CV_AUGMENTED_TEMPLATE` — when Roboflow geometry is available, injects pixel-precise line/length as grounding facts before asking Gemini for shot/bowler type etc.

**This is the highest-leverage file for improving accuracy.** Better prompts = better results with zero code changes.

---

### `src/intelligence/extractor.py` — Gemini Vision Extractor
**Purpose:** Talks to the Gemini API, uploads videos, parses structured responses.  
**Key methods:**
- `extract_from_clip()` — analyse a single ball clip
- `extract_batch()` — process a directory of clips one by one
- `extract_from_video()` — **batch mode** — upload full video, Gemini returns array of balls

**CV Override logic:** If Roboflow geometry is available AND confident (≥ 0.85), it overrides Gemini's line/length answer (Gemini can hallucinate; pixel math is deterministic).

---

### `src/validation/normalizer.py` — Schema Validator
**Purpose:** Clean up Gemini's output before saving to DB.  
**Does:**
1. **Fuzzy normalization** — `"just outside off"` → `outside_off`, `"back of a length"` → `short_of_length`
2. **Cross-field consistency** — if shot = `leave`, force contact = `miss`; if outcome = `dot`, set `runs_scored = 0`
3. **Confidence flagging** — mark records below 0.5 avg confidence as needing human review
4. **Unknown count** — if 3+ fields are `unknown`, flag as poor extraction quality

---

### `src/storage/db.py` — SQLite Database Layer
**Purpose:** Store and query all ball records using SQLAlchemy ORM.  
**Tables:**
- `matches` — match metadata (teams, format, date)
- `balls` — one row per delivery, all fields + confidence scores + review status

**Key methods:** `save_ball()`, `get_balls_for_match()`, `get_balls_needing_review()`, `update_ball_review()`, `get_stats()`  
**DB file:** `data/cricket_intelligence.db`

---

### `src/api/main.py` — REST API (FastAPI)
**Purpose:** HTTP API to query ball data from external systems or frontends.  
**Endpoints:**
| Method | Path | Action |
|---|---|---|
| GET | `/balls?match_id=X` | List balls for a match |
| GET | `/balls?needs_review=true` | Get low-confidence balls |
| PUT | `/balls/{id}/review` | Submit human correction |
| GET | `/analytics/summary` | Confidence stats, outcome distribution |
| GET | `/clips/{ball_id}` | Serve the video clip file |

```bash
python -m src.api.main
# Docs at http://localhost:8000/docs
```

---

### `ui/app.py` — Streamlit Review Dashboard
**Purpose:** Human review interface. Watch the ball clip, see what Gemini extracted, correct wrong fields.  
**Modes:**
- **Dashboard** — total balls, reviewed count, avg confidence, outcome bar chart
- **Review Balls** — per-ball review: video player + editable dropdowns for all fields
- **Full Dataset** — table view of all balls with CSV export

**This is where ground truth is built.** Every correction here feeds the continuous learning loop.

---

### `src/detection/detect.py` — Roboflow CV + YOLO Detection *(Optional, Future)*
**Purpose:** Computer vision layer for player/ball/stump detection.  
**Status:** Built and functional. Not used in primary pipeline because Roboflow trial expires 28 Apr 2026 and video angles haven't shown stumps clearly in testing.  
**Contains:**
- `CricketDetector` — detects players and ball using Roboflow `cricket-oftm6/3` (mAP 96.2%)
- `DualModelDetector` — runs both `cricket-oftm6/3` (scene) and `stumps/10` (geometry) together
- `LineLengthEstimator` — converts stumps pixel position + ball pixel position into geometric line/length with stumps-width as unit of measurement

**When this becomes useful:** Broadcast-quality, side-on camera footage where stumps are clearly visible. At that point, geometry-based line/length will be significantly more accurate than Gemini's estimation.

---

### `src/tracking/tracker.py` — Ball Trajectory Tracker *(Future)*
**Purpose:** Track the ball across every frame of a video to build a pixel-level trajectory.  
**Uses:** YOLO (`yolov8n.pt` by default, custom model when available)  
**Output:** Per-frame JSON `{frame, time_sec, x, y, confidence}` + optional 2D pitch map PNG  
**Future value:**
- Exact pitch landing point
- Ball speed (distance/time between frames)
- Pre/post-bounce deviation angle (seam/swing measurement)
- Automatic delivery timestamp detection (when ball leaves bowler's hand → when it passes the batsman)

```bash
python -m src.tracking.tracker --video clip.mp4 --output tracked.mp4 --pitch-map
```

---

### `scripts/train_yolo.py` — Custom YOLO Training *(Future)*
**Purpose:** Fine-tune YOLOv8 on cricket-specific data to replace the generic `yolov8n.pt` base model.  
**Pipeline:**
1. Downloads Cric-360 dataset from HuggingFace (`sarimshahzad/Cric-360`)
2. Fine-tunes YOLOv8 on cricket broadcast frames
3. Saves best weights to `models/`
4. Validates on held-out Cric-360 test split

**When to run:** After annotating 500+ ball bounding boxes. The Cric-360 dataset provides scene frames; ball bounding box annotations still need to be added via Roboflow's annotation tool.

---

### `scripts/validate_cric360.py` — Model Validation *(Future)*
**Purpose:** Benchmark how well the current detection model performs on real broadcast footage.  
**Downloads** 100 test frames from `sarimshahzad/Cric-360` (HuggingFace) and runs the Roboflow model, reporting detection rate per class and confidence distribution.

---

### `data/roboflowinference.py` — Quick Roboflow Test Script
**Purpose:** Standalone script to test a single image through the Roboflow model and print raw predictions. Used for debugging and verifying API connectivity.

---

## Data Flow End-to-End {#data-flow}

```
YouTube URL or local .mp4
        │
        ▼
src/ingestion/downloader.py
  → data/raw_videos/<match-id>.mp4
        │
        ├── BATCH MODE (recommended)
        │   └── src/intelligence/extractor.py::extract_from_video()
        │       → upload full video to Gemini
        │       → Gemini returns JSON array [ball_1, ball_2, ...]
        │
        └── SEGMENTED MODE
            └── src/segmentation/clip_extractor.py
                → data/ball_clips/<match-id>/*.mp4
                    │
                    ├── [optional] src/detection/detect.py::DualModelDetector
                    │   → Roboflow geometry (line, length from pixels)
                    │
                    └── src/intelligence/extractor.py::extract_from_clip()
                        → Gemini per clip
        │
        ▼
src/intelligence/schema.py::BallRecord
        │
        ▼
src/validation/normalizer.py
  → fuzzy text normalization
  → cross-field consistency
  → confidence flagging
        │
        ▼
src/storage/db.py → data/cricket_intelligence.db
        │
        ├── data/<match-id>_extracted.json
        ├── ui/app.py  (Streamlit review UI)
        └── src/api/main.py  (REST API)
```

---

## Phase 2 — Player & Pitch Insights {#phase-2}

**Objective:** Answer questions like:
- *"What's Kohli's weakness against short-pitched pace outside off?"*
- *"Which bowler has the best death over economy in the last 3 matches?"*
- *"Where does this pitch assist reverse swing vs spin?"*

### What Phase 1 already provides for Phase 2

Every ball record already has: `bowler_name`, `batsman_name`, `line`, `length`, `shot_type`, `outcome`, `movement`, `bounce_behavior`. Once you have 500+ reviewed balls, Phase 2 analytics are simple SQL/Pandas aggregations on top of the existing DB.

### Files to build

**`src/analytics/player_engine.py`**
```python
# Query patterns from DB
def bowler_line_length_heatmap(bowler_name, match_ids) → dict
def batsman_weakness_profile(batsman_name) → dict
  # e.g. {"outside_off + short_of_length": {"balls": 42, "wickets": 8, "boundary_pct": 0.05}}
def head_to_head(bowler, batsman) → dict
```

**`src/analytics/pitch_engine.py`**
```python
def pitch_behavior_by_zone(match_id) → dict
  # split pitch into 6 zones, count bounce_behavior per zone
def movement_frequency(match_id) → dict
  # how much seam/swing/turn in this match
```

**`ui/analytics.py`** — Add a new Streamlit page:
- Heatmap of ball landing zones per bowler
- Batsman weakness radar chart
- Head-to-head comparison table

### Strategy
1. Get 200+ reviewed balls from Phase 1 UI
2. Build aggregation queries in `player_engine.py` — no new AI needed
3. Add analytics page to Streamlit UI
4. Expose via new API endpoints in `main.py`

---

## Phase 3 — Strategy & AI Coach {#phase-3}

**Objective:** Proactive recommendations:
- *"For this batsman, pitch it up on off stump — he's averaging 12 in that zone"*
- *"This pitch is taking turn from over 15 — bring spinner on now"*

### What's needed beyond Phase 2

| Component | Approach |
|---|---|
| **Match context awareness** | Feed current match state (score, wickets, overs) into prompts |
| **Historical pattern DB** | Phase 2 analytics DB + CricSheet ground truth linked by player name |
| **Recommendation engine** | LLM (Gemini) queried with structured context: `"Batsman X, pitch Y, match situation Z → suggest bowling plan"` |
| **Confidence calibration** | Compare Gemini's Phase 1 outputs to CricSheet ground truth → measure accuracy → weight recommendations |

### Files to build

**`src/intelligence/coach.py`**
```python
class AICoach:
    def bowling_plan(self, batsman_name, pitch_behavior, match_situation) → str
    def field_placement(self, batsman_weakness, bowler_type) → dict
    def over_strategy(self, remaining_overs, required_rate, batsman_profile) → str
```

**`src/analytics/cricsheet_linker.py`**  
Link ball records to CricSheet data by match + over + ball number → enables accuracy measurement by comparing extracted line/length to ground truth.

### Strategy
1. Complete Phase 2 (player profiles must exist before strategy is meaningful)
2. Download CricSheet data for your test matches to create ground truth
3. Build `AICoach` as a Gemini-backed reasoning layer with structured context injection
4. Gate recommendations behind confidence thresholds — only suggest when pattern has ≥ 10 data points

---

## Continuous Learning Loop {#continuous-learning-loop}

```
Gemini extracts ball data
        │
        ▼
Confidence < 0.5 OR unknowns ≥ 3?
        │
       YES → flagged for human review in UI
        │
        ▼
Human corrects in ui/app.py
        │
        ▼
Corrected record saved (is_reviewed=True, reviewed_by="human")
        │
        ▼
Ground truth dataset grows in SQLite
        │
        ├── Short-term: improves prompt engineering
        │   (look at patterns in corrections → refine prompt.py)
        │
        └── Long-term: fine-tune custom model on corrections
            (scripts/train_yolo.py for CV, or Gemini fine-tuning via API)
```

**Key insight:** Every human correction in the UI is a labeled training example. The `is_reviewed` flag and `review_notes` field in the schema exist precisely to track this.

---

## Success Metrics {#success-metrics}

### Phase 1 (Now)
| Metric | Target |
|---|---|
| % balls correctly classified (manual check) | > 75% |
| Avg confidence across all fields | > 0.75 |
| % "unknown" outputs | < 20% |
| Processing time per ball (batch mode) | < 30 sec |
| Balls in reviewed dataset | 200+ |

### Phase 2
| Metric | Target |
|---|---|
| Bowler line/length accuracy vs CricSheet | > 80% |
| Batsman weakness profiles built | > 10 players |

### Phase 3
| Metric | Target |
|---|---|
| Coach recommendation acceptance rate (human eval) | > 60% |
| Strategy suggestions backed by ≥ 10 data points | 100% |

---

## Non-Goals (Current Phase)

- Exact ball speed detection (requires calibrated camera or speed gun integration)
- Real-time streaming analysis
- Field placement reconstruction from video
- Perfect ball tracking (YOLO tracker works but needs custom trained weights)
- Captain decision engine (Phase 3+)
