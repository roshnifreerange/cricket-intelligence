# 🏏 Cricket Intelligence Engine — C4 & Sequence Diagrams

All diagrams use [Mermaid](https://mermaid.js.org/) syntax — rendered natively in GitHub, VS Code, and Notion.

---

## C1 — System Context Diagram

> Who uses the system and what external services does it depend on?

```mermaid
C4Context
    title System Context — Cricket Intelligence Engine

    Person(analyst, "Cricket Analyst", "Reviews extracted ball data, corrects AI outputs, uses insights for strategy")
    Person(coach, "Team Coach", "Consumes player weakness reports and bowling recommendations")

    System(cie, "Cricket Intelligence Engine", "Converts cricket match videos into structured ball-level intelligence — line, length, shot type, outcome, confidence scores")

    System_Ext(youtube, "YouTube", "Source of match highlight videos")
    System_Ext(gemini, "Google Gemini API", "Vision AI that watches video clips and extracts structured cricket intelligence")
    System_Ext(roboflow, "Roboflow API", "Optional CV layer — detects players, ball, stumps for geometric line/length (trial expires Apr 2026)")
    System_Ext(huggingface, "HuggingFace", "Hosts Cric-360 dataset (3,558 broadcast frames) for future YOLO training")
    System_Ext(cricsheet, "CricSheet", "Ground truth ball-by-ball data for accuracy measurement (Phase 2+)")

    Rel(analyst, cie, "Reviews ball records, corrects AI outputs", "Browser / Streamlit")
    Rel(coach, cie, "Reads player weakness reports", "Browser / REST API")
    Rel(cie, youtube, "Downloads match videos", "yt-dlp / HTTPS")
    Rel(cie, gemini, "Sends video clips for structured extraction", "HTTPS / REST")
    Rel(cie, roboflow, "Runs player & stump detection (optional)", "HTTPS / REST")
    Rel(cie, huggingface, "Downloads Cric-360 training data (future)", "HTTPS")
    Rel(cie, cricsheet, "Imports ground truth for validation (future)", "CSV / API")
```

---

## C2 — Container Diagram

> What are the deployable units inside the system?

```mermaid
C4Container
    title Container Diagram — Cricket Intelligence Engine

    Person(analyst, "Cricket Analyst")

    Container_Boundary(cie, "Cricket Intelligence Engine") {
        Container(pipeline, "Pipeline Orchestrator", "Python / CLI", "run_pipeline.py — coordinates all steps from ingestion to storage")
        Container(ingestion, "Video Ingestion", "Python", "Downloads YouTube videos or registers local files via yt-dlp")
        Container(segmentation, "Clip Segmentation", "Python / ffmpeg", "Splits full match video into per-delivery clips")
        Container(gemini_ext, "Gemini Extractor", "Python / Google GenAI SDK", "Uploads clips to Gemini, parses structured JSON responses")
        Container(cv, "CV Detection Layer", "Python / Roboflow SDK", "Optional: detects players, ball, stumps — computes geometric line/length")
        Container(tracker, "Ball Tracker", "Python / YOLO / OpenCV", "Frame-by-frame ball trajectory, pitch map generation (future)")
        Container(validator, "Validator", "Python / Pydantic", "Normalizes fuzzy text, cross-field consistency checks, confidence flagging")
        Container(db, "Database", "SQLite / SQLAlchemy", "Stores all ball records, match metadata, review status")
        Container(api, "REST API", "Python / FastAPI", "HTTP endpoints for querying, reviewing, and exporting ball data")
        Container(ui, "Review Dashboard", "Python / Streamlit", "Human review UI — watch clips, correct AI fields, build ground truth")
    }

    System_Ext(gemini_api, "Google Gemini API")
    System_Ext(roboflow_api, "Roboflow API")
    System_Ext(youtube, "YouTube")

    Rel(analyst, ui, "Reviews and corrects ball data", "HTTPS / Browser")
    Rel(analyst, api, "Queries ball records programmatically", "HTTPS / REST")
    Rel(pipeline, ingestion, "Step 1: ingest video")
    Rel(pipeline, segmentation, "Step 2: segment into clips")
    Rel(pipeline, cv, "Step 2.5: optional CV analysis")
    Rel(pipeline, gemini_ext, "Step 3: extract intelligence")
    Rel(pipeline, validator, "Step 4: validate & normalize")
    Rel(pipeline, db, "Step 5: store records")
    Rel(ui, db, "Read / write ball records")
    Rel(api, db, "Read / write ball records")
    Rel(ingestion, youtube, "Downloads video", "yt-dlp")
    Rel(gemini_ext, gemini_api, "Uploads video, receives JSON", "HTTPS")
    Rel(cv, roboflow_api, "Sends frame, receives detections", "HTTPS")
```

---

## C3 — Component Diagram: Intelligence Module

> Components inside the most important container — the Gemini Intelligence layer

```mermaid
C4Component
    title Component Diagram — Intelligence Module (src/intelligence/)

    Container_Boundary(intel, "Intelligence Module") {
        Component(schema, "schema.py", "Pydantic / Python Enums", "BallRecord model, all field enums (Line, Length, ShotType...), GEMINI_JSON_SCHEMA for structured output forcing")
        Component(prompt, "prompt.py", "Python strings", "SYSTEM_PROMPT (cricket analyst persona), EXTRACTION_PROMPT (single clip), BATCH_EXTRACTION_PROMPT (full video auto-detect), CV_AUGMENTED_TEMPLATE (inject Roboflow geometry)")
        Component(extractor, "extractor.py", "Python / Google GenAI", "GeminiExtractor class: extract_from_clip(), extract_batch(), extract_from_video(). Handles file upload, response parsing, CV override logic")
    }

    Container(pipeline, "Pipeline Orchestrator", "run_pipeline.py")
    Container(validator, "Validator", "src/validation/normalizer.py")
    System_Ext(gemini, "Google Gemini API")

    Rel(pipeline, extractor, "calls extract_from_video() or extract_batch()")
    Rel(extractor, prompt, "fetches prompt text for each mode")
    Rel(extractor, schema, "parses response into BallRecord, reads GEMINI_JSON_SCHEMA")
    Rel(extractor, gemini, "uploads video file, sends prompt, receives JSON array", "HTTPS")
    Rel(extractor, validator, "returns List[BallRecord] for validation")
    Rel(schema, prompt, "GEMINI_JSON_SCHEMA used in response_schema parameter")
```

---

## C3 — Component Diagram: Storage & API

> Components inside the data storage and API layer

```mermaid
C4Component
    title Component Diagram — Storage & API Layer

    Container_Boundary(storage, "Storage & API") {
        Component(db, "db.py", "SQLAlchemy / SQLite", "CricketDB class. Tables: matches, balls. CRUD operations: save_ball, get_balls_for_match, get_balls_needing_review, update_ball_review, get_stats")
        Component(models, "ORM Models", "SQLAlchemy", "MatchRecord (matches table), BallDBRecord (balls table) — all fields including confidence scores and review flags")
        Component(api, "main.py", "FastAPI", "REST endpoints: GET /balls, PUT /balls/{id}/review, GET /analytics/summary, GET /clips/{ball_id}, POST /matches")
        Component(ui_app, "app.py", "Streamlit", "3 modes: Dashboard (charts), Review Balls (video + edit form), Full Dataset (table + CSV export)")
    }

    Container(validator, "Validator")
    Person(analyst, "Cricket Analyst")

    Rel(validator, db, "save_balls_batch(validated_records)")
    Rel(db, models, "reads/writes via SQLAlchemy ORM")
    Rel(api, db, "queries CricketDB for all endpoints")
    Rel(ui_app, db, "reads balls, writes reviewed corrections")
    Rel(analyst, ui_app, "watches clips, corrects fields", "Browser")
    Rel(analyst, api, "programmatic access", "REST / HTTPS")
```

---

## C3 — Component Diagram: CV & Tracking Layer (Future)

> The optional computer vision layer — active when Roboflow API key is valid

```mermaid
C4Component
    title Component Diagram — CV & Tracking Layer (src/detection/ + src/tracking/)

    Container_Boundary(cv_layer, "CV & Tracking Layer") {
        Component(cricket_detector, "CricketDetector", "Python / Requests", "Single-model cloud inference on cricket-oftm6/3 (mAP 96.2%). Detects: ball, batsman, bowler, wicketkeeper, nonstriker, umpire")
        Component(dual_detector, "DualModelDetector", "Python / Requests", "Runs BOTH cricket-oftm6/3 (scene) and stumps/10 (geometry). Fuses results — players from Model 1, stumps exclusively from Model 2")
        Component(estimator, "LineLengthEstimator", "Python / NumPy", "Geometric calculation: norm_x = (ball_cx - stump_cx) / stump_width → line bucket. norm_y = (ball_cy - stump_bottom) / stump_height → length bucket")
        Component(http_client, "_RoboflowHTTPClient", "Python / Requests", "Calls Roboflow serverless REST API directly (no SDK). Sends base64-encoded JPEG, receives prediction JSON")
        Component(tracker, "BallTracker", "Python / YOLO / OpenCV", "Frame-by-frame YOLO detection → trajectory list [{frame, x, y, conf}]. Draws colored trail overlay. Generates 2D pitch map PNG")
    }

    System_Ext(roboflow, "Roboflow Serverless API")
    Container(extractor, "Gemini Extractor")

    Rel(dual_detector, http_client, "calls infer() for each model")
    Rel(dual_detector, estimator, "passes stumps + ball detections")
    Rel(http_client, roboflow, "POST base64 image → predictions JSON", "HTTPS")
    Rel(extractor, dual_detector, "receives cv_context for CV override", "optional")
    Rel(cricket_detector, http_client, "uses same HTTP client")
```

---

## Sequence Diagram 1 — Batch Mode Pipeline (Primary)

> Full flow when user runs `python run_pipeline.py --batch-mode`

```mermaid
sequenceDiagram
    actor User
    participant CLI as run_pipeline.py
    participant Ingest as downloader.py
    participant Extractor as extractor.py
    participant Gemini as Google Gemini API
    participant Validator as normalizer.py
    participant DB as db.py (SQLite)
    participant UI as app.py (Streamlit)

    User->>CLI: python run_pipeline.py --video match.mp4 --batch-mode
    CLI->>Ingest: register_local_video(match.mp4)
    Ingest-->>CLI: metadata {match_id, video_path, duration}

    Note over CLI: Skip segmentation entirely

    CLI->>Extractor: extract_from_video(match.mp4, match_id)
    Extractor->>Gemini: files.upload(match.mp4)
    Gemini-->>Extractor: uploaded_file.uri
    Extractor->>Gemini: generate_content(video_uri, BATCH_EXTRACTION_PROMPT, response_schema=array)
    Gemini-->>Extractor: JSON array [{ball_1}, {ball_2}, {ball_3}, {ball_4}]
    Extractor-->>CLI: List[BallRecord] (4 records)

    CLI->>Validator: validate_batch(records)
    Validator->>Validator: normalize fuzzy text
    Validator->>Validator: cross-field consistency checks
    Validator->>Validator: flag low confidence records
    Validator-->>CLI: (validated_records, stats)

    CLI->>DB: create_match(match_id)
    CLI->>DB: save_balls_batch(validated_records)
    DB-->>CLI: 4/4 saved

    CLI->>CLI: export JSON to data/match_extracted.json
    CLI-->>User: ✅ Pipeline Complete! (4 balls, avg conf 0.91)

    User->>UI: streamlit run ui/app.py
    UI->>DB: get_balls_for_match(match_id)
    DB-->>UI: List[BallDBRecord]
    UI-->>User: Dashboard + Review interface
```

---

## Sequence Diagram 2 — Segmented Mode with Roboflow CV

> Flow when using `--uniform` mode with Roboflow pre-analysis active

```mermaid
sequenceDiagram
    actor User
    participant CLI as run_pipeline.py
    participant Seg as clip_extractor.py
    participant CV as DualModelDetector
    participant Roboflow as Roboflow API
    participant Extractor as extractor.py
    participant Gemini as Google Gemini API
    participant DB as db.py

    User->>CLI: python run_pipeline.py --video match.mp4 --uniform --segment-duration 8

    loop for each clip (up to max-clips)
        CLI->>Seg: extract_uniform_segments(video, segment_duration=8)
        Seg->>Seg: ffmpeg -ss 0 -to 8 → clip_ov1_b1.mp4
        Seg-->>CLI: clip paths list
    end

    Note over CLI: Step 2.5 — CV pre-analysis (one key frame per clip)

    loop for each clip
        CLI->>CV: analyze_frame(frame at 40% into clip)
        CV->>Roboflow: POST base64 frame → cricket-oftm6/3
        Roboflow-->>CV: scene detections (players, ball)
        CV->>Roboflow: POST base64 frame → stumps/10
        Roboflow-->>CV: stumps + ball detections
        CV->>CV: LineLengthEstimator.estimate(ball, stumps)
        CV-->>CLI: {geometry: {line, length, norm_x, norm_y}, scene: {...}}
    end

    loop for each clip
        CLI->>Extractor: extract_from_clip(clip, cv_context)
        Extractor->>Gemini: upload clip + CV_AUGMENTED_PROMPT (with geometry facts injected)
        Gemini-->>Extractor: JSON {bowler_type, shot_type, footwork, outcome, ...}
        Extractor->>Extractor: CV override: if stumps detected, use pixel geometry for line/length
        Extractor-->>CLI: BallRecord
    end

    CLI->>DB: save_balls_batch(records)
```

---

## Sequence Diagram 3 — Human Review & Correction Loop

> How analyst corrections build ground truth over time

```mermaid
sequenceDiagram
    actor Analyst
    participant UI as app.py (Streamlit)
    participant DB as db.py (SQLite)
    participant API as main.py (FastAPI)

    Note over DB: After pipeline run — balls saved with is_reviewed=False

    Analyst->>UI: Open Review Balls mode
    UI->>DB: get_balls_needing_review() [ordered by lowest confidence first]
    DB-->>UI: List of unreviewed balls (sorted: conf=0.00 first)

    loop for each ball needing review
        UI-->>Analyst: Show: video clip + AI-extracted fields + confidence scores
        Analyst->>Analyst: Watch video, assess AI output
        alt AI output is correct
            Analyst->>UI: Click "Save Review" (no changes)
            UI->>DB: update_ball_review(ball_id, {}, is_reviewed=True)
        else AI output is wrong
            Analyst->>UI: Change line="outside_off", length="good", shot_type="drive"
            UI->>DB: update_ball_review(ball_id, corrections, reviewed_by="human")
            Note over DB: is_reviewed=True, review_notes saved
        end
    end

    Note over DB: Ground truth dataset grows with each correction

    Analyst->>API: GET /analytics/summary
    API->>DB: get_stats()
    DB-->>API: {total, reviewed, avg_confidence, outcomes}
    API-->>Analyst: JSON summary

    Note over UI,DB: Corrections feed back into prompt improvement (manual) and future fine-tuning (automatic)
```

---

## Sequence Diagram 4 — Phase 2 Player Weakness Engine (Future)

> How Phase 2 analytics will work once 500+ reviewed balls exist

```mermaid
sequenceDiagram
    actor Coach
    participant UI as analytics.py (Phase 2 UI)
    participant Analytics as player_engine.py (to build)
    participant DB as db.py (SQLite)
    participant Gemini as Google Gemini API

    Coach->>UI: "Show Kohli's weakness profile"

    UI->>Analytics: batsman_weakness_profile("Kohli")
    Analytics->>DB: SELECT line, length, outcome, shot_type FROM balls WHERE batsman_name='Kohli'
    DB-->>Analytics: 200 ball records
    Analytics->>Analytics: group by (line, length) → compute wicket%, boundary%, dot%
    Analytics-->>UI: {outside_off+short: {balls:42, wickets:8, wkt_pct:19%}, ...}

    UI-->>Coach: Heatmap: "Outside off, short of length → 19% wicket rate"

    Coach->>UI: "Suggest bowling plan for next 2 overs"
    UI->>Gemini: generate_content(context={batsman_profile, pitch_behavior, match_state, overs_left})
    Gemini-->>UI: "Bowl outside off, back of length. Set 3 slips. Change of pace every 3rd ball."
    UI-->>Coach: AI bowling plan with data backing
```

---

## Entity Relationship Diagram — Database Schema

```mermaid
erDiagram
    MATCHES {
        string match_id PK
        string format
        string team_a
        string team_b
        string venue
        string date
        string source_url
        string video_path
        datetime created_at
    }

    BALLS {
        string ball_id PK
        string match_id FK
        int innings
        int over_number
        int ball_number
        string bowler_name
        string batsman_name
        string bowler_type
        string line
        string length
        string variation
        string shot_type
        string footwork
        string contact_quality
        string outcome
        int runs_scored
        string bounce_behavior
        string movement
        float confidence_line
        float confidence_length
        float confidence_shot_type
        float confidence_outcome
        float confidence_bowler_type
        float confidence_contact
        string clip_path
        text raw_description
        bool is_reviewed
        string reviewed_by
        text review_notes
        datetime created_at
        datetime updated_at
    }

    MATCHES ||--o{ BALLS : "has many"
```
