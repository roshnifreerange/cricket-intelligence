"""
Cricket Intelligence Engine - FastAPI REST API
Endpoints for the cricket intelligence pipeline.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.storage.db import CricketDB

app = FastAPI(
    title="Cricket Intelligence Engine API",
    description="Ball-level video understanding for cricket",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = CricketDB()


class MatchCreate(BaseModel):
    match_id: str
    format: str = "T20"
    team_a: str = ""
    team_b: str = ""


class BallUpdate(BaseModel):
    line: Optional[str] = None
    length: Optional[str] = None
    shot_type: Optional[str] = None
    outcome: Optional[str] = None
    reviewed_by: str = "human"


@app.get("/")
def root():
    return {"service": "Cricket Intelligence Engine", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "db_stats": db.get_stats()}


@app.post("/matches")
def create_match(match: MatchCreate):
    db.create_match(match.model_dump())
    return {"message": f"Match {match.match_id} created"}


@app.get("/matches")
def list_matches():
    matches = db.list_matches()
    return [{"match_id": m.match_id, "format": m.format, "team_a": m.team_a, "team_b": m.team_b} for m in matches]


@app.get("/balls")
def list_balls(match_id: Optional[str] = None, needs_review: bool = False):
    if needs_review:
        balls = db.get_balls_needing_review(match_id)
    elif match_id:
        balls = db.get_balls_for_match(match_id)
    else:
        return []

    return [{
        "ball_id": b.ball_id, "over": b.over_number, "ball": b.ball_number,
        "bowler_type": b.bowler_type, "line": b.line, "length": b.length,
        "shot_type": b.shot_type, "outcome": b.outcome,
        "confidence_avg": round((b.confidence_line + b.confidence_length + b.confidence_shot_type) / 3, 2),
        "is_reviewed": b.is_reviewed, "raw_description": b.raw_description,
        "clip_path": b.clip_path,
    } for b in balls]


@app.put("/balls/{ball_id}/review")
def review_ball(ball_id: str, update: BallUpdate):
    updates = {k: v for k, v in update.model_dump().items() if v is not None and k != "reviewed_by"}
    success = db.update_ball_review(ball_id, updates, reviewed_by=update.reviewed_by)
    if not success:
        raise HTTPException(status_code=404, detail="Ball not found")
    return {"message": f"Ball {ball_id} reviewed"}


@app.get("/analytics/summary")
def analytics_summary(match_id: Optional[str] = None):
    return db.get_stats(match_id)


@app.get("/clips/{ball_id}")
def serve_clip(ball_id: str):
    ball = db.get_ball(ball_id)
    if not ball or not ball.clip_path:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip_path = Path(ball.clip_path)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file missing")
    return FileResponse(clip_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
