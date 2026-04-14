"""
Cricket Intelligence Engine - Database Storage Layer
SQLAlchemy models and database operations for ball-level cricket data.
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

Base = declarative_base()


# ===== SQLAlchemy Models =====

class MatchRecord(Base):
    __tablename__ = "matches"

    match_id = Column(String, primary_key=True)
    format = Column(String, default="T20")
    team_a = Column(String, default="")
    team_b = Column(String, default="")
    venue = Column(String, nullable=True)
    date = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    video_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class BallDBRecord(Base):
    __tablename__ = "balls"

    ball_id = Column(String, primary_key=True)
    match_id = Column(String, ForeignKey("matches.match_id"), nullable=False)
    innings = Column(Integer, default=1)
    over_number = Column(Integer, default=0)
    ball_number = Column(Integer, default=1)

    # Players
    bowler_name = Column(String, nullable=True)
    batsman_name = Column(String, nullable=True)

    # Bowling analysis
    bowler_type = Column(String, default="unknown")
    line = Column(String, default="unknown")
    length = Column(String, default="unknown")
    variation = Column(String, default="none")
    bounce_behavior = Column(String, default="unknown")
    movement = Column(String, default="unknown")

    # Batting analysis
    shot_type = Column(String, default="unknown")
    footwork = Column(String, default="unknown")
    contact_quality = Column(String, default="unknown")

    # Result
    outcome = Column(String, default="unknown")
    runs_scored = Column(Integer, default=0)

    # Confidence scores
    confidence_bowler_type = Column(Float, default=0.0)
    confidence_line = Column(Float, default=0.0)
    confidence_length = Column(Float, default=0.0)
    confidence_shot_type = Column(Float, default=0.0)
    confidence_outcome = Column(Float, default=0.0)
    confidence_contact = Column(Float, default=0.0)

    # Metadata
    clip_path = Column(String, nullable=True)
    clip_start_time = Column(String, nullable=True)
    clip_end_time = Column(String, nullable=True)
    raw_description = Column(Text, nullable=True)

    # Review status
    is_reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String, nullable=True)
    review_notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ===== Database Manager =====

class CricketDB:
    """Database operations for cricket intelligence data."""

    def __init__(self, db_url: str = None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./data/cricket_intelligence.db")

        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        console.print(f"[green]✓[/green] Database initialized: {db_url}")

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ===== Match Operations =====

    def create_match(self, match_data: dict) -> MatchRecord:
        session = self.get_session()
        try:
            record = MatchRecord(**match_data)
            session.merge(record)
            session.commit()
            console.print(f"[green]✓[/green] Match saved: {match_data['match_id']}")
            return record
        finally:
            session.close()

    def get_match(self, match_id: str) -> Optional[MatchRecord]:
        session = self.get_session()
        try:
            return session.query(MatchRecord).filter_by(match_id=match_id).first()
        finally:
            session.close()

    def list_matches(self) -> list[MatchRecord]:
        session = self.get_session()
        try:
            return session.query(MatchRecord).all()
        finally:
            session.close()

    # ===== Ball Operations =====

    def save_ball(self, ball_record) -> BallDBRecord:
        """Save a BallRecord (Pydantic model) to the database."""
        session = self.get_session()
        try:
            db_record = BallDBRecord(
                ball_id=ball_record.ball_id,
                match_id=ball_record.match_id,
                innings=ball_record.innings,
                over_number=ball_record.over,
                ball_number=ball_record.ball_number,
                bowler_name=ball_record.bowler_name,
                batsman_name=ball_record.batsman_name,
                bowler_type=ball_record.bowler_type.value,
                line=ball_record.line.value,
                length=ball_record.length.value,
                variation=ball_record.variation.value,
                shot_type=ball_record.shot_type.value,
                footwork=ball_record.footwork.value,
                contact_quality=ball_record.contact_quality.value,
                outcome=ball_record.outcome.value,
                runs_scored=ball_record.runs_scored,
                bounce_behavior=ball_record.bounce_behavior.value,
                movement=ball_record.movement.value,
                confidence_bowler_type=ball_record.confidence.bowler_type,
                confidence_line=ball_record.confidence.line,
                confidence_length=ball_record.confidence.length,
                confidence_shot_type=ball_record.confidence.shot_type,
                confidence_outcome=ball_record.confidence.outcome,
                confidence_contact=ball_record.confidence.contact_quality,
                clip_path=ball_record.clip_path,
                raw_description=ball_record.raw_description,
                is_reviewed=ball_record.is_reviewed,
            )
            session.merge(db_record)
            session.commit()
            return db_record
        finally:
            session.close()

    def save_balls_batch(self, ball_records: list) -> int:
        """Save multiple ball records at once."""
        count = 0
        for record in ball_records:
            try:
                self.save_ball(record)
                count += 1
            except Exception as e:
                console.print(f"[red]✗[/red] Error saving {record.ball_id}: {e}")
        console.print(f"[green]✓[/green] Saved {count}/{len(ball_records)} ball records")
        return count

    def get_ball(self, ball_id: str) -> Optional[BallDBRecord]:
        session = self.get_session()
        try:
            return session.query(BallDBRecord).filter_by(ball_id=ball_id).first()
        finally:
            session.close()

    def get_balls_for_match(self, match_id: str) -> list[BallDBRecord]:
        session = self.get_session()
        try:
            return (
                session.query(BallDBRecord)
                .filter_by(match_id=match_id)
                .order_by(BallDBRecord.innings, BallDBRecord.over_number, BallDBRecord.ball_number)
                .all()
            )
        finally:
            session.close()

    def get_balls_needing_review(self, match_id: str = None) -> list[BallDBRecord]:
        """Get all unreviewed low-confidence balls."""
        session = self.get_session()
        try:
            query = session.query(BallDBRecord).filter_by(is_reviewed=False)
            if match_id:
                query = query.filter_by(match_id=match_id)
            # Order by lowest confidence first
            return query.order_by(
                (BallDBRecord.confidence_line + BallDBRecord.confidence_length + BallDBRecord.confidence_shot_type)
            ).all()
        finally:
            session.close()

    def update_ball_review(
        self,
        ball_id: str,
        updates: dict,
        reviewed_by: str = "human",
    ) -> bool:
        """Update a ball record after human review."""
        session = self.get_session()
        try:
            record = session.query(BallDBRecord).filter_by(ball_id=ball_id).first()
            if not record:
                return False

            for key, value in updates.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            record.is_reviewed = True
            record.reviewed_by = reviewed_by
            record.updated_at = datetime.now()
            session.commit()
            return True
        finally:
            session.close()

    # ===== Analytics Queries =====

    def get_stats(self, match_id: str = None) -> dict:
        """Get summary statistics."""
        session = self.get_session()
        try:
            query = session.query(BallDBRecord)
            if match_id:
                query = query.filter_by(match_id=match_id)

            balls = query.all()
            if not balls:
                return {"total": 0}

            total = len(balls)
            reviewed = sum(1 for b in balls if b.is_reviewed)
            outcomes = {}
            for b in balls:
                outcomes[b.outcome] = outcomes.get(b.outcome, 0) + 1

            avg_confidence = sum(
                (b.confidence_line + b.confidence_length + b.confidence_shot_type) / 3
                for b in balls
            ) / total

            return {
                "total": total,
                "reviewed": reviewed,
                "review_pct": reviewed / total * 100,
                "avg_confidence": round(avg_confidence, 3),
                "outcomes": outcomes,
                "unknown_count": sum(
                    1 for b in balls
                    if b.line == "unknown" or b.length == "unknown" or b.shot_type == "unknown"
                ),
            }
        finally:
            session.close()


# ===== CLI Entry Point =====
if __name__ == "__main__":
    db = CricketDB()
    stats = db.get_stats()
    console.print(f"\n[bold]Database Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")
