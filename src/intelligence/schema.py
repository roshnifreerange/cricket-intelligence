"""
Cricket Intelligence Engine - Ball-Level Schema
Defines the structured data model for ball-by-ball cricket analysis.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


# ===== Enum Definitions =====

class BowlerType(str, Enum):
    PACE = "pace"
    SPIN = "spin"
    UNKNOWN = "unknown"


class Line(str, Enum):
    OUTSIDE_OFF = "outside_off"
    OFF_STUMP = "off_stump"
    MIDDLE = "middle"
    LEG = "leg"
    OUTSIDE_LEG = "outside_leg"
    UNKNOWN = "unknown"


class Length(str, Enum):
    YORKER = "yorker"
    FULL = "full"
    GOOD = "good"
    SHORT_OF_LENGTH = "short_of_length"
    SHORT = "short"
    UNKNOWN = "unknown"


class Variation(str, Enum):
    NONE = "none"
    SLOWER = "slower"
    CUTTER = "cutter"
    BOUNCER = "bouncer"
    YORKER = "yorker"
    SPIN_VARIATION = "spin_variation"
    UNKNOWN = "unknown"


class ShotType(str, Enum):
    DRIVE = "drive"
    CUT = "cut"
    PULL = "pull"
    HOOK = "hook"
    DEFEND = "defend"
    SWEEP = "sweep"
    REVERSE_SWEEP = "reverse_sweep"
    GLANCE = "glance"
    FLICK = "flick"
    LOFTED = "lofted"
    LEAVE = "leave"
    UNKNOWN = "unknown"


class Footwork(str, Enum):
    FRONT_FOOT = "front_foot"
    BACK_FOOT = "back_foot"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ContactQuality(str, Enum):
    CLEAN = "clean"
    MISTIMED = "mistimed"
    EDGE = "edge"
    MISS = "miss"
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    DOT = "dot"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    SIX = "6"
    WICKET = "wicket"
    WIDE = "wide"
    NO_BALL = "no_ball"
    UNKNOWN = "unknown"


class BounceBehavior(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    STEEP = "steep"
    UNKNOWN = "unknown"


class Movement(str, Enum):
    NONE = "none"
    SEAM = "seam"
    SWING = "swing"
    TURN = "turn"
    UNKNOWN = "unknown"


# ===== Confidence Scores =====

class ConfidenceScores(BaseModel):
    """Confidence scores for key fields (0.0 to 1.0)"""
    bowler_type: float = Field(default=0.0, ge=0.0, le=1.0)
    line: float = Field(default=0.0, ge=0.0, le=1.0)
    length: float = Field(default=0.0, ge=0.0, le=1.0)
    shot_type: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: float = Field(default=0.0, ge=0.0, le=1.0)
    contact_quality: float = Field(default=0.0, ge=0.0, le=1.0)


# ===== Main Ball Record =====

class BallRecord(BaseModel):
    """Complete structured record for a single ball delivery."""
    ball_id: str = Field(..., description="Unique identifier: match_over_ball e.g. 'match001_14_3'")
    match_id: str = Field(..., description="Match identifier")
    innings: int = Field(default=1, ge=1, le=4)
    over: int = Field(default=0, ge=0)
    ball_number: int = Field(default=1, ge=1, le=10)  # up to 10 for extras

    # Players
    bowler_name: Optional[str] = None
    batsman_name: Optional[str] = None

    # Bowling analysis
    bowler_type: BowlerType = BowlerType.UNKNOWN
    line: Line = Line.UNKNOWN
    length: Length = Length.UNKNOWN
    variation: Variation = Variation.NONE
    bounce_behavior: BounceBehavior = BounceBehavior.UNKNOWN
    movement: Movement = Movement.UNKNOWN

    # Batting analysis
    shot_type: ShotType = ShotType.UNKNOWN
    footwork: Footwork = Footwork.UNKNOWN
    contact_quality: ContactQuality = ContactQuality.UNKNOWN

    # Result
    outcome: Outcome = Outcome.UNKNOWN
    runs_scored: int = Field(default=0, ge=0)

    # Confidence
    confidence: ConfidenceScores = Field(default_factory=ConfidenceScores)

    # Metadata
    clip_path: Optional[str] = None
    clip_start_time: Optional[str] = None
    clip_end_time: Optional[str] = None
    raw_description: Optional[str] = None
    is_reviewed: bool = False
    reviewed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MatchMetadata(BaseModel):
    """Metadata for a cricket match."""
    match_id: str
    format: str = Field(default="T20", description="T20, ODI, or Test")
    team_a: str = ""
    team_b: str = ""
    venue: Optional[str] = None
    date: Optional[str] = None
    source_url: Optional[str] = None
    video_path: Optional[str] = None


# ===== Gemini API Schema (for structured output) =====

GEMINI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bowler_type": {"type": "string", "enum": ["pace", "spin", "unknown"]},
        "line": {"type": "string", "enum": ["outside_off", "off_stump", "middle", "leg", "outside_leg", "unknown"]},
        "length": {"type": "string", "enum": ["yorker", "full", "good", "short_of_length", "short", "unknown"]},
        "variation": {"type": "string", "enum": ["none", "slower", "cutter", "bouncer", "yorker", "spin_variation", "unknown"]},
        "shot_type": {"type": "string", "enum": ["drive", "cut", "pull", "hook", "defend", "sweep", "reverse_sweep", "glance", "flick", "lofted", "leave", "unknown"]},
        "footwork": {"type": "string", "enum": ["front_foot", "back_foot", "neutral", "unknown"]},
        "contact_quality": {"type": "string", "enum": ["clean", "mistimed", "edge", "miss", "unknown"]},
        "outcome": {"type": "string", "enum": ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "no_ball", "unknown"]},
        "bounce_behavior": {"type": "string", "enum": ["low", "normal", "steep", "unknown"]},
        "movement": {"type": "string", "enum": ["none", "seam", "swing", "turn", "unknown"]},
        "bowler_name": {"type": "string"},
        "batsman_name": {"type": "string"},
        "raw_description": {"type": "string", "description": "Free-form 1-2 sentence description of what happened"},
        "confidence": {
            "type": "object",
            "properties": {
                "bowler_type": {"type": "number"},
                "line": {"type": "number"},
                "length": {"type": "number"},
                "shot_type": {"type": "number"},
                "outcome": {"type": "number"},
                "contact_quality": {"type": "number"}
            },
            "required": ["bowler_type", "line", "length", "shot_type", "outcome", "contact_quality"]
        }
    },
    "required": ["bowler_type", "line", "length", "shot_type", "outcome", "confidence", "raw_description"]
}
