"""
Cricket Intelligence Engine - Validation & Normalization Layer
Ensures Gemini outputs conform to schema and normalizes fuzzy text to enums.
"""

import re
from typing import Optional

from rich.console import Console

from src.intelligence.schema import (
    BallRecord, Line, Length, ShotType, BowlerType,
    Footwork, ContactQuality, Outcome, Variation,
    BounceBehavior, Movement,
)

console = Console()


# ===== Normalization Maps =====
# Maps fuzzy/natural language terms to our strict enum values

LINE_NORMALIZATIONS = {
    # outside_off
    "outside off": Line.OUTSIDE_OFF,
    "wide outside off": Line.OUTSIDE_OFF,
    "just outside off": Line.OUTSIDE_OFF,
    "wide of off": Line.OUTSIDE_OFF,
    "outside off stump": Line.OUTSIDE_OFF,
    "5th stump": Line.OUTSIDE_OFF,
    "fourth stump": Line.OUTSIDE_OFF,
    # off_stump
    "off stump": Line.OFF_STUMP,
    "on off": Line.OFF_STUMP,
    "off": Line.OFF_STUMP,
    "on the off": Line.OFF_STUMP,
    "off stump line": Line.OFF_STUMP,
    # middle
    "middle": Line.MIDDLE,
    "middle stump": Line.MIDDLE,
    "on middle": Line.MIDDLE,
    "middle and off": Line.MIDDLE,
    "middle and leg": Line.MIDDLE,
    # leg
    "leg": Line.LEG,
    "leg stump": Line.LEG,
    "on leg": Line.LEG,
    "on the pads": Line.LEG,
    "on his pads": Line.LEG,
    "leg stump line": Line.LEG,
    # outside_leg
    "outside leg": Line.OUTSIDE_LEG,
    "down leg": Line.OUTSIDE_LEG,
    "down the leg side": Line.OUTSIDE_LEG,
    "wide down leg": Line.OUTSIDE_LEG,
}

LENGTH_NORMALIZATIONS = {
    # yorker
    "yorker": Length.YORKER,
    "yorker length": Length.YORKER,
    "full toss": Length.FULL,  # technically different but close enough for POC
    # full
    "full": Length.FULL,
    "full length": Length.FULL,
    "overpitched": Length.FULL,
    "fullish": Length.FULL,
    "half volley": Length.FULL,
    # good
    "good": Length.GOOD,
    "good length": Length.GOOD,
    "nagging length": Length.GOOD,
    "testing length": Length.GOOD,
    # short_of_length
    "short of length": Length.SHORT_OF_LENGTH,
    "short of a length": Length.SHORT_OF_LENGTH,
    "short-ish": Length.SHORT_OF_LENGTH,
    "shortish": Length.SHORT_OF_LENGTH,
    "back of a length": Length.SHORT_OF_LENGTH,
    "back of length": Length.SHORT_OF_LENGTH,
    # short
    "short": Length.SHORT,
    "short ball": Length.SHORT,
    "bouncer": Length.SHORT,
    "very short": Length.SHORT,
    "banged in short": Length.SHORT,
}

SHOT_NORMALIZATIONS = {
    "drive": ShotType.DRIVE,
    "cover drive": ShotType.DRIVE,
    "straight drive": ShotType.DRIVE,
    "on drive": ShotType.DRIVE,
    "off drive": ShotType.DRIVE,
    "cut": ShotType.CUT,
    "square cut": ShotType.CUT,
    "late cut": ShotType.CUT,
    "upper cut": ShotType.CUT,
    "pull": ShotType.PULL,
    "pull shot": ShotType.PULL,
    "hook": ShotType.HOOK,
    "hook shot": ShotType.HOOK,
    "defend": ShotType.DEFEND,
    "defensive": ShotType.DEFEND,
    "block": ShotType.DEFEND,
    "forward defense": ShotType.DEFEND,
    "forward defence": ShotType.DEFEND,
    "back foot defense": ShotType.DEFEND,
    "back foot defence": ShotType.DEFEND,
    "sweep": ShotType.SWEEP,
    "sweep shot": ShotType.SWEEP,
    "slog sweep": ShotType.SWEEP,
    "reverse sweep": ShotType.REVERSE_SWEEP,
    "reverse": ShotType.REVERSE_SWEEP,
    "glance": ShotType.GLANCE,
    "leg glance": ShotType.GLANCE,
    "flick": ShotType.FLICK,
    "wrist flick": ShotType.FLICK,
    "clip": ShotType.FLICK,
    "lofted": ShotType.LOFTED,
    "lofted shot": ShotType.LOFTED,
    "slog": ShotType.LOFTED,
    "big shot": ShotType.LOFTED,
    "aerial": ShotType.LOFTED,
    "leave": ShotType.LEAVE,
    "left alone": ShotType.LEAVE,
    "shouldered arms": ShotType.LEAVE,
    "no shot": ShotType.LEAVE,
}

OUTCOME_NORMALIZATIONS = {
    "dot": Outcome.DOT,
    "dot ball": Outcome.DOT,
    "no run": Outcome.DOT,
    "0": Outcome.DOT,
    "single": Outcome.ONE,
    "1 run": Outcome.ONE,
    "one": Outcome.ONE,
    "double": Outcome.TWO,
    "2 runs": Outcome.TWO,
    "two": Outcome.TWO,
    "three": Outcome.THREE,
    "3 runs": Outcome.THREE,
    "four": Outcome.FOUR,
    "boundary": Outcome.FOUR,
    "4 runs": Outcome.FOUR,
    "six": Outcome.SIX,
    "maximum": Outcome.SIX,
    "6 runs": Outcome.SIX,
    "over the rope": Outcome.SIX,
    "wicket": Outcome.WICKET,
    "out": Outcome.WICKET,
    "bowled": Outcome.WICKET,
    "caught": Outcome.WICKET,
    "lbw": Outcome.WICKET,
    "stumped": Outcome.WICKET,
    "run out": Outcome.WICKET,
    "wide": Outcome.WIDE,
    "wide ball": Outcome.WIDE,
    "no ball": Outcome.NO_BALL,
    "no-ball": Outcome.NO_BALL,
}


def normalize_field(value: str, normalization_map: dict, default=None):
    """
    Normalize a fuzzy text value to a strict enum value.

    Args:
        value: Raw text value from model output
        normalization_map: Dict mapping fuzzy text → enum value
        default: Value to return if no match found
    """
    if not value or value.lower() in ("unknown", ""):
        return default

    cleaned = value.lower().strip()

    # Direct match
    if cleaned in normalization_map:
        return normalization_map[cleaned]

    # Partial match (check if any key is contained in the value)
    for key, enum_val in normalization_map.items():
        if key in cleaned or cleaned in key:
            return enum_val

    return default


class BallRecordValidator:
    """Validates and normalizes ball records."""

    def validate_record(self, record: BallRecord) -> tuple[BallRecord, list[str]]:
        """
        Validate and normalize a BallRecord.

        Returns:
            Tuple of (normalized record, list of warnings)
        """
        warnings = []

        # Normalize fields if they contain fuzzy text
        if record.raw_description:
            description_lower = record.raw_description.lower()

            # Try to infer missing fields from raw description
            if record.line == Line.UNKNOWN:
                inferred_line = normalize_field(
                    record.raw_description, LINE_NORMALIZATIONS, Line.UNKNOWN
                )
                if inferred_line != Line.UNKNOWN:
                    record.line = inferred_line
                    warnings.append(f"Inferred line '{inferred_line.value}' from description")

            if record.length == Length.UNKNOWN:
                inferred_length = normalize_field(
                    record.raw_description, LENGTH_NORMALIZATIONS, Length.UNKNOWN
                )
                if inferred_length != Length.UNKNOWN:
                    record.length = inferred_length
                    warnings.append(f"Inferred length '{inferred_length.value}' from description")

        # Cross-field consistency checks
        if record.shot_type == ShotType.LEAVE and record.contact_quality != ContactQuality.MISS:
            record.contact_quality = ContactQuality.MISS
            warnings.append("Set contact_quality to 'miss' for leave")

        if record.outcome == Outcome.WICKET and record.contact_quality == ContactQuality.CLEAN:
            warnings.append("Wicket with clean contact — possible catch or run out")

        if record.outcome == Outcome.DOT:
            record.runs_scored = 0
        elif record.outcome in (Outcome.ONE, Outcome.TWO, Outcome.THREE, Outcome.FOUR, Outcome.SIX):
            record.runs_scored = int(record.outcome.value)

        # Flag low confidence records for human review
        avg_confidence = (
            record.confidence.line
            + record.confidence.length
            + record.confidence.shot_type
        ) / 3

        if avg_confidence < 0.5:
            warnings.append(f"Low average confidence ({avg_confidence:.2f}) — needs human review")

        # Count unknowns
        unknown_count = sum(1 for field in [
            record.line, record.length, record.shot_type,
            record.bowler_type, record.contact_quality
        ] if hasattr(field, 'value') and field.value == "unknown")

        if unknown_count >= 3:
            warnings.append(f"High unknown count ({unknown_count}/5) — poor extraction quality")

        return record, warnings

    def validate_batch(self, records: list[BallRecord]) -> tuple[list[BallRecord], dict]:
        """
        Validate a batch of records.

        Returns:
            Tuple of (validated records, summary stats)
        """
        validated = []
        all_warnings = []
        low_confidence_count = 0
        high_unknown_count = 0

        for record in records:
            validated_record, warnings = self.validate_record(record)
            validated.append(validated_record)
            all_warnings.extend(warnings)

            avg_conf = (
                record.confidence.line
                + record.confidence.length
                + record.confidence.shot_type
            ) / 3
            if avg_conf < 0.5:
                low_confidence_count += 1

            unknown_count = sum(1 for field in [
                record.line, record.length, record.shot_type,
                record.bowler_type, record.contact_quality
            ] if hasattr(field, 'value') and field.value == "unknown")
            if unknown_count >= 3:
                high_unknown_count += 1

        stats = {
            "total_records": len(records),
            "low_confidence": low_confidence_count,
            "high_unknowns": high_unknown_count,
            "warnings_count": len(all_warnings),
            "needs_review_pct": (
                low_confidence_count / len(records) * 100 if records else 0
            ),
        }

        console.print(f"\n[bold]Validation Summary:[/bold]")
        console.print(f"  Total records: {stats['total_records']}")
        console.print(f"  Low confidence: {stats['low_confidence']}")
        console.print(f"  High unknowns: {stats['high_unknowns']}")
        console.print(f"  Needs review: {stats['needs_review_pct']:.1f}%")

        return validated, stats
