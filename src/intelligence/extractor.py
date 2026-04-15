"""
Cricket Intelligence Engine - Gemini Vision Extractor
Sends ball clips to Gemini API and extracts structured cricket intelligence.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

from src.intelligence.schema import BallRecord, ConfidenceScores, GEMINI_JSON_SCHEMA
from src.intelligence.prompt import get_single_ball_prompt, get_cv_augmented_prompt, get_system_prompt, get_batch_prompt

load_dotenv()
console = Console()


class GeminiExtractor:
    """Extracts structured cricket intelligence from video clips using Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env file or environment.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        console.print(f"[green]✓[/green] Gemini extractor initialized with model: {model_name}")

    def extract_from_clip(
        self,
        clip_path: str,
        match_id: str = "unknown",
        over: int = 0,
        ball_number: int = 1,
        innings: int = 1,
        cv_context: dict | None = None,
    ) -> Optional[BallRecord]:
        """
        Analyze a single ball clip and return structured data.

        Args:
            clip_path:   Path to the video clip file
            match_id:    Match identifier
            over:        Over number
            ball_number: Ball number within the over
            innings:     Innings number
            cv_context:  Optional output from DualModelDetector.analyze_frame().
                         When provided:
                           - Uses CV-augmented prompt (tells Gemini about geometric facts)
                           - Overrides line/length with CV geometry if confidence ≥ 0.85
                           - Gemini focuses on shot_type, contact, movement, bowler_type

        Returns:
            BallRecord with extracted fields, or None on failure
        """
        clip_file = Path(clip_path)
        if not clip_file.exists():
            console.print(f"[red]✗[/red] Clip not found: {clip_path}")
            return None

        try:
            console.print(f"[blue]⟳[/blue] Analyzing: {clip_file.name}...")

            # Upload the video file to Gemini
            uploaded_file = self.client.files.upload(file=clip_path)

            # Wait for processing
            while uploaded_file.state == "PROCESSING":
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state == "FAILED":
                console.print(f"[red]✗[/red] Video processing failed for {clip_file.name}")
                return None

            # Choose prompt: CV-augmented if we have detection context, plain otherwise
            if cv_context:
                prompt_text = get_cv_augmented_prompt(cv_context)
                console.print("  [cyan]↳ Using CV-augmented prompt (Roboflow context injected)[/cyan]")
            else:
                prompt_text = get_single_ball_prompt()

            # Call Gemini with structured output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            types.Part.from_text(text=prompt_text),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=get_system_prompt(),
                    response_mime_type="application/json",
                    response_schema=GEMINI_JSON_SCHEMA,
                    temperature=0.2,
                ),
            )

            # Parse the response
            raw_json = json.loads(response.text)

            # Build BallRecord from Gemini's response
            ball_id = f"{match_id}_{over}_{ball_number}"
            record = BallRecord(
                ball_id=ball_id,
                match_id=match_id,
                innings=innings,
                over=over,
                ball_number=ball_number,
                bowler_type=raw_json.get("bowler_type", "unknown"),
                line=raw_json.get("line", "unknown"),
                length=raw_json.get("length", "unknown"),
                variation=raw_json.get("variation", "none"),
                shot_type=raw_json.get("shot_type", "unknown"),
                footwork=raw_json.get("footwork", "unknown"),
                contact_quality=raw_json.get("contact_quality", "unknown"),
                outcome=raw_json.get("outcome", "unknown"),
                bounce_behavior=raw_json.get("bounce_behavior", "unknown"),
                movement=raw_json.get("movement", "unknown"),
                bowler_name=raw_json.get("bowler_name"),
                batsman_name=raw_json.get("batsman_name"),
                raw_description=raw_json.get("raw_description", ""),
                clip_path=str(clip_path),
                confidence=ConfidenceScores(**raw_json.get("confidence", {})),
            )

            # ── CV Override: if YOLO geometry is high-confidence, trust it over Gemini ──
            # Gemini can hallucinate line/length; stumps geometry is deterministic.
            if cv_context:
                geo = cv_context.get("geometry")
                if geo:
                    CV_LINE_CONF_THRESHOLD = 0.85   # override when stumps clearly visible
                    cv_line   = geo.get("line")
                    cv_length = geo.get("length")

                    if cv_line and cv_line not in ("unknown", None):
                        record.line = cv_line
                        record.confidence.line = max(
                            record.confidence.line, CV_LINE_CONF_THRESHOLD
                        )
                        console.print(
                            f"  [cyan]↳ CV override: line={cv_line} "
                            f"(conf → {record.confidence.line:.2f})[/cyan]"
                        )

                    if cv_length and cv_length not in ("unknown", None):
                        record.length = cv_length
                        record.confidence.length = max(
                            record.confidence.length, CV_LINE_CONF_THRESHOLD
                        )
                        console.print(
                            f"  [cyan]↳ CV override: length={cv_length} "
                            f"(conf → {record.confidence.length:.2f})[/cyan]"
                        )

            # Log result
            avg_confidence = (
                record.confidence.line
                + record.confidence.length
                + record.confidence.shot_type
            ) / 3
            color = "green" if avg_confidence > 0.7 else "yellow" if avg_confidence > 0.4 else "red"
            console.print(
                f"[{color}]✓[/{color}] Ball {ball_id}: "
                f"{record.bowler_type.value} | {record.line.value} | "
                f"{record.length.value} | {record.shot_type.value} → "
                f"{record.outcome.value} (confidence: {avg_confidence:.2f})"
            )

            # Clean up uploaded file
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass  # Non-critical cleanup

            return record

        except Exception as e:
            console.print(f"[red]✗[/red] Error analyzing {clip_file.name}: {e}")
            return None

    def extract_batch(
        self,
        clips_dir: str,
        match_id: str = "unknown",
        innings: int = 1,
        start_over: int = 1,
        cv_contexts: dict[str, dict | None] | None = None,
    ) -> list[BallRecord]:
        """
        Process all clips in a directory.

        Args:
            clips_dir:   Directory containing ball clip videos
            match_id:    Match identifier
            innings:     Innings number
            start_over:  Starting over number for ball numbering
            cv_contexts: Optional dict mapping clip_path → DualModelDetector result.
                         Built by run_pipeline.py in Step 2.5.
                         When present, each clip gets its CV context forwarded into
                         extract_from_clip for prompt augmentation + line/length override.

        Returns:
            List of BallRecord objects
        """
        clips_path = Path(clips_dir)
        clip_files = sorted(clips_path.glob("*.mp4")) + sorted(clips_path.glob("*.webm"))

        if not clip_files:
            console.print(f"[red]✗[/red] No video clips found in {clips_dir}")
            return []

        cv_contexts = cv_contexts or {}
        cv_enabled  = bool(cv_contexts)
        console.print(
            f"\n[bold]Processing {len(clip_files)} clips from {clips_dir}[/bold]"
            + (f"  [cyan](CV-augmented)[/cyan]" if cv_enabled else "")
        )

        records = []
        for i, clip_file in enumerate(track(clip_files, description="Extracting...")):
            over = start_over + (i // 6)   # 6 balls per over
            ball = (i % 6) + 1

            # Look up CV context by absolute path (how run_pipeline builds the dict)
            cv_ctx = cv_contexts.get(str(clip_file)) or cv_contexts.get(clip_file.name)

            record = self.extract_from_clip(
                clip_path=str(clip_file),
                match_id=match_id,
                over=over,
                ball_number=ball,
                innings=innings,
                cv_context=cv_ctx,         # None if no CV data for this clip
            )

            if record:
                records.append(record)

            # Rate limiting — be kind to the API
            time.sleep(1)

        console.print(f"\n[bold green]✓ Extracted {len(records)}/{len(clip_files)} balls[/bold green]")
        return records

    def extract_from_video(
        self,
        video_path: str,
        match_id: str = "unknown",
        innings: int = 1,
        start_over: int = 1,
    ) -> list[BallRecord]:
        """
        Send a full video to Gemini and let it auto-detect all ball deliveries.

        Gemini watches the entire video, identifies each delivery, and returns
        a JSON array — one object per ball. No ffmpeg segmentation required.

        Args:
            video_path: Path to the full match/highlight video
            match_id:   Match identifier
            innings:    Innings number
            start_over: Starting over number for ball numbering

        Returns:
            List of BallRecord objects, one per detected delivery
        """
        video_file = Path(video_path)
        if not video_file.exists():
            console.print(f"[red]✗[/red] Video not found: {video_path}")
            return []

        console.print(f"[blue]⟳[/blue] Uploading video to Gemini: {video_file.name}")
        console.print("  [dim]Gemini will auto-detect ball deliveries — no pre-segmentation[/dim]")

        try:
            uploaded_file = self.client.files.upload(file=video_path)

            while uploaded_file.state == "PROCESSING":
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state == "FAILED":
                console.print(f"[red]✗[/red] Video processing failed")
                return []

            batch_schema = {
                "type": "array",
                "items": GEMINI_JSON_SCHEMA,
            }

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            types.Part.from_text(text=get_batch_prompt()),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=get_system_prompt(),
                    response_mime_type="application/json",
                    response_schema=batch_schema,
                    temperature=0.2,
                ),
            )

            raw_list = json.loads(response.text)
            if not isinstance(raw_list, list):
                raw_list = [raw_list]

            console.print(f"[green]✓[/green] Gemini detected [bold]{len(raw_list)}[/bold] ball deliveries")

            records = []
            for i, raw_json in enumerate(raw_list):
                over = start_over + (i // 6)
                ball = (i % 6) + 1
                ball_id = f"{match_id}_{over}_{ball}"

                record = BallRecord(
                    ball_id=ball_id,
                    match_id=match_id,
                    innings=innings,
                    over=over,
                    ball_number=ball,
                    bowler_type=raw_json.get("bowler_type", "unknown"),
                    line=raw_json.get("line", "unknown"),
                    length=raw_json.get("length", "unknown"),
                    variation=raw_json.get("variation", "none"),
                    shot_type=raw_json.get("shot_type", "unknown"),
                    footwork=raw_json.get("footwork", "unknown"),
                    contact_quality=raw_json.get("contact_quality", "unknown"),
                    outcome=raw_json.get("outcome", "unknown"),
                    bounce_behavior=raw_json.get("bounce_behavior", "unknown"),
                    movement=raw_json.get("movement", "unknown"),
                    bowler_name=raw_json.get("bowler_name"),
                    batsman_name=raw_json.get("batsman_name"),
                    raw_description=raw_json.get("raw_description", ""),
                    clip_path=str(video_path),
                    confidence=ConfidenceScores(**raw_json.get("confidence", {})),
                )
                records.append(record)

                conf = (record.confidence.line + record.confidence.length + record.confidence.shot_type) / 3
                color = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "red"
                console.print(
                    f"  [{color}]Ball {ball_id}[/{color}]: "
                    f"{record.bowler_type.value} | {record.line.value} | "
                    f"{record.length.value} | {record.shot_type.value} → "
                    f"{record.outcome.value} (conf: {conf:.2f})"
                )

            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            return records

        except Exception as e:
            console.print(f"[red]✗[/red] Batch video analysis failed: {e}")
            return []

    def export_to_json(self, records: list[BallRecord], output_path: str) -> None:
        """Export ball records to a JSON file."""
        data = [record.model_dump(mode="json") for record in records]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"[green]✓[/green] Exported {len(records)} records to {output_path}")


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract cricket intelligence from ball clips")
    parser.add_argument("--clip", type=str, help="Path to a single ball clip")
    parser.add_argument("--dir", type=str, help="Directory of ball clips")
    parser.add_argument("--match-id", type=str, default="test_match_001")
    parser.add_argument("--output", type=str, default="data/extracted_balls.json")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    extractor = GeminiExtractor(model_name=args.model)

    if args.clip:
        record = extractor.extract_from_clip(args.clip, match_id=args.match_id)
        if record:
            extractor.export_to_json([record], args.output)
    elif args.dir:
        records = extractor.extract_batch(args.dir, match_id=args.match_id)
        extractor.export_to_json(records, args.output)
    else:
        parser.print_help()
