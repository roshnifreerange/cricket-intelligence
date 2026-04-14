"""
Cricket Intelligence Engine - Main Pipeline Runner
Orchestrates the full pipeline: ingest → segment → extract → validate → store.
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_full_pipeline(
    video_path: str = None,
    youtube_url: str = None,
    match_id: str = "test_match_001",
    format_type: str = "T20",
    team_a: str = "Team A",
    team_b: str = "Team B",
    timestamps_file: str = None,
    use_uniform_split: bool = False,
    segment_duration: float = 8.0,
    max_clips: int = 30,
    gemini_model: str = "gemini-2.5-flash",
    skip_extraction: bool = False,
    use_cv_detection: bool = True,       # NEW: enable Roboflow CV pre-analysis
    cv_frame_offset: float = 0.4,        # NEW: sample frame at 40% into each clip
):
    """Run the complete cricket intelligence pipeline."""

    console.print(Panel.fit(
        "[bold cyan]🏏 Cricket Intelligence Engine[/bold cyan]\n"
        "Ball-Level Video Understanding Pipeline",
        border_style="cyan",
    ))

    # ===== Step 1: Video Ingestion =====
    console.print("\n[bold]Step 1: Video Ingestion[/bold]")

    from src.ingestion.downloader import VideoIngestion
    ingestion = VideoIngestion()

    if youtube_url:
        metadata = ingestion.download_from_youtube(
            youtube_url, match_id, format_type, team_a, team_b
        )
        if not metadata:
            console.print("[red]Failed to download video. Exiting.[/red]")
            return
        video_path = metadata["video_path"]
    elif video_path:
        metadata = ingestion.register_local_video(
            video_path, match_id, format_type, team_a, team_b
        )
    else:
        console.print("[red]Provide --video or --youtube-url[/red]")
        return

    # ===== Step 2: Ball Segmentation =====
    console.print("\n[bold]Step 2: Ball Clip Segmentation[/bold]")

    from src.segmentation.clip_extractor import ClipExtractor
    extractor = ClipExtractor()

    if timestamps_file:
        clips = extractor.extract_from_timestamps(video_path, timestamps_file, match_id)
    elif use_uniform_split:
        clips = extractor.extract_uniform_segments(
            video_path, match_id,
            segment_duration=segment_duration,
            max_clips=max_clips,
        )
    else:
        console.print(
            "[yellow]⚠ No timestamps provided.[/yellow]\n"
            "Use --timestamps FILE or --uniform to split the video.\n"
            "Generate a template with: python -m src.segmentation.clip_extractor "
            f"--template --match-id {match_id}"
        )
        return

    clip_paths = [c["clip_path"] for c in clips if c.get("clip_path")]
    console.print(f"[green]✓ {len(clip_paths)} clips ready[/green]")

    if skip_extraction:
        console.print("[yellow]Skipping Gemini extraction (--skip-extraction)[/yellow]")
        return

    # ===== Step 2.5: CV Frame Analysis (Roboflow) =====
    # Sample a key frame from each clip and run DualModelDetector.
    # This gives us geometric line/length from stumps + scene context.
    # Passed into Gemini as grounding — dramatically improves line/length accuracy.
    cv_contexts: dict[str, dict | None] = {}   # clip_path → cv_context dict

    if use_cv_detection:
        console.print("\n[bold]Step 2.5: CV Frame Analysis (Roboflow)[/bold]")
        try:
            import cv2 as _cv2
            from src.detection.detect import DualModelDetector
            dual_detector = DualModelDetector()

            for clip_path in clip_paths:
                try:
                    cap = _cv2.VideoCapture(clip_path)
                    total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                    # Sample at cv_frame_offset into the clip (default 40%)
                    # This typically captures the ball in flight / at pitch
                    sample_frame_idx = max(0, int(total_frames * cv_frame_offset))
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
                    ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None:
                        cv_result = dual_detector.analyze_frame(frame)
                        cv_contexts[clip_path] = cv_result
                        geo = cv_result.get("geometry")
                        if geo:
                            console.print(
                                f"  [green]✓[/green] {Path(clip_path).name}: "
                                f"line=[cyan]{geo['line']}[/cyan] "
                                f"length=[cyan]{geo['length']}[/cyan]"
                            )
                        else:
                            console.print(
                                f"  [yellow]⚠[/yellow] {Path(clip_path).name}: "
                                "stumps/ball not detected — Gemini will estimate geometry"
                            )
                            cv_contexts[clip_path] = None
                    else:
                        cv_contexts[clip_path] = None

                except Exception as e:
                    console.print(f"  [yellow]⚠[/yellow] CV failed for {clip_path}: {e}")
                    cv_contexts[clip_path] = None

            detected = sum(1 for v in cv_contexts.values() if v and v.get("geometry"))
            console.print(
                f"[green]✓ CV geometry computed for {detected}/{len(clip_paths)} clips[/green]"
            )

        except ImportError as e:
            console.print(f"[yellow]⚠ CV detection skipped — missing dependency: {e}[/yellow]")
            console.print("  Install with: pip install inference-sdk opencv-python")
            use_cv_detection = False
        except ValueError as e:
            console.print(f"[yellow]⚠ CV detection skipped: {e}[/yellow]")
            console.print("  Set ROBOFLOW_API_KEY in .env to enable pre-analysis")
            use_cv_detection = False
    else:
        console.print("\n[dim]Step 2.5: CV detection disabled (--no-cv)[/dim]")

    # ===== Step 3: Gemini Intelligence Extraction =====
    console.print("\n[bold]Step 3: Gemini Intelligence Extraction[/bold]")
    if use_cv_detection:
        console.print("  [cyan]Mode: CV-augmented (Roboflow geometry → Gemini)[/cyan]")
    else:
        console.print("  [dim]Mode: Gemini-only (no CV pre-analysis)[/dim]")

    from src.intelligence.extractor import GeminiExtractor
    gemini = GeminiExtractor(model_name=gemini_model)

    clips_dir = f"data/ball_clips/{match_id}"
    records = gemini.extract_batch(
        clips_dir,
        match_id=match_id,
        cv_contexts=cv_contexts if use_cv_detection else {},
    )

    # ===== Step 4: Validation =====
    console.print("\n[bold]Step 4: Validation & Normalization[/bold]")

    from src.validation.normalizer import BallRecordValidator
    validator = BallRecordValidator()
    validated_records, val_stats = validator.validate_batch(records)

    # ===== Step 5: Storage =====
    console.print("\n[bold]Step 5: Database Storage[/bold]")

    from src.storage.db import CricketDB
    db = CricketDB()

    db.create_match({
        "match_id": match_id,
        "format": format_type,
        "team_a": team_a,
        "team_b": team_b,
    })

    saved = db.save_balls_batch(validated_records)

    # ===== Step 6: Export =====
    output_json = f"data/{match_id}_extracted.json"
    gemini.export_to_json(validated_records, output_json)

    # ===== Summary =====
    stats = db.get_stats(match_id)
    console.print(Panel.fit(
        f"[bold green]✅ Pipeline Complete![/bold green]\n\n"
        f"Match: {match_id} ({team_a} vs {team_b})\n"
        f"Balls processed: {stats['total']}\n"
        f"Avg confidence: {stats['avg_confidence']:.1%}\n"
        f"Needs review: {stats['total'] - stats['reviewed']}\n"
        f"JSON export: {output_json}\n\n"
        f"[cyan]Next steps:[/cyan]\n"
        f"  1. Review: streamlit run ui/app.py\n"
        f"  2. API: python -m src.api.main\n"
        f"  3. Track ball: python -m src.tracking.tracker --video <clip>",
        border_style="green",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🏏 Cricket Intelligence Engine - Full Pipeline"
    )
    parser.add_argument("--video", type=str, help="Local video file path")
    parser.add_argument("--youtube-url", type=str, help="YouTube URL")
    parser.add_argument("--match-id", type=str, default="test_match_001")
    parser.add_argument("--format", type=str, default="T20")
    parser.add_argument("--team-a", type=str, default="Team A")
    parser.add_argument("--team-b", type=str, default="Team B")
    parser.add_argument("--timestamps", type=str, help="Timestamps JSON file")
    parser.add_argument("--uniform", action="store_true", help="Uniform split")
    parser.add_argument("--segment-duration", type=float, default=8.0)
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--no-cv", action="store_true",
                        help="Disable Roboflow CV pre-analysis (Gemini-only mode)")
    parser.add_argument("--cv-frame-offset", type=float, default=0.4,
                        help="Fraction into each clip to sample for CV (default: 0.4)")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        youtube_url=args.youtube_url,
        match_id=args.match_id,
        format_type=args.format,
        team_a=args.team_a,
        team_b=args.team_b,
        timestamps_file=args.timestamps,
        use_uniform_split=args.uniform,
        segment_duration=args.segment_duration,
        max_clips=args.max_clips,
        gemini_model=args.model,
        skip_extraction=args.skip_extraction,
        use_cv_detection=not args.no_cv,
        cv_frame_offset=args.cv_frame_offset,
    )
