"""
Cricket Intelligence Engine - Ball Clip Segmentation
Splits match videos into per-ball clips for analysis.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


class ClipExtractor:
    """Extracts individual ball clips from match videos."""

    def __init__(self, output_dir: str = "data/ball_clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_clip(
        self,
        video_path: str,
        start_time: str,
        end_time: str,
        match_id: str,
        over: int,
        ball: int,
        innings: int = 1,
    ) -> Optional[str]:
        """
        Extract a single ball clip from a video.

        Args:
            video_path: Path to source video
            start_time: Start timestamp (HH:MM:SS or seconds)
            end_time: End timestamp
            match_id: Match identifier
            over: Over number
            ball: Ball number
            innings: Innings number

        Returns:
            Path to the extracted clip, or None on failure
        """
        clip_name = f"{match_id}_inn{innings}_ov{over}_b{ball}.mp4"
        clip_path = self.output_dir / match_id / clip_name
        clip_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-crf", "23",
                str(clip_path),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0 and clip_path.exists():
                console.print(
                    f"[green]✓[/green] Clipped: Over {over}.{ball} "
                    f"({start_time} → {end_time}) → {clip_name}"
                )
                return str(clip_path)
            else:
                console.print(f"[red]✗[/red] Failed to clip: {result.stderr[:100]}")
                return None

        except FileNotFoundError:
            console.print("[red]✗[/red] ffmpeg not found. Install with: brew install ffmpeg")
            return None
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            return None

    def extract_from_timestamps(
        self,
        video_path: str,
        timestamps_file: str,
        match_id: str,
    ) -> list[dict]:
        """
        Extract multiple clips using a timestamps JSON file.

        Timestamps file format:
        [
            {"over": 1, "ball": 1, "start": "00:01:23", "end": "00:01:31", "innings": 1},
            {"over": 1, "ball": 2, "start": "00:01:45", "end": "00:01:53", "innings": 1},
            ...
        ]
        """
        with open(timestamps_file) as f:
            timestamps = json.load(f)

        console.print(
            f"\n[bold]Extracting {len(timestamps)} clips from {video_path}[/bold]\n"
        )

        results = []
        for ts in timestamps:
            clip_path = self.extract_clip(
                video_path=video_path,
                start_time=ts["start"],
                end_time=ts["end"],
                match_id=match_id,
                over=ts["over"],
                ball=ts["ball"],
                innings=ts.get("innings", 1),
            )
            results.append({
                **ts,
                "clip_path": clip_path,
                "match_id": match_id,
                "ball_id": f"{match_id}_{ts['over']}_{ts['ball']}",
            })

        successful = sum(1 for r in results if r["clip_path"])
        console.print(
            f"\n[bold green]✓ Extracted {successful}/{len(timestamps)} clips[/bold green]"
        )
        return results

    def extract_uniform_segments(
        self,
        video_path: str,
        match_id: str,
        segment_duration: float = 8.0,
        start_offset: float = 0.0,
        max_clips: int = 50,
        innings: int = 1,
    ) -> list[dict]:
        """
        Split video into uniform-length segments (quick & dirty for POC).
        Useful when you don't have exact ball timestamps.

        Args:
            video_path: Path to source video
            match_id: Match identifier
            segment_duration: Duration of each clip in seconds
            start_offset: Where to start splitting from
            max_clips: Maximum number of clips to extract
            innings: Innings number
        """
        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        total_duration = float(
            json.loads(probe_result.stdout).get("format", {}).get("duration", 0)
        )

        if total_duration == 0:
            console.print("[red]✗[/red] Could not determine video duration")
            return []

        console.print(
            f"\n[bold]Splitting {total_duration:.0f}s video into "
            f"{segment_duration}s segments[/bold]\n"
        )

        results = []
        current_time = start_offset
        clip_num = 0

        while current_time + segment_duration <= total_duration and clip_num < max_clips:
            over = (clip_num // 6) + 1
            ball = (clip_num % 6) + 1

            clip_path = self.extract_clip(
                video_path=video_path,
                start_time=str(current_time),
                end_time=str(current_time + segment_duration),
                match_id=match_id,
                over=over,
                ball=ball,
                innings=innings,
            )

            results.append({
                "over": over,
                "ball": ball,
                "start": current_time,
                "end": current_time + segment_duration,
                "clip_path": clip_path,
                "match_id": match_id,
            })

            current_time += segment_duration
            clip_num += 1

        return results

    def list_clips(self, match_id: str = None) -> list[Path]:
        """List all extracted clips, optionally filtered by match."""
        if match_id:
            search_dir = self.output_dir / match_id
        else:
            search_dir = self.output_dir

        return sorted(search_dir.rglob("*.mp4"))


# ===== Timestamp Template Generator =====
def generate_timestamp_template(
    match_id: str,
    num_overs: int = 3,
    output_path: str = None,
) -> str:
    """
    Generate a blank timestamps JSON file for manual filling.
    This helps the user manually mark ball delivery times.
    """
    timestamps = []
    for over in range(1, num_overs + 1):
        for ball in range(1, 7):
            timestamps.append({
                "over": over,
                "ball": ball,
                "start": "00:00:00",
                "end": "00:00:08",
                "innings": 1,
                "notes": f"Over {over}, Ball {ball} — fill in timestamps",
            })

    if not output_path:
        output_path = f"data/{match_id}_timestamps.json"

    with open(output_path, "w") as f:
        json.dump(timestamps, f, indent=2)

    console.print(f"[green]✓[/green] Template saved: {output_path}")
    console.print(f"  Fill in start/end times for {len(timestamps)} deliveries")
    return output_path


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ball clips from match videos")
    parser.add_argument("--video", type=str, help="Path to match video")
    parser.add_argument("--match-id", type=str, required=True)
    parser.add_argument("--timestamps", type=str, help="Path to timestamps JSON")
    parser.add_argument(
        "--uniform", action="store_true",
        help="Split into uniform segments (no timestamps needed)"
    )
    parser.add_argument("--segment-duration", type=float, default=8.0)
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument(
        "--template", action="store_true",
        help="Generate a blank timestamps template"
    )
    parser.add_argument("--overs", type=int, default=3, help="Number of overs for template")
    args = parser.parse_args()

    extractor = ClipExtractor()

    if args.template:
        generate_timestamp_template(args.match_id, args.overs)
    elif args.timestamps and args.video:
        extractor.extract_from_timestamps(args.video, args.timestamps, args.match_id)
    elif args.uniform and args.video:
        extractor.extract_uniform_segments(
            args.video, args.match_id,
            segment_duration=args.segment_duration,
            max_clips=args.max_clips,
        )
    else:
        parser.print_help()
