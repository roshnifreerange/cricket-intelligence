"""
Cricket Intelligence Engine - Video Ingestion Module
Handles downloading and storing cricket match videos.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.console import Console

console = Console()


class VideoIngestion:
    """Downloads and manages cricket match videos."""

    def __init__(self, output_dir: str = "data/raw_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_from_youtube(
        self,
        url: str,
        match_id: str,
        format_type: str = "T20",
        team_a: str = "",
        team_b: str = "",
        max_height: int = 720,
    ) -> Optional[dict]:
        """
        Download a cricket video from YouTube.

        Args:
            url: YouTube video URL
            match_id: Unique match identifier
            format_type: Cricket format (T20, ODI, Test)
            team_a: Team A name
            team_b: Team B name
            max_height: Maximum video height (720 for POC to save space)

        Returns:
            Metadata dict with video info, or None on failure
        """
        output_path = self.output_dir / f"{match_id}.mp4"

        console.print(f"[blue]⟳[/blue] Downloading: {url}")
        console.print(f"  Match ID: {match_id} | Format: {format_type}")

        try:
            # Use yt-dlp for robust YouTube downloading
            cmd = [
                "yt-dlp",
                "-f", f"best[height<={max_height}]",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                "--no-playlist",
                "--write-info-json",
                url,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                console.print(f"[red]✗[/red] Download failed: {result.stderr[:200]}")
                return None

            # Get video duration and metadata
            probe_cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(output_path),
            ]
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True
            )

            duration = "unknown"
            if probe_result.returncode == 0:
                probe_data = json.loads(probe_result.stdout)
                duration = probe_data.get("format", {}).get("duration", "unknown")

            metadata = {
                "match_id": match_id,
                "format": format_type,
                "team_a": team_a,
                "team_b": team_b,
                "source_url": url,
                "video_path": str(output_path),
                "duration_seconds": duration,
                "downloaded_at": datetime.now().isoformat(),
            }

            # Save metadata alongside video
            meta_path = self.output_dir / f"{match_id}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            console.print(f"[green]✓[/green] Downloaded: {output_path} ({duration}s)")
            return metadata

        except FileNotFoundError:
            console.print(
                "[red]✗[/red] yt-dlp not found. Install with: pip install yt-dlp"
            )
            return None
        except subprocess.TimeoutExpired:
            console.print("[red]✗[/red] Download timed out (10 min limit)")
            return None
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            return None

    def register_local_video(
        self,
        video_path: str,
        match_id: str,
        format_type: str = "T20",
        team_a: str = "",
        team_b: str = "",
    ) -> Optional[dict]:
        """
        Register a locally available video file.

        Args:
            video_path: Path to existing video file
            match_id: Unique match identifier
            format_type: Cricket format
            team_a, team_b: Team names

        Returns:
            Metadata dict
        """
        path = Path(video_path)
        if not path.exists():
            console.print(f"[red]✗[/red] Video not found: {video_path}")
            return None

        metadata = {
            "match_id": match_id,
            "format": format_type,
            "team_a": team_a,
            "team_b": team_b,
            "source_url": "local",
            "video_path": str(path.resolve()),
            "downloaded_at": datetime.now().isoformat(),
        }

        meta_path = self.output_dir / f"{match_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[green]✓[/green] Registered: {video_path}")
        return metadata

    def list_videos(self) -> list[dict]:
        """List all registered/downloaded videos."""
        meta_files = sorted(self.output_dir.glob("*_meta.json"))
        videos = []
        for mf in meta_files:
            with open(mf) as f:
                videos.append(json.load(f))
        return videos


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download cricket match videos")
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--match-id", type=str, required=True, help="Match identifier")
    parser.add_argument("--format", type=str, default="T20", choices=["T20", "ODI", "Test"])
    parser.add_argument("--team-a", type=str, default="")
    parser.add_argument("--team-b", type=str, default="")
    parser.add_argument("--local", type=str, help="Register a local video file instead")
    args = parser.parse_args()

    ingestion = VideoIngestion()

    if args.local:
        ingestion.register_local_video(
            args.local, args.match_id, args.format, args.team_a, args.team_b
        )
    elif args.url:
        ingestion.download_from_youtube(
            args.url, args.match_id, args.format, args.team_a, args.team_b
        )
    else:
        parser.print_help()
