"""
Cric-360 Validation Script
===========================
Validates the Roboflow model (cricket-oftm6/3) on real broadcast frames
from the Cric-360 dataset (HuggingFace: sarimshahzad/Cric-360).

What this does:
  1. Downloads N frames from Cric-360 (test split)
  2. Runs our Roboflow model on each frame
  3. Reports: detection rate, confidence distribution, per-class stats
  4. Saves annotated frames + stats plots

NOTE: Cric-360 is a scene-understanding dataset (ground segmentation,
stadium geometry). It does NOT have ball bounding-box labels.
Validation here is QUALITATIVE — we check:
  ✓ Does the model detect the ball in real broadcast frames?
  ✓ Are player class detections plausible?
  ✓ What are confidence distributions?

Run:
    python scripts/validate_cric360.py
    python scripts/validate_cric360.py --frames 200 --conf 0.35
    python scripts/validate_cric360.py --images-dir data/cric360/images  # if already downloaded
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.detect import CricketDetector

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
CRIC360_DIR   = ROOT / "data" / "cric360"
RESULTS_DIR   = ROOT / "data" / "validation_results"
ANNOTATED_DIR = RESULTS_DIR / "annotated_cric360"

for d in [CRIC360_DIR / "images", RESULTS_DIR, ANNOTATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Download Cric-360 frames
# ══════════════════════════════════════════════════════════════════════════════

def download_cric360_frames(num_frames: int = 100) -> Path:
    """
    Download broadcast frames from Cric-360 (HuggingFace).

    Dataset:  sarimshahzad/Cric-360
    Content:  3,558 HD broadcast cricket ground frames
    Split:    We use the test split (535 images)
    Purpose:  Real-world validation of our detection model
    """
    images_dir = CRIC360_DIR / "images"

    existing = list(images_dir.glob("*.jpg"))
    if existing:
        console.print(f"[cyan]ℹ[/cyan] Found {len(existing)} existing Cric-360 images in {images_dir}")
        console.print("  Skipping download. Delete the folder to re-download.")
        return images_dir

    console.print(Panel(
        f"📡 Downloading {num_frames} frames from Cric-360\n"
        "   Dataset: sarimshahzad/Cric-360 (HuggingFace)",
        style="blue"
    ))

    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]✗ 'datasets' library not installed.[/red]")
        console.print("  Install: pip install datasets")
        console.print()
        console.print("[bold]Manual download alternative:[/bold]")
        console.print("  1. Go to https://huggingface.co/datasets/sarimshahzad/Cric-360")
        console.print("  2. Accept dataset conditions (free)")
        console.print("  3. Download images and place in: data/cric360/images/")
        console.print("  4. Re-run this script with --images-dir data/cric360/images")
        sys.exit(1)

    try:
        console.print(f"  Loading test split ({num_frames} frames)...")
        ds = load_dataset(
            "sarimshahzad/Cric-360",
            split=f"test[:{num_frames}]",
            trust_remote_code=True,
        )

        saved = 0
        for i, sample in enumerate(track(ds, description="Saving frames...")):
            # Cric-360 may use 'image' or 'img' key
            img = None
            for key in ("image", "img", "frame"):
                if key in sample:
                    img = sample[key]
                    break

            if img is None:
                # Try any key that looks like an image
                for key, val in sample.items():
                    if hasattr(val, "save") or isinstance(val, np.ndarray):
                        img = val
                        break

            if img is None:
                continue

            out_path = images_dir / f"cric360_{i:04d}.jpg"
            if hasattr(img, "save"):                  # PIL Image
                img.save(out_path, quality=95)
            elif isinstance(img, np.ndarray):         # NumPy
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved += 1

        console.print(f"[green]✓[/green] Downloaded {saved} frames to: {images_dir}")

    except Exception as e:
        console.print(f"[red]✗ HuggingFace download failed: {e}[/red]")
        console.print()
        console.print("Possible fixes:")
        console.print("  • Log in:  huggingface-cli login")
        console.print("  • Install: pip install huggingface_hub")
        console.print("  • Download manually from https://huggingface.co/datasets/sarimshahzad/Cric-360")
        sys.exit(1)

    return images_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Run Model on Cric-360 frames
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(images_dir: Path, detector: CricketDetector, save_annotated_limit: int = 30) -> list[dict]:
    """Run detector on all images in the directory, save annotated samples."""
    image_paths = sorted(images_dir.glob("*.jpg"))

    if not image_paths:
        console.print(f"[red]✗ No .jpg images found in {images_dir}[/red]")
        sys.exit(1)

    console.print(Panel(
        f"🔍 Running detection on {len(image_paths)} Cric-360 frames\n"
        f"   Backend: {detector.backend} | Conf ≥ {detector.conf_threshold}",
        style="green"
    ))

    results = []

    for i, img_path in enumerate(track(image_paths, description="Detecting...")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        dets  = detector.detect_frame(frame)
        scene = detector.parse_scene(dets)

        record = {
            "image":      img_path.name,
            "detections": dets,
            "scene_has":  {
                k: (v is not None if k != "umpires" else len(v) > 0)
                for k, v in scene.items()
            },
        }
        results.append(record)

        # Save annotated frame (first N only to avoid huge disk usage)
        if i < save_annotated_limit and dets:
            annotated = detector._draw_detections(frame, dets)
            cv2.imwrite(str(ANNOTATED_DIR / img_path.name), annotated)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list[dict]) -> dict:
    """Compute and print validation metrics, return summary dict."""
    total = len(results)
    if total == 0:
        console.print("[red]No results to report.[/red]")
        return {}

    # Per-class detection counts
    class_frame_hits = defaultdict(int)  # frames where class was detected ≥1 time
    class_confs      = defaultdict(list)

    for r in results:
        seen_classes = set()
        for det in r["detections"]:
            cls  = det["class_name"].lower()
            conf = det["confidence"]
            class_confs[cls].append(conf)
            if cls not in seen_classes:
                class_frame_hits[cls] += 1
                seen_classes.add(cls)

    # Summary table
    table = Table(
        title=f"Cric-360 Validation — {total} frames",
        style="cyan",
        show_lines=True,
    )
    table.add_column("Class",           style="bold white")
    table.add_column("Frames Hit",      justify="right")
    table.add_column("Hit Rate",        justify="right")
    table.add_column("Avg Confidence",  justify="right")
    table.add_column("Min Conf",        justify="right")
    table.add_column("Max Conf",        justify="right")

    EXPECTED_CLASSES = ["ball", "batsman", "bowler", "wicketkeeper", "nonstriker", "umpire"]
    summary = {}

    for cls in EXPECTED_CLASSES:
        hits    = class_frame_hits.get(cls, 0)
        confs   = class_confs.get(cls, [])
        rate    = hits / total * 100
        avg_c   = np.mean(confs) if confs else 0
        min_c   = min(confs)     if confs else 0
        max_c   = max(confs)     if confs else 0

        # Color code
        if   rate >= 50:  rate_style = "[green]"
        elif rate >= 20:  rate_style = "[yellow]"
        else:             rate_style = "[red]"

        table.add_row(
            cls,
            str(hits),
            f"{rate_style}{rate:.1f}%[/{rate_style[1:]}",
            f"{avg_c:.3f}" if confs else "—",
            f"{min_c:.3f}" if confs else "—",
            f"{max_c:.3f}" if confs else "—",
        )

        summary[cls] = {
            "frames_hit": hits,
            "hit_rate_pct": round(rate, 2),
            "avg_confidence": round(avg_c, 3),
        }

    console.print()
    console.print(table)

    # Overall health check
    ball_rate = summary.get("ball", {}).get("hit_rate_pct", 0)
    console.print()
    if ball_rate >= 40:
        console.print(f"[bold green]✓ Ball detection: {ball_rate:.1f}% — GOOD for broadcast frames[/bold green]")
    elif ball_rate >= 15:
        console.print(f"[yellow]⚠ Ball detection: {ball_rate:.1f}% — MODERATE (ball is tiny in wide-angle shots)[/yellow]")
    else:
        console.print(f"[red]✗ Ball detection: {ball_rate:.1f}% — LOW[/red]")
        console.print("  Note: Cric-360 uses wide-angle/aerial views where ball may not be visible.")
        console.print("  This is expected — validate on close-up ball-clip videos for better numbers.")

    console.print(f"\n  Annotated samples: [cyan]{ANNOTATED_DIR}[/cyan]")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Save Results
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: list[dict], summary: dict):
    out = {
        "model":   "cricket-oftm6/3 (Roboflow)",
        "frames":  len(results),
        "summary": summary,
        "per_frame": results,
    }
    path = RESULTS_DIR / "cric360_validation.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[green]✓[/green] Full results saved to: [cyan]{path}[/cyan]")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate Roboflow cricket model on Cric-360 broadcast frames"
    )
    parser.add_argument("--frames",     type=int,   default=100,
                        help="Number of Cric-360 frames to download (default: 100)")
    parser.add_argument("--conf",       type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--images-dir", type=str,   default=None,
                        help="Use existing images dir instead of downloading")
    parser.add_argument("--api-key",    type=str,   default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🏏 Cricket Intelligence Engine[/bold cyan]\n"
        "Cric-360 Model Validation Pipeline",
        style="cyan"
    ))

    # 1. Get images
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = download_cric360_frames(num_frames=args.frames)

    # 2. Init detector
    detector = CricketDetector(
        api_key=args.api_key,
        conf_threshold=args.conf,
    )

    # 3. Run detection
    results = run_validation(images_dir, detector)

    # 4. Report
    summary = generate_report(results)

    # 5. Save
    save_results(results, summary)


if __name__ == "__main__":
    main()
