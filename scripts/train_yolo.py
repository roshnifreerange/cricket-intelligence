"""
Cricket Intelligence Engine
============================
Step 1: Download cricket dataset from Roboflow
Step 2: Train YOLOv8 (or YOLOv12) on cricket data
Step 3: Validate trained model on Cric-360 frames

Run:
    python scripts/train_yolo.py --step all
    python scripts/train_yolo.py --step download
    python scripts/train_yolo.py --step train
    python scripts/train_yolo.py --step validate
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ROBOFLOW_DIR = DATA_DIR / "roboflow_dataset"
CRIC360_DIR = DATA_DIR / "cric360"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "data" / "validation_results"

for d in [ROBOFLOW_DIR, CRIC360_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download Dataset from Roboflow
# ══════════════════════════════════════════════════════════════════════════════

def download_roboflow_dataset(api_key: str | None = None):
    """
    Download a cricket ball detection dataset from Roboflow Universe.

    Best datasets to use (pick ONE):
      1. daniyalworkpace/cricket-ball-detection-8uv1o  (most images)
      2. appmaker/cricket_ball_dataset                 (clean annotations)
      3. aakkoo/cricket-ball-hn0su                     (good diversity)

    Classes you'll get: ['ball'] or ['cricket ball']
    """
    console.print(Panel("📥 [bold]Step 1: Downloading Roboflow Dataset[/bold]", style="blue"))

    try:
        from roboflow import Roboflow
    except ImportError:
        console.print("[red]✗ roboflow not installed. Run: pip install roboflow[/red]")
        return None

    # Use env key or passed key
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        console.print("[yellow]⚠  No ROBOFLOW_API_KEY found.[/yellow]")
        console.print("   Get a free key at: https://app.roboflow.com → Settings → API Keys")
        console.print("   Then set: export ROBOFLOW_API_KEY=your_key_here")
        console.print()
        console.print("[bold]Or download manually:[/bold]")
        console.print("  1. Go to: https://universe.roboflow.com/daniyalworkpace/cricket-ball-detection-8uv1o")
        console.print("  2. Click 'Download Dataset'")
        console.print("  3. Choose format: [green]YOLOv8[/green]")
        console.print(f"  4. Unzip into: {ROBOFLOW_DIR}/")
        console.print("  5. Make sure data.yaml is at: data/roboflow_dataset/data.yaml")
        return None

    rf = Roboflow(api_key=api_key)

    # ── RECOMMENDED DATASET ──────────────────────────────────────────────────
    # Choose the one with the most relevant data for your use case:
    # Option A: Pure ball tracking (best for our project)
    project = rf.workspace("daniyalworkpace").project("cricket-ball-detection-8uv1o")

    # Option B: More diverse classes
    # project = rf.workspace("appmaker").project("cricket_ball_dataset")

    dataset = project.version(1).download(
        model_format="yolov8",
        location=str(ROBOFLOW_DIR),
        overwrite=True,
    )

    console.print(f"[green]✓ Dataset downloaded to: {ROBOFLOW_DIR}[/green]")

    # Check what classes are in the dataset
    yaml_path = ROBOFLOW_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        console.print(f"[cyan]  Classes found: {cfg.get('names', [])}[/cyan]")
        console.print(f"  Train images: {cfg.get('nc', '?')} classes")

    return str(yaml_path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train YOLO
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    dataset_yaml: str,
    model_variant: str = "yolov8n",   # yolov8n / yolov8s / yolov8m / yolov12n
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    device: str = "cpu",              # "cpu" | "0" (GPU) | "mps" (Apple Silicon)
) -> str:
    """
    Fine-tune YOLO on cricket ball dataset.

    Model size guide:
      yolov8n  → fastest, least accurate   (good for POC/testing)
      yolov8s  → balanced speed/accuracy   (recommended for POC)
      yolov8m  → better accuracy, slower   (use if GPU available)
      yolov12n → latest architecture       (requires: pip install ultralytics>=8.3)

    Device guide (Mac):
      'cpu'  → always works, slow
      'mps'  → Apple Silicon GPU, fast (M1/M2/M3/M4 Macs)
    """
    console.print(Panel("🏋️  [bold]Step 2: Training YOLO Model[/bold]", style="green"))

    try:
        from ultralytics import YOLO
    except ImportError:
        console.print("[red]✗ ultralytics not installed. Run: pip install ultralytics[/red]")
        return ""

    # Use MPS on Apple Silicon if available and not specified
    if device == "cpu":
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                console.print("[cyan]🍎 Apple Silicon detected → using MPS (GPU)[/cyan]")
        except Exception:
            pass

    console.print(f"  Model:   [bold]{model_variant}.pt[/bold]")
    console.print(f"  Dataset: {dataset_yaml}")
    console.print(f"  Epochs:  {epochs}")
    console.print(f"  Img size:{img_size}px")
    console.print(f"  Device:  {device}")
    console.print()

    model = YOLO(f"{model_variant}.pt")   # Downloads pretrained weights automatically

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(MODELS_DIR / "yolo_cricket"),
        name="train",
        exist_ok=True,
        patience=15,            # Early stopping patience
        save=True,
        plots=True,             # Saves training curves as images
        device=device,
        # Small-object tuning (cricket ball is tiny in broadcast frames)
        mosaic=1.0,
        mixup=0.1,
        degrees=5.0,            # Small rotations
        scale=0.5,
        fliplr=0.5,
    )

    best_weights = MODELS_DIR / "yolo_cricket" / "train" / "weights" / "best.pt"
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Best weights saved to: [cyan]{best_weights}[/cyan]")
    console.print(f"  Training plots at:     [cyan]{MODELS_DIR}/yolo_cricket/train/[/cyan]")

    return str(best_weights)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Download & Validate on Cric-360
# ══════════════════════════════════════════════════════════════════════════════

def download_cric360(num_images: int = 100):
    """
    Download Cric-360 frames from HuggingFace for validation.

    Cric-360 = 3,558 broadcast-quality cricket ground frames
    Source: https://huggingface.co/datasets/sarimshahzad/Cric-360

    NOTE: Cric-360 is a scene-understanding dataset (pitch segmentation,
    stadium geometry), NOT a ball bounding-box dataset.
    We use it to visually validate that our model can detect the ball
    in real broadcast frames (qualitative validation).
    """
    console.print(Panel("📡 [bold]Step 3a: Downloading Cric-360 Frames[/bold]", style="magenta"))

    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[yellow]⚠  datasets library not found.[/yellow]")
        console.print("   Install with: pip install datasets")
        console.print()
        console.print("[bold]Manual alternative:[/bold]")
        console.print("  1. Go to: https://huggingface.co/datasets/sarimshahzad/Cric-360")
        console.print("  2. Accept the dataset access conditions")
        console.print("  3. Download and place images into: data/cric360/images/")
        return False

    console.print(f"  Downloading {num_images} frames from Cric-360...")
    console.print("  (Dataset: sarimshahzad/Cric-360 on HuggingFace)\n")

    try:
        # Load just the test split for validation
        ds = load_dataset(
            "sarimshahzad/Cric-360",
            split=f"test[:{num_images}]",
            trust_remote_code=True,
        )

        images_dir = CRIC360_DIR / "images"
        images_dir.mkdir(exist_ok=True)

        saved = 0
        for i, sample in enumerate(ds):
            img = sample.get("image") or sample.get("img")
            if img is None:
                # Try to find image key
                img_key = next((k for k in sample if "image" in k.lower() or "img" in k.lower()), None)
                if img_key:
                    img = sample[img_key]

            if img is not None:
                out_path = images_dir / f"cric360_{i:04d}.jpg"
                if hasattr(img, "save"):          # PIL Image
                    img.save(out_path)
                elif isinstance(img, np.ndarray): # NumPy array
                    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                saved += 1

        console.print(f"[green]✓ Saved {saved} Cric-360 frames to: {images_dir}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Failed to download Cric-360: {e}[/red]")
        console.print("  Tip: You may need to log in with: huggingface-cli login")
        return False


def validate_on_cric360(weights_path: str, conf_threshold: float = 0.25):
    """
    Run our trained YOLO model on Cric-360 frames and report results.

    This is QUALITATIVE validation — Cric-360 doesn't have ball bounding-box
    labels, so we check:
      ✓ Does the model find the ball in real broadcast frames?
      ✓ What is the detection confidence distribution?
      ✓ How many frames have at least 1 ball detection?
    """
    console.print(Panel("🔍 [bold]Step 3b: Validating on Cric-360 Frames[/bold]", style="yellow"))

    images_dir = CRIC360_DIR / "images"
    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        console.print("[red]✗ No Cric-360 images found. Run --step download_cric360 first.[/red]")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        console.print("[red]✗ ultralytics not installed[/red]")
        return

    model = YOLO(weights_path)
    console.print(f"  Model: {weights_path}")
    console.print(f"  Conf threshold: {conf_threshold}")
    console.print()

    image_paths = sorted(images_dir.glob("*.jpg"))
    console.print(f"  Running on {len(image_paths)} frames...\n")

    results_log = []
    detected_count = 0
    annotated_dir = RESULTS_DIR / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    for img_path in image_paths:
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            verbose=False,
            save=False,
        )

        frame_result = {
            "image": img_path.name,
            "detections": [],
        }

        for r in results:
            for box in r.boxes:
                det = {
                    "class": r.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
                }
                frame_result["detections"].append(det)

        has_ball = any(
            "ball" in d["class"].lower()
            for d in frame_result["detections"]
        )
        if has_ball:
            detected_count += 1

        results_log.append(frame_result)

        # Save annotated image (first 20 only)
        if len(results_log) <= 20:
            frame = cv2.imread(str(img_path))
            for det in frame_result["detections"]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                conf = det["confidence"]
                label = f"{det['class']} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(annotated_dir / img_path.name), frame)

    # ── Save results JSON ────────────────────────────────────────────────────
    results_path = RESULTS_DIR / "cric360_validation.json"
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)

    # ── Print Summary ────────────────────────────────────────────────────────
    total = len(image_paths)
    detection_rate = detected_count / total * 100 if total else 0

    all_confs = [
        d["confidence"]
        for r in results_log
        for d in r["detections"]
        if "ball" in d["class"].lower()
    ]
    avg_conf = np.mean(all_confs) if all_confs else 0

    table = Table(title="Cric-360 Validation Summary", style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total frames tested", str(total))
    table.add_row("Frames with ball detected", f"{detected_count} ({detection_rate:.1f}%)")
    table.add_row("Total ball detections", str(len(all_confs)))
    table.add_row("Avg confidence (ball)", f"{avg_conf:.3f}")
    table.add_row("Results saved to", str(results_path))
    table.add_row("Annotated frames", str(annotated_dir))

    console.print(table)

    # ── Interpretation ───────────────────────────────────────────────────────
    console.print()
    if detection_rate >= 50:
        console.print("[bold green]✓ GOOD — model detects ball in majority of broadcast frames[/bold green]")
    elif detection_rate >= 25:
        console.print("[yellow]⚠ MODERATE — some broadcast frames missed. Consider more training data.[/yellow]")
    else:
        console.print("[red]✗ POOR — model struggles on real broadcast frames.[/red]")
        console.print("  Try: more epochs, larger model (yolov8s/m), or higher imgsz=1280")

    return results_log


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cricket YOLO Training + Cric-360 Validation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["download", "train", "download_cric360", "validate", "all"],
        default="all",
        help=(
            "download         → Download Roboflow dataset\n"
            "train            → Train YOLO on cricket data\n"
            "download_cric360 → Download Cric-360 validation frames\n"
            "validate         → Run model on Cric-360 frames\n"
            "all              → Run all steps in order"
        ),
    )
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--dataset", type=str, help="Path to data.yaml (skip download)")
    parser.add_argument("--weights", type=str, help="Path to .pt weights (skip training)")
    parser.add_argument("--model", type=str, default="yolov8n",
                        help="Model variant: yolov8n / yolov8s / yolov8m / yolov12n (default: yolov8n)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | mps (Apple Silicon) | 0 (CUDA GPU)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for validation (default: 0.25)")
    parser.add_argument("--cric360-frames", type=int, default=100,
                        help="Number of Cric-360 frames to download (default: 100)")
    args = parser.parse_args()

    dataset_yaml = args.dataset
    weights_path = args.weights

    console.print(Panel(
        "[bold cyan]🏏 Cricket Intelligence Engine[/bold cyan]\n"
        "YOLO Training + Cric-360 Validation Pipeline",
        style="cyan"
    ))

    # ── Step 1: Download Roboflow ──────────────────────────────────────────
    if args.step in ("download", "all"):
        result = download_roboflow_dataset(api_key=args.api_key)
        if result:
            dataset_yaml = result

    # ── Step 2: Train ──────────────────────────────────────────────────────
    if args.step in ("train", "all"):
        yaml_to_use = dataset_yaml or str(ROBOFLOW_DIR / "data.yaml")
        if not Path(yaml_to_use).exists():
            console.print(f"[red]✗ data.yaml not found at: {yaml_to_use}[/red]")
            console.print("  Run --step download first, or pass --dataset path/to/data.yaml")
        else:
            weights_path = train_model(
                dataset_yaml=yaml_to_use,
                model_variant=args.model,
                epochs=args.epochs,
                img_size=args.imgsz,
                batch=args.batch,
                device=args.device,
            )

    # ── Step 3a: Download Cric-360 ────────────────────────────────────────
    if args.step in ("download_cric360", "all"):
        download_cric360(num_images=args.cric360_frames)

    # ── Step 3b: Validate ─────────────────────────────────────────────────
    if args.step in ("validate", "all"):
        wts = weights_path or str(MODELS_DIR / "yolo_cricket" / "train" / "weights" / "best.pt")
        if not Path(wts).exists():
            console.print(f"[red]✗ Weights not found at: {wts}[/red]")
            console.print("  Run --step train first, or pass --weights path/to/best.pt")
        else:
            validate_on_cric360(weights_path=wts, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
