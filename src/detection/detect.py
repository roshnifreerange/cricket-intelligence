"""
Cricket Intelligence Engine — YOLO Detection
=============================================
Two Roboflow models work together to give full scene understanding:

  Model 1 — cricket-oftm6/3  (Scene: who is where)
    mAP@50: 96.2% | Classes: ball, batsman, bowler, wicketkeeper, nonstriker, umpire

  Model 2 — stumps/10  (Geometry: precise reference for line/length)
    Classes: ball, Batsman, Stumps
    Stumps bbox → off/leg stump positions → geometric line & length calculation

Use DualModelDetector to run both and get a fused result including
calculated line (outside_off / on_off / middle / leg / outside_leg)
and length (full_toss / yorker / full / good / short / bouncer).
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import cv2
import numpy as np
from rich.console import Console

load_dotenv()   # load ROBOFLOW_API_KEY and others from .env
console = Console()

# ── Model config ──────────────────────────────────────────────────────────────
ROBOFLOW_API_URL = "https://detect.roboflow.com"

# Model 1: Full scene — 6 classes
SCENE_MODEL_ID = "cricket-oftm6/3"

# Model 2: Geometry reference — stumps + ball + batsman
STUMPS_MODEL_ID = "stumps/10"

CRICKET_CLASSES = {
    1: "ball",
    2: "batsman",
    3: "bowler",
    4: "nonstriker",
    5: "umpire",
    6: "wicketkeeper",
    7: "stumps",
}

# ── Roboflow HTTP client (pure requests — no inference-sdk needed) ────────────
# inference-sdk requires Python <3.13; we are on Python 3.14.
# This drop-in replacement calls the same REST endpoint directly.

import requests as _requests

class _RoboflowHTTPClient:
    """
    Minimal Roboflow Serverless Inference client.
    Equivalent to InferenceHTTPClient from inference-sdk but works on Python 3.14+.
    Sends base64-encoded JPEG to the Roboflow serverless endpoint.
    """
    def __init__(self, api_url: str, api_key: str):
        self._base = api_url.rstrip("/")
        self._key  = api_key

    def infer(self, image_path: str, model_id: str) -> dict:
        """
        Run inference on a local image file.

        Args:
            image_path: Path to JPEG/PNG image.
            model_id:   Roboflow model ID, e.g. 'cricket-oftm6/3'.

        Returns:
            Parsed JSON response dict with 'predictions' list.
        """
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("utf-8")

        url = f"{self._base}/{model_id}"
        resp = _requests.post(
            url,
            params={"api_key": self._key},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


ROBOFLOW_SDK_AVAILABLE = True   # always True — no external SDK needed
InferenceHTTPClient = _RoboflowHTTPClient  # alias so rest of code is unchanged

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  Core Detector — Roboflow Cloud (primary) + Local YOLO (fallback)
# ══════════════════════════════════════════════════════════════════════════════

class CricketDetector:
    """
    Multi-backend cricket object detector.

    Priority:
      1. Roboflow Cloud API  (cricket-oftm6/3 — 96.2% mAP, 6 classes)
      2. Local YOLO weights  (fallback if SDK not installed or no API key)
    """

    def __init__(
        self,
        api_key: str | None = None,
        local_weights: str | None = None,
        conf_threshold: float = 0.4,
    ):
        """
        Args:
            api_key:        Roboflow API key. Falls back to ROBOFLOW_API_KEY env var.
            local_weights:  Path to local .pt weights (fallback only).
            conf_threshold: Minimum confidence to keep a detection.
        """
        self.conf_threshold = conf_threshold
        self.backend = None
        self._client = None
        self._local_model = None

        # Try Roboflow cloud first
        self._api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if ROBOFLOW_SDK_AVAILABLE and self._api_key:
            self._client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=self._api_key,
            )
            self.backend = "roboflow_cloud"
            console.print(
                f"[green]✓[/green] Using Roboflow cloud model "
                f"[cyan]{SCENE_MODEL_ID}[/cyan] (mAP 96.2%)"
            )

        # Fallback: local YOLO
        elif YOLO_AVAILABLE:
            weights = local_weights or "yolov8n.pt"
            self._local_model = YOLO(weights)
            self.backend = "local_yolo"
            if local_weights:
                console.print(f"[green]✓[/green] Using local YOLO weights: {local_weights}")
            else:
                console.print("[yellow]⚠[/yellow] Using base YOLOv8n (not cricket-specific)")
                console.print("  → Set ROBOFLOW_API_KEY to use the 96.2% mAP cloud model")

        else:
            raise RuntimeError(
                "No detection backend available.\n"
                "  Option A: pip install inference-sdk  (+ set ROBOFLOW_API_KEY)\n"
                "  Option B: pip install ultralytics"
            )

    # ── Single Frame ──────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detect cricket objects in a single OpenCV frame (BGR numpy array).

        Returns list of dicts:
          {class_name, class_id, confidence, bbox:[x1,y1,x2,y2], center:[cx,cy]}
        """
        if self.backend == "roboflow_cloud":
            return self._detect_frame_cloud(frame)
        else:
            return self._detect_frame_local(frame)

    def _detect_frame_cloud(self, frame: np.ndarray) -> list[dict]:
        """Call Roboflow serverless API with a single frame."""
        # Encode frame as JPEG to send over HTTP
        _, buf = cv2.imencode(".jpg", frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name

        try:
            raw = self._client.infer(tmp_path, model_id=SCENE_MODEL_ID)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return self._parse_roboflow_response(raw)

    def _detect_frame_local(self, frame: np.ndarray) -> list[dict]:
        """Run local YOLO model on a frame."""
        results = self._local_model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": r.names[int(box.cls[0])],
                    "class_id":   int(box.cls[0]),
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox":       [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center":     [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                })
        return detections

    def _parse_roboflow_response(self, raw: dict) -> list[dict]:
        """Normalize Roboflow API response to our internal format."""
        detections = []
        for pred in raw.get("predictions", []):
            conf = pred.get("confidence", 0)
            if conf < self.conf_threshold:
                continue

            # Roboflow gives center (x,y) + width/height → convert to xyxy
            cx, cy = pred["x"], pred["y"]
            w, h   = pred["width"], pred["height"]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2

            detections.append({
                "class_name":   pred["class"],
                "class_id":     pred["class_id"],
                "confidence":   round(conf, 3),
                "bbox":         [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center":       [round(cx, 1), round(cy, 1)],
                "detection_id": pred.get("detection_id", ""),
            })
        return detections

    # ── Structured Scene Parsing ──────────────────────────────────────────────

    def parse_scene(self, detections: list[dict]) -> dict:
        """
        Convert raw detections into a structured cricket scene dict.

        Returns:
          {
            ball:         {bbox, center, confidence} | None,
            batsman:      {bbox, center, confidence} | None,
            bowler:       {bbox, center, confidence} | None,
            wicketkeeper: {bbox, center, confidence} | None,
            nonstriker:   {bbox, center, confidence} | None,
            umpires:      [{bbox, center, confidence}, ...],
          }
        """
        scene = {
            "ball":         None,
            "batsman":      None,
            "bowler":       None,
            "wicketkeeper": None,
            "nonstriker":   None,
            "umpires":      [],
        }

        for det in detections:
            cls = det["class_name"].lower()
            info = {k: det[k] for k in ("bbox", "center", "confidence")}

            if cls == "ball":
                # Keep highest-confidence ball detection
                if scene["ball"] is None or det["confidence"] > scene["ball"]["confidence"]:
                    scene["ball"] = info
            elif cls == "batsman":
                if scene["batsman"] is None or det["confidence"] > scene["batsman"]["confidence"]:
                    scene["batsman"] = info
            elif cls == "bowler":
                if scene["bowler"] is None or det["confidence"] > scene["bowler"]["confidence"]:
                    scene["bowler"] = info
            elif cls == "wicketkeeper":
                if scene["wicketkeeper"] is None or det["confidence"] > scene["wicketkeeper"]["confidence"]:
                    scene["wicketkeeper"] = info
            elif cls == "nonstriker":
                if scene["nonstriker"] is None or det["confidence"] > scene["nonstriker"]["confidence"]:
                    scene["nonstriker"] = info
            elif cls == "umpire":
                scene["umpires"].append(info)

        return scene

    # ── Full Video Processing ─────────────────────────────────────────────────

    def detect_in_video(
        self,
        video_path: str,
        output_path: str | None = None,
        sample_every_n_frames: int = 1,
        save_annotated: bool = True,
    ) -> list[dict]:
        """
        Run detection across every N frames of a video.

        Args:
            video_path:           Path to input video
            output_path:          Where to save annotated video (optional)
            sample_every_n_frames: Process 1 of every N frames (use 3-5 to save API calls)
            save_annotated:       Whether to save annotated output video

        Returns:
            List of per-frame results: [{frame_idx, detections, scene}, ...]

        Note:
            If using cloud backend, sample_every_n_frames=3 is recommended
            to avoid excessive API calls (each frame = 1 API call).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS)
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        console.print(f"[blue]⟳[/blue] Processing: {video_path}")
        console.print(f"  {total_f} frames @ {fps:.1f}fps — sampling every {sample_every_n_frames} frame(s)")

        writer = None
        if save_annotated and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx   = 0
        ball_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n_frames == 0:
                detections = self.detect_frame(frame)
                scene      = self.parse_scene(detections)

                if scene["ball"] is not None:
                    ball_frames += 1

                all_results.append({
                    "frame_idx":  frame_idx,
                    "detections": detections,
                    "scene":      scene,
                })

                # Annotate frame
                if writer:
                    annotated = self._draw_detections(frame, detections)
                    writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        processed = len(all_results)
        console.print(
            f"[green]✓[/green] Processed {processed} frames — "
            f"ball detected in {ball_frames} ({ball_frames/max(processed,1)*100:.1f}%)"
        )
        if output_path and save_annotated:
            console.print(f"  Annotated video: {output_path}")

        return all_results

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes + labels on a frame."""
        CLASS_COLORS = {
            "ball":         (0, 255, 255),   # Yellow
            "batsman":      (0, 255, 0),     # Green
            "bowler":       (255, 128, 0),   # Orange
            "wicketkeeper": (255, 0, 255),   # Magenta
            "nonstriker":   (0, 128, 255),   # Light blue
            "umpire":       (128, 128, 128), # Grey
            "stumps":       (0, 200, 255),   # Cyan
        }

        out = frame.copy()
        for det in detections:
            cls   = det["class_name"].lower()
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf  = det["confidence"]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return out


# ── Backwards-compatible alias ────────────────────────────────────────────────
class CricketBallDetector(CricketDetector):
    """Alias for CricketDetector (backwards compatibility)."""
    def __init__(self, model_path: str | None = None, **kwargs):
        super().__init__(local_weights=model_path, **kwargs)

    def detect_in_frame(self, frame, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        return self.detect_frame(frame)


# ══════════════════════════════════════════════════════════════════════════════
#  Line & Length Geometry Calculator
# ══════════════════════════════════════════════════════════════════════════════

class LineLengthEstimator:
    """
    Calculates cricket line and length from stumps + ball pixel positions.

    How it works:
      - Stumps bbox gives us the wicket position in the image frame.
      - Real stumps are 9 inches (22.86cm) wide and 28 inches (71.1cm) tall.
      - We use the stumps pixel width as a scale reference.
      - Ball pixel position relative to stumps center → LINE
      - Ball pixel y position relative to stumps bottom → LENGTH
        (perspective: ball pitched close to camera appears lower in image)

    Coordinate note:
      - Image y increases downward (top=0, bottom=height)
      - Stumps bottom ≈ ground level / popping crease
      - Ball y > stumps_bottom → ball past popping crease (full/yorker)
      - Ball y < stumps_top    → ball short of good length (short/bouncer)
    """

    # Real-world stumps width in inches (used as pixel scale reference)
    STUMPS_REAL_WIDTH_IN = 9.0

    # Line thresholds (in units of stumps_width from center)
    # Positive = off side (assuming standard right-hand batsman view)
    LINE_THRESHOLDS = [
        (-2.5, "wide_leg"),
        (-1.5, "outside_leg"),
        (-0.5, "leg_stump"),
        ( 0.5, "middle"),
        ( 1.5, "off_stump"),
        ( 2.5, "outside_off"),
        ( float("inf"), "wide_outside_off"),
    ]

    # Length thresholds (in units of stumps_height from stumps bottom)
    # Negative = above stumps bottom (ball closer to bowler end)
    LENGTH_THRESHOLDS = [
        (-3.0, "bouncer"),
        (-1.5, "short"),
        (-0.5, "good_length"),
        ( 0.3, "full"),
        ( 0.8, "yorker"),
        ( float("inf"), "full_toss"),
    ]

    def estimate(self, ball: dict, stumps: dict) -> dict | None:
        """
        Estimate line and length from ball and stumps detections.

        Args:
            ball:   Detection dict with 'center' and 'bbox'
            stumps: Detection dict with 'center' and 'bbox'

        Returns:
            {
              'line':          str,   e.g. 'outside_off'
              'length':        str,   e.g. 'good_length'
              'ball_px':       [cx, cy],
              'stumps_center': [cx, cy],
              'stumps_size':   [w, h],
              'normalized_x':  float,  # signed, in stumps-widths from center
              'normalized_y':  float,  # signed, in stumps-heights from bottom
            }
            Returns None if ball or stumps is missing.
        """
        if ball is None or stumps is None:
            return None

        ball_cx, ball_cy     = ball["center"]
        stump_cx, stump_cy   = stumps["center"]
        x1, y1, x2, y2      = stumps["bbox"]
        stump_w = x2 - x1
        stump_h = y2 - y1
        stump_bottom = y2   # ground / popping crease level

        if stump_w <= 0 or stump_h <= 0:
            return None

        # Normalized horizontal offset: 0 = stump center, +ve = off side
        norm_x = (ball_cx - stump_cx) / stump_w

        # Normalized vertical offset from ground level:
        # negative means ball is above popping crease (short)
        # positive means ball is below (full / yorker territory)
        norm_y = (ball_cy - stump_bottom) / stump_h

        line   = self._classify(norm_x, self.LINE_THRESHOLDS)
        length = self._classify(norm_y, self.LENGTH_THRESHOLDS)

        return {
            "line":           line,
            "length":         length,
            "ball_px":        [ball_cx, ball_cy],
            "stumps_center":  [stump_cx, stump_cy],
            "stumps_size":    [round(stump_w, 1), round(stump_h, 1)],
            "normalized_x":   round(norm_x, 3),
            "normalized_y":   round(norm_y, 3),
        }

    @staticmethod
    def _classify(value: float, thresholds: list[tuple]) -> str:
        for threshold, label in thresholds:
            if value <= threshold:
                return label
        return thresholds[-1][1]


# ══════════════════════════════════════════════════════════════════════════════
#  Dual Model Detector — Combines both Roboflow models
# ══════════════════════════════════════════════════════════════════════════════

class DualModelDetector:
    """
    Runs BOTH Roboflow models on each frame and produces a fused result.

    Model 1 (cricket-oftm6/3): bowler, wicketkeeper, nonstriker, umpire, ball
    Model 2 (stumps/10):       stumps, ball, batsman  → geometric line/length

    Usage:
        detector = DualModelDetector()
        result   = detector.analyze_frame(frame)
        # result has: scene, geometry (line/length), all_detections
    """

    def __init__(
        self,
        api_key: str | None = None,
        conf_threshold: float = 0.4,
    ):
        self._api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.conf_threshold = conf_threshold
        self._estimator = LineLengthEstimator()

        if not ROBOFLOW_SDK_AVAILABLE:
            raise ImportError("inference-sdk not installed. Run: pip install inference-sdk")
        if not self._api_key:
            raise ValueError("ROBOFLOW_API_KEY not set.")

        self._client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=self._api_key,
        )
        console.print("[green]✓[/green] DualModelDetector ready")
        console.print(f"  Scene model:  [cyan]{SCENE_MODEL_ID}[/cyan]")
        console.print(f"  Stumps model: [cyan]{STUMPS_MODEL_ID}[/cyan]")

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Run both models on a single frame and return a fused analysis.

        Returns:
          {
            'scene':       { ball, batsman, bowler, wicketkeeper, nonstriker, stumps, umpires },
            'geometry':    { line, length, normalized_x, normalized_y, ... } | None,
            'scene_dets':  [...],   # raw detections from Model 1
            'stumps_dets': [...],   # raw detections from Model 2
          }
        """
        # Write frame to temp file once, reuse for both calls
        _, buf = cv2.imencode(".jpg", frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name

        try:
            raw_scene  = self._client.infer(tmp_path, model_id=SCENE_MODEL_ID)
            raw_stumps = self._client.infer(tmp_path, model_id=STUMPS_MODEL_ID)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Parse both
        scene_dets  = self._parse(raw_scene)
        stumps_dets = self._parse(raw_stumps)

        # Build fused scene:
        # - Player roles from Model 1
        # - Stumps from Model 2 (much more accurate for geometry)
        # - Ball: prefer Model 2 (stumps model trained specifically for ball+stump geometry)
        scene = self._fuse_scene(scene_dets, stumps_dets)

        # Geometry: line + length from stumps model
        stumps_scene = self._parse_stumps_scene(stumps_dets)
        geometry = self._estimator.estimate(
            ball=stumps_scene.get("ball"),
            stumps=stumps_scene.get("stumps"),
        )

        return {
            "scene":       scene,
            "geometry":    geometry,
            "scene_dets":  scene_dets,
            "stumps_dets": stumps_dets,
        }

    def _parse(self, raw: dict) -> list[dict]:
        """Parse Roboflow API response into internal format."""
        detections = []
        for pred in raw.get("predictions", []):
            conf = pred.get("confidence", 0)
            if conf < self.conf_threshold:
                continue
            cx, cy = pred["x"], pred["y"]
            w,  h  = pred["width"], pred["height"]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            detections.append({
                "class_name":   pred["class"].lower(),
                "class_id":     pred["class_id"],
                "confidence":   round(conf, 3),
                "bbox":         [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                "center":       [round(cx,1), round(cy,1)],
                "detection_id": pred.get("detection_id", ""),
            })
        return detections

    def _best(self, dets: list[dict], class_name: str) -> dict | None:
        """Return highest-confidence detection for a given class."""
        matches = [d for d in dets if d["class_name"] == class_name]
        return max(matches, key=lambda d: d["confidence"]) if matches else None

    def _parse_stumps_scene(self, dets: list[dict]) -> dict:
        return {
            "ball":    self._best(dets, "ball"),
            "stumps":  self._best(dets, "stumps"),
            "batsman": self._best(dets, "batsman"),
        }

    def _fuse_scene(self, scene_dets: list[dict], stumps_dets: list[dict]) -> dict:
        """Merge both model outputs into a unified scene dict."""
        info = lambda d: {k: d[k] for k in ("bbox", "center", "confidence")}

        # Player roles from Model 1
        scene = {
            "ball":         self._best(scene_dets, "ball"),
            "batsman":      self._best(scene_dets, "batsman"),
            "bowler":       self._best(scene_dets, "bowler"),
            "wicketkeeper": self._best(scene_dets, "wicketkeeper"),
            "nonstriker":   self._best(scene_dets, "nonstriker"),
            "umpires":      [info(d) for d in scene_dets if d["class_name"] == "umpire"],
            # Stumps exclusively from Model 2 (better trained for this)
            "stumps":       self._best(stumps_dets, "stumps"),
        }

        # Override ball with Model 2 if it's more confident
        ball_m2 = self._best(stumps_dets, "ball")
        if ball_m2 and (
            scene["ball"] is None
            or ball_m2["confidence"] > scene["ball"]["confidence"]
        ):
            scene["ball"] = ball_m2

        return scene

    def draw_fused(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw all detections + geometry overlay on frame.
        Green = scene model | Cyan = stumps model | Yellow label = line/length result
        """
        CLASS_COLORS = {
            "ball":         (0, 255, 255),
            "batsman":      (0, 255, 0),
            "bowler":       (255, 128, 0),
            "wicketkeeper": (255, 0, 255),
            "nonstriker":   (0, 128, 255),
            "umpire":       (128, 128, 128),
            "stumps":       (0, 200, 255),
        }
        out = frame.copy()

        # Draw scene detections (solid box)
        for det in result["scene_dets"]:
            cls   = det["class_name"]
            color = CLASS_COLORS.get(cls, (180, 180, 180))
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{cls} {det['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw stumps model (dashed-style — draw thinner in a different color)
        for det in result["stumps_dets"]:
            cls   = det["class_name"]
            color = CLASS_COLORS.get(cls, (180, 180, 180))
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)  # thinner

        # Draw geometry result
        geo = result.get("geometry")
        if geo:
            label = f"Line: {geo['line']}  |  Length: {geo['length']}"
            cv2.rectangle(out, (10, 10), (len(label) * 9 + 20, 38), (0, 0, 0), -1)
            cv2.putText(out, label, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        return out

    def analyze_video(
        self,
        video_path: str,
        output_path: str | None = None,
        sample_every_n_frames: int = 3,
    ) -> list[dict]:
        """
        Run both Roboflow models across every N frames of a video.

        Args:
            video_path:            Path to input video
            output_path:           Where to save annotated video (optional)
            sample_every_n_frames: Process 1 of every N frames.
                                   Default=3 saves ~66% of API calls.
                                   Use 1 for every frame, 5+ for long clips.

        Returns:
            List of per-frame results:
            [{
              'frame_idx':  int,
              'timestamp':  float,   # seconds into video
              'scene':      dict,    # fused scene from both models
              'geometry':   dict | None,  # line + length if ball+stumps visible
            }, ...]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_s   = total_f / fps

        console.print(f"[blue]⏳[/blue]  Video: {Path(video_path).name}")
        console.print(
            f"   {total_f} frames @ {fps:.1f}fps ({dur_s:.1f}s) — "
            f"analysing every {sample_every_n_frames} frame(s) — "
            f"~{total_f // sample_every_n_frames} API call pairs"
        )

        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx   = 0
        geo_frames  = 0   # frames where geometry was computed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            if frame_idx % sample_every_n_frames == 0:
                try:
                    result = self.analyze_frame(frame)
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Frame {frame_idx} failed: {e}")
                    frame_idx += 1
                    if writer:
                        writer.write(frame)   # write unannotated frame
                    continue

                geo = result.get("geometry")
                if geo:
                    geo_frames += 1

                all_results.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "scene":     result["scene"],
                    "geometry":  geo,
                })

                # Annotate and write frame
                if writer:
                    annotated = self.draw_fused(frame, result)
                    writer.write(annotated)
            else:
                if writer:
                    writer.write(frame)   # write unannotated frames as-is

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        processed = len(all_results)
        console.print(
            f"[green]✓[/green] Analysed {processed} frames — "
            f"geometry computed in [cyan]{geo_frames}[/cyan] "
            f"({geo_frames / max(processed, 1) * 100:.0f}% had ball+stumps)"
        )
        if output_path:
            console.print(f"  Annotated video: [cyan]{output_path}[/cyan]")

        return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Cricket Detection — single model or dual model with line/length geometry"
    )
    parser.add_argument("--image",  type=str, help="Single image to analyze")
    parser.add_argument("--video",  type=str, help="Video file to run detection on")
    parser.add_argument("--output", type=str, help="Output path for annotated image/video")
    parser.add_argument("--dual",   action="store_true",
                        help="Use DualModelDetector (both models + line/length geometry)")
    parser.add_argument("--conf",   type=float, default=0.4)
    parser.add_argument("--sample", type=int,   default=3,
                        help="Sample every N frames for video (default: 3)")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            console.print(f"[red]✗ Cannot read image: {args.image}[/red]")
        elif args.dual:
            # ── Dual model: scene + geometry ──────────────────────────────
            dual = DualModelDetector(conf_threshold=args.conf)
            result = dual.analyze_frame(frame)
            console.print_json(json.dumps({
                "scene":    {k: v for k, v in result["scene"].items() if v},
                "geometry": result["geometry"],
            }, indent=2))
            if args.output:
                annotated = dual.draw_fused(frame, result)
                cv2.imwrite(args.output, annotated)
                console.print(f"[green]✓[/green] Saved: {args.output}")
        else:
            # ── Single model: scene only ───────────────────────────────────
            detector = CricketDetector(conf_threshold=args.conf)
            dets  = detector.detect_frame(frame)
            scene = detector.parse_scene(dets)
            console.print_json(json.dumps({"detections": dets, "scene": scene}, indent=2))
            if args.output:
                annotated = detector._draw_detections(frame, dets)
                cv2.imwrite(args.output, annotated)
                console.print(f"[green]✓[/green] Saved: {args.output}")

    elif args.video:
        if args.dual:
            # ── Dual model video ────────────────────────────────────────────────
            dual = DualModelDetector(conf_threshold=args.conf)
            results = dual.analyze_video(
                args.video,
                output_path=args.output,
                sample_every_n_frames=args.sample,
            )
            # Print first 3 frames as sample
            console.print_json(json.dumps([
                {"frame": r["frame_idx"], "t": r["timestamp"],
                 "geometry": r["geometry"],
                 "ball":     bool(r["scene"].get("ball")),
                 "stumps":   bool(r["scene"].get("stumps"))}
                for r in results[:3]
            ], indent=2))
            # Save full results JSON
            if args.output:
                json_out = args.output.replace(".mp4", ".json").replace(".avi", ".json")
                if json_out == args.output:
                    json_out = args.output + ".json"
                with open(json_out, "w") as fh:
                    json.dump(results, fh, indent=2)
                console.print(f"[green]✓[/green] Results JSON: [cyan]{json_out}[/cyan]")
        else:
            # ── Single model video ────────────────────────────────────────────────
            detector = CricketDetector(conf_threshold=args.conf)
            results = detector.detect_in_video(
                args.video,
                output_path=args.output,
                sample_every_n_frames=args.sample,
            )
            console.print_json(json.dumps(results[:3], indent=2))

    else:
        parser.print_help()
