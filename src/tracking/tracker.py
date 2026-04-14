"""
Cricket Intelligence Engine - Ball Tracker
Tracks detected cricket ball across video frames to build trajectory.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

try:
    import supervision as sv
    SV_AVAILABLE = True
except ImportError:
    SV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class BallTracker:
    """Tracks cricket ball trajectory across video frames."""

    def __init__(self, model_path: str = None):
        """
        Initialize tracker.

        Args:
            model_path: Path to YOLO model for ball detection
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")
            console.print("[yellow]⚠[/yellow] Using pre-trained model. Fine-tune for better tracking.")

        self.trajectory_points = []

    def track_ball_in_video(
        self,
        video_path: str,
        output_path: str = None,
        ball_class_name: str = "sports ball",
        conf_threshold: float = 0.25,
        draw_trajectory: bool = True,
        max_trajectory_length: int = 50,
    ) -> dict:
        """
        Track the ball through a video and optionally render trajectory overlay.

        Args:
            video_path: Input video path
            output_path: Output video path (with trajectory overlay)
            ball_class_name: Class name for ball in the model
            conf_threshold: Detection confidence threshold
            draw_trajectory: Whether to draw trajectory line on video
            max_trajectory_length: Max points to keep in trajectory trail

        Returns:
            Dict with trajectory data and stats
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            console.print(f"[red]✗[/red] Cannot open video: {video_path}")
            return {}

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        console.print(f"[blue]⟳[/blue] Tracking ball in: {video_path}")
        console.print(f"  Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")

        # Setup output video writer
        writer = None
        if output_path and draw_trajectory:
            if not output_path:
                output_path = str(Path(video_path).with_suffix('.tracked.mp4'))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        trajectory = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = self.model.predict(frame, conf=conf_threshold, verbose=False)

            ball_center = None
            ball_conf = 0.0

            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    if ball_class_name.lower() in class_name.lower():
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        ball_center = (cx, cy)
                        ball_conf = float(box.conf[0])
                        break

            # Record trajectory point
            trajectory.append({
                "frame": frame_idx,
                "time_sec": frame_idx / fps if fps > 0 else 0,
                "detected": ball_center is not None,
                "x": ball_center[0] if ball_center else None,
                "y": ball_center[1] if ball_center else None,
                "confidence": ball_conf,
            })

            # Draw trajectory on frame
            if writer and draw_trajectory:
                annotated_frame = frame.copy()

                # Draw trajectory trail
                points_with_detection = [
                    (t["x"], t["y"]) for t in trajectory[-max_trajectory_length:]
                    if t["detected"]
                ]

                for i in range(1, len(points_with_detection)):
                    # Color gradient: older = dimmer
                    alpha = i / len(points_with_detection)
                    color = (
                        int(0 * (1 - alpha) + 0 * alpha),
                        int(255 * alpha),
                        int(255 * (1 - alpha)),
                    )
                    thickness = max(1, int(3 * alpha))
                    cv2.line(
                        annotated_frame,
                        points_with_detection[i - 1],
                        points_with_detection[i],
                        color,
                        thickness,
                    )

                # Draw current detection
                if ball_center:
                    cv2.circle(annotated_frame, ball_center, 8, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, ball_center, 12, (0, 255, 255), 2)
                    cv2.putText(
                        annotated_frame,
                        f"Ball ({ball_conf:.2f})",
                        (ball_center[0] + 15, ball_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )

                # Frame info overlay
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_idx}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )

                writer.write(annotated_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # Compute stats
        detected_frames = sum(1 for t in trajectory if t["detected"])
        stats = {
            "video_path": video_path,
            "output_path": output_path,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detected_frames / total_frames if total_frames > 0 else 0,
            "trajectory": trajectory,
        }

        console.print(
            f"[green]✓[/green] Tracking complete: "
            f"{detected_frames}/{total_frames} frames with ball "
            f"({stats['detection_rate']:.1%})"
        )

        if output_path:
            console.print(f"  Output: {output_path}")

        return stats

    def generate_pitch_map(
        self,
        trajectory: list[dict],
        output_path: str = "data/pitch_map.png",
        width: int = 400,
        height: int = 600,
    ) -> str:
        """
        Generate a simple 2D pitch map showing where the ball pitched.

        This is a simplified visualization — a proper pitch map would need
        perspective transformation from the broadcast camera angle.
        """
        # Create pitch background
        pitch = np.ones((height, width, 3), dtype=np.uint8) * 34  # Dark green

        # Draw pitch rectangle
        pitch_margin = 40
        cv2.rectangle(
            pitch,
            (pitch_margin, pitch_margin),
            (width - pitch_margin, height - pitch_margin),
            (45, 120, 45), -1,
        )

        # Draw crease lines
        crease_y_top = pitch_margin + 80
        crease_y_bottom = height - pitch_margin - 80
        cv2.line(pitch, (pitch_margin, crease_y_top), (width - pitch_margin, crease_y_top), (200, 200, 200), 2)
        cv2.line(pitch, (pitch_margin, crease_y_bottom), (width - pitch_margin, crease_y_bottom), (200, 200, 200), 2)

        # Draw stumps
        stump_x = width // 2
        cv2.rectangle(pitch, (stump_x - 8, crease_y_top - 5), (stump_x + 8, crease_y_top + 5), (200, 200, 200), -1)
        cv2.rectangle(pitch, (stump_x - 8, crease_y_bottom - 5), (stump_x + 8, crease_y_bottom + 5), (200, 200, 200), -1)

        # Plot ball positions (normalize from video coords to pitch coords)
        detected_points = [(t["x"], t["y"]) for t in trajectory if t["detected"]]

        if detected_points:
            xs = [p[0] for p in detected_points]
            ys = [p[1] for p in detected_points]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            for x, y in detected_points:
                # Normalize to pitch dimensions
                nx = int(pitch_margin + (x - x_min) / (x_max - x_min + 1) * (width - 2 * pitch_margin))
                ny = int(pitch_margin + (y - y_min) / (y_max - y_min + 1) * (height - 2 * pitch_margin))
                cv2.circle(pitch, (nx, ny), 4, (0, 100, 255), -1)

        # Labels
        cv2.putText(pitch, "BOWLER", (width // 2 - 30, pitch_margin - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(pitch, "BATSMAN", (width // 2 - 35, height - pitch_margin + 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(pitch, "Ball Trajectory Map", (width // 2 - 80, height - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imwrite(output_path, pitch)
        console.print(f"[green]✓[/green] Pitch map saved: {output_path}")
        return output_path


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track cricket ball in video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video with tracking overlay")
    parser.add_argument("--model", type=str, help="YOLO model weights path")
    parser.add_argument("--ball-class", type=str, default="sports ball")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pitch-map", action="store_true", help="Generate pitch map")
    args = parser.parse_args()

    tracker = BallTracker(model_path=args.model)
    stats = tracker.track_ball_in_video(
        args.video,
        output_path=args.output,
        ball_class_name=args.ball_class,
        conf_threshold=args.conf,
    )

    if args.pitch_map and stats.get("trajectory"):
        tracker.generate_pitch_map(stats["trajectory"])

    # Save trajectory data
    traj_path = str(Path(args.video).with_suffix('.trajectory.json'))
    with open(traj_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"[green]✓[/green] Trajectory data saved: {traj_path}")
