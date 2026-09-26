"""Content-aware scene detector with a deterministic fixed-window fallback."""
from dataclasses import dataclass
from typing import List
import os
import cv2

@dataclass
class SceneBoundary:
    start_ts: float
    end_ts: float

class SceneDetector:
    def __init__(self, target_scene_sec: float = 6.0, threshold: float = 27.0):
        self.target_scene_sec = float(target_scene_sec)
        self.threshold = float(threshold)

    def _duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not fps or fps <= 0 or total_frames <= 0:
                raise ValueError(f"Invalid video metadata: fps={fps}, frames={total_frames}")
            return total_frames / fps
        finally:
            cap.release()

    def _fixed_windows(self, duration: float) -> List[SceneBoundary]:
        result = []
        start = 0.0
        while start < duration:
            end = min(start + self.target_scene_sec, duration)
            if end > start:
                result.append(SceneBoundary(round(start, 3), round(end, 3)))
            start = end
        return result

    def detect_scenes(self, video_path: str) -> List[SceneBoundary]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        duration = self._duration(video_path)
        try:
            from scenedetect import detect, ContentDetector
            raw_scenes = detect(video_path, ContentDetector(threshold=self.threshold))
            scenes = [
                SceneBoundary(round(start.get_seconds(), 3), round(end.get_seconds(), 3))
                for start, end in raw_scenes
                if end.get_seconds() > start.get_seconds()
            ]
            if scenes:
                print(f"[SceneDetector] Content-aware scenes: {len(scenes)}")
                return scenes
        except Exception as exc:
            print(f"[SceneDetector] Content detector unavailable; fixed-window fallback: {exc}")
        scenes = self._fixed_windows(duration)
        print(f"[SceneDetector] Fixed-window fallback: {len(scenes)}")
        return scenes
