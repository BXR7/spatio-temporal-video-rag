"""Adaptive scene/speech chunking plus timestamped keyframe extraction."""
import os
import shutil
import subprocess
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from .scene_detector import SceneDetector


class VideoProcessor:
    def __init__(
        self,
        min_clip: float = 1.5,
        max_clip: float = 10.0,
        dedup_gap: float = 1.25,
        frame_sample_fps: float = 1.0,
        max_keyframes_per_clip: int = 8,
        scene_detector: Optional[SceneDetector] = None,
    ):
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)
        self.dedup_gap = float(dedup_gap)
        self.frame_sample_fps = max(float(frame_sample_fps), 0.25)
        self.max_keyframes_per_clip = max(int(max_keyframes_per_clip), 3)
        self.scene_detector = scene_detector or SceneDetector()

    @staticmethod
    def video_metadata(video_path: str) -> Dict[str, float]:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frames / fps if fps > 0 and frames > 0 else 0.0
            return {
                "fps": fps,
                "frames": frames,
                "width": width,
                "height": height,
                "duration": duration,
            }
        finally:
            cap.release()

    def _build_clips(self, duration: float, scene_bounds, asr_boundaries: List[float]):
        boundaries = {0.0, round(duration, 3)}
        for scene in scene_bounds:
            boundaries.add(round(float(scene.start_ts), 3))
            boundaries.add(round(float(scene.end_ts), 3))
        for boundary in asr_boundaries:
            if 0.0 < float(boundary) < duration:
                boundaries.add(round(float(boundary), 3))

        ordered = sorted(boundaries)
        deduped = [ordered[0]]
        for boundary in ordered[1:]:
            if boundary - deduped[-1] >= self.dedup_gap:
                deduped.append(boundary)
            else:
                deduped[-1] = max(deduped[-1], boundary)
        if deduped[-1] < duration:
            deduped.append(duration)

        clips = []
        start = deduped[0]
        for end in deduped[1:]:
            while end - start > self.max_clip:
                split = start + self.max_clip
                clips.append((start, split))
                start = split
            if end - start >= self.min_clip:
                clips.append((start, end))
                start = end

        if start < duration:
            if clips and duration - start < self.min_clip:
                previous_start, _ = clips.pop()
                clips.append((previous_start, duration))
            else:
                clips.append((start, duration))
        return [(round(a, 3), round(b, 3)) for a, b in clips if b > a]

    def _sample_timestamps(self, start_ts: float, end_ts: float) -> List[float]:
        duration = max(end_ts - start_ts, 0.001)
        step = 1.0 / self.frame_sample_fps
        values = list(np.arange(start_ts, end_ts, step))
        values.extend([start_ts, start_ts + duration / 2.0, max(start_ts, end_ts - 0.05)])
        values = sorted({round(min(max(v, start_ts), end_ts - 1e-3), 3) for v in values})
        if len(values) > self.max_keyframes_per_clip:
            idx = np.linspace(0, len(values) - 1, self.max_keyframes_per_clip).round().astype(int)
            values = [values[i] for i in idx]
        return values

    def process_video(
        self, video_path: str, asr_boundaries: List[float], output_dir: str
    ) -> List[Dict[str, Any]]:
        metadata = self.video_metadata(video_path)
        duration, fps = metadata["duration"], metadata["fps"]
        scenes = self.scene_detector.detect_scenes(video_path)
        clip_bounds = self._build_clips(duration, scenes, asr_boundaries)

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        clips_data: List[Dict[str, Any]] = []
        try:
            for idx, (start_ts, end_ts) in enumerate(clip_bounds):
                if idx % 25 == 0:
                    print(
                        f"[VideoProcessor] Extracting clip {idx + 1}/"
                        f"{len(clip_bounds)}"
                    )
                keyframes = []
                for frame_order, timestamp in enumerate(self._sample_timestamps(start_ts, end_ts)):
                    frame_index = int(round(timestamp * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    height, width = frame.shape[:2]
                    path = os.path.join(output_dir, f"kf_{idx}_{frame_order}_{timestamp:.3f}.jpg")
                    if not cv2.imwrite(
                        path,
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
                    ):
                        del frame
                        continue
                    keyframes.append({
                        "frame_id": f"clip_{idx}_frame_{frame_order}",
                        "path": path,
                        "timestamp": float(timestamp),
                        "frame_index": frame_index,
                        "width": int(width),
                        "height": int(height),
                        "order": frame_order,
                    })
                    del frame

                if idx % 25 == 0:
                    import gc
                    gc.collect()

                clips_data.append({
                    "clip_id": f"clip_{idx}",
                    "scene_id": f"scene_{idx}",
                    "start_ts": float(start_ts),
                    "end_ts": float(end_ts),
                    "duration": round(end_ts - start_ts, 3),
                    "source_video_path": video_path,
                    "keyframes": keyframes,
                    "keyframe_paths": [item["path"] for item in keyframes],
                    "keyframe_timestamps": [item["timestamp"] for item in keyframes],
                })
        finally:
            cap.release()
        return clips_data

    def extract_audio(self, video_path_or_url: str, output_audio_path: str) -> str:
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        command = [
            ffmpeg_bin, "-y", "-i", video_path_or_url,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_audio_path,
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[Audio Extraction]  {output_audio_path}")
        return output_audio_path
