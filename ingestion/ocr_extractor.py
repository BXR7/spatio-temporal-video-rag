"""Memory-stable spatial OCR for Colab T4 x2.

EasyOCR is the default because it can be pinned explicitly to cuda:1 and
loaded independently from the ingestion models on cuda:0. PaddleOCR is not
loaded in this notebook, preventing PP-OCRv6 pipeline initialization from
killing the Colab kernel.
"""
import gc
import os
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Any, Tuple

try:
    import torch
except Exception:
    torch = None

from PIL import Image


class OCRExtractor:
    def __init__(
        self,
        languages: List[str] = None,
        device: Optional[str] = None,
        autoload: bool = False,
        max_frames_per_clip: int = 6,
        canvas_size: int = 1280,
    ):
        self.languages = languages or ["en"]
        self.device = device or os.environ.get("VIDEO_RAG_OCR_DEVICE", "cpu")
        self.max_frames_per_clip = max(1, int(max_frames_per_clip))
        self.canvas_size = max(640, int(canvas_size))
        self.backend = "none"
        self.reader = None
        self.processed_images = 0
        if autoload:
            self.load_model()

    def load_model(self):
        if self.reader is not None:
            return self

        requested_device = str(self.device)
        print(
            f"[OCRExtractor] Loading EasyOCR; requested={requested_device}, "
            f"languages={self.languages}..."
        )
        try:
            import easyocr

            gpu_argument: Any = False
            if requested_device.startswith("cuda"):
                if torch is None or not torch.cuda.is_available():
                    print("[OCRExtractor] CUDA unavailable; using CPU")
                    gpu_argument = False
                    self.device = "cpu"
                else:
                    # EasyOCR wraps its detector/recognizer with torch.nn.DataParallel.
                    # DataParallel requires the source module on cuda:0. Passing cuda:1
                    # produces: "module must have its parameters ... on cuda:0".
                    # OCR runs after Whisper is released, so cuda:0 is free here.
                    requested_index = int(requested_device.split(":")[-1])
                    if requested_index != 0:
                        print(
                            f"[OCRExtractor] EasyOCR DataParallel cannot use "
                            f"{requested_device} as its source device; remapping to cuda:0"
                        )
                    self.device = "cuda:0"
                    torch.cuda.set_device(0)
                    # True lets EasyOCR choose its supported primary CUDA path.
                    gpu_argument = True

            self.reader = easyocr.Reader(
                self.languages,
                gpu=gpu_argument,
                verbose=False,
                quantize=(gpu_argument is False),
                cudnn_benchmark=False,
            )
            self.backend = "easyocr"
            print(
                f"[OCRExtractor]  EasyOCR loaded; "
                f"effective_device={self.device}, gpu_argument={gpu_argument}"
            )
        except Exception as exc:
            self.reader = None
            self.backend = "none"
            raise RuntimeError(f"EasyOCR could not be loaded: {exc}") from exc
        return self

    def release_model(self):
        if self.reader is not None:
            try:
                del self.reader
            except Exception:
                pass
        self.reader = None
        self.backend = "released"
        gc.collect()

        if torch is not None and torch.cuda.is_available():
            try:
                if str(self.device).startswith("cuda"):
                    index = int(str(self.device).split(":")[-1])
                    with torch.cuda.device(index):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            except Exception:
                torch.cuda.empty_cache()
        print(f"[OCRExtractor]  Released OCR from {self.device}")

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _box_from_polygon(points: Any) -> Optional[List[int]]:
        try:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return [
                int(round(min(xs))),
                int(round(min(ys))),
                int(round(max(xs))),
                int(round(max(ys))),
            ]
        except Exception:
            return None

    @staticmethod
    def _bbox_norm(bbox: List[int], width: int, height: int) -> List[int]:
        if not bbox or width <= 0 or height <= 0:
            return []
        x1, y1, x2, y2 = bbox
        return [
            int(round(1000 * x1 / width)),
            int(round(1000 * y1 / height)),
            int(round(1000 * x2 / width)),
            int(round(1000 * y2 / height)),
        ]

    @staticmethod
    def _dedupe_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
        for region in regions:
            key = (
                region.get("text", "").casefold(),
                tuple(region.get("bbox_norm", []) or []),
            )
            if key not in best or region.get("confidence", 0.0) > best[key].get("confidence", 0.0):
                best[key] = region
        return list(best.values())

    def extract_regions_from_frame(
        self,
        image_path: str,
        timestamp: float = 0.0,
        frame_id: str = "",
    ) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {
                "frame_id": frame_id,
                "path": image_path,
                "timestamp": float(timestamp),
                "text": "",
                "regions": [],
                "width": 0,
                "height": 0,
            }

        self.load_model()

        with Image.open(image_path) as image:
            width, height = image.size

        regions: List[Dict[str, Any]] = []
        try:
            result = self.reader.readtext(
                image_path,
                decoder="greedy",
                beamWidth=1,
                batch_size=1,
                workers=0,
                detail=1,
                paragraph=False,
                min_size=8,
                text_threshold=0.55,
                low_text=0.30,
                link_threshold=0.30,
                canvas_size=self.canvas_size,
                mag_ratio=1.0,
                rotation_info=None,
            )
            for item in result:
                if len(item) < 3:
                    continue
                polygon, text, confidence = item[0], item[1], float(item[2])
                clean = self._clean(text)
                if not clean or confidence < 0.18:
                    continue
                bbox = self._box_from_polygon(polygon) or []
                regions.append({
                    "text": clean,
                    "confidence": confidence,
                    "bbox": bbox,
                    "bbox_norm": self._bbox_norm(bbox, width, height) if bbox else [],
                })
        except RuntimeError as exc:
            error_text = str(exc).casefold()
            if "module must have its parameters" in error_text:
                raise RuntimeError(
                    "EasyOCR DataParallel device mismatch. The OCR engine must "
                    "start on cuda:0; use Video RAG 3.4 or set "
                    "VIDEO_RAG_OCR_DEVICE=cuda:0 before importing the project."
                ) from exc

            # Retry once on CPU if a CUDA allocation fails.
            if "out of memory" in error_text and str(self.device).startswith("cuda"):
                print("[OCRExtractor] CUDA OOM; retrying this frame on CPU")
                self.release_model()
                self.device = "cpu"
                self.load_model()
                return self.extract_regions_from_frame(
                    image_path,
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
            raise

        self.processed_images += 1
        if self.processed_images % 50 == 0:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                try:
                    index = int(str(self.device).split(":")[-1])
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            print(f"[OCRExtractor] Processed {self.processed_images} frames")

        regions = self._dedupe_regions(regions)
        text = " ".join(dict.fromkeys(
            region["text"] for region in regions if region.get("text")
        ))
        return {
            "frame_id": frame_id,
            "path": image_path,
            "timestamp": float(timestamp),
            "text": text,
            "regions": regions,
            "width": int(width),
            "height": int(height),
        }

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[\w-]+", a.casefold()))
        b_tokens = set(re.findall(r"[\w-]+", b.casefold()))
        if not a_tokens or not b_tokens:
            return 0.0
        jaccard = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
        sequence = SequenceMatcher(
            None,
            " ".join(sorted(a_tokens)),
            " ".join(sorted(b_tokens)),
        ).ratio()
        return max(jaccard, sequence)

    def build_ocr_intervals(
        self,
        observations: List[Dict[str, Any]],
        max_gap: float = 1.6,
    ) -> List[Dict[str, Any]]:
        intervals: List[Dict[str, Any]] = []
        for observation in sorted(
            observations,
            key=lambda item: item.get("timestamp", 0.0),
        ):
            text = observation.get("text", "")
            if not text:
                continue
            timestamp = float(observation.get("timestamp", 0.0))
            if (
                intervals
                and timestamp - intervals[-1]["end_ts"] <= max_gap
                and self._similarity(text, intervals[-1]["text"]) >= 0.45
            ):
                intervals[-1]["end_ts"] = timestamp
                intervals[-1]["observations"].append(observation)
                if len(text) > len(intervals[-1]["text"]):
                    intervals[-1]["text"] = text
            else:
                intervals.append({
                    "text": text,
                    "start_ts": timestamp,
                    "end_ts": timestamp,
                    "observations": [observation],
                })

        for interval in intervals:
            if interval["end_ts"] <= interval["start_ts"]:
                interval["end_ts"] = interval["start_ts"] + 0.5
        return intervals

    def _select_frames(
        self,
        keyframes: List[Dict[str, Any]],
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = max_frames or self.max_frames_per_clip
        if len(keyframes) <= limit:
            return keyframes
        indices = [
            round(index * (len(keyframes) - 1) / (limit - 1))
            for index in range(limit)
        ]
        return [keyframes[index] for index in indices]

    def process_clip_ocr_detailed(
        self,
        keyframes: List[Dict[str, Any]],
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.load_model()
        selected = self._select_frames(keyframes, max_frames=max_frames)

        observations = []
        for frame in selected:
            observations.append(
                self.extract_regions_from_frame(
                    frame.get("path", ""),
                    frame.get("timestamp", 0.0),
                    frame.get("frame_id", ""),
                )
            )

        combined = " ".join(dict.fromkeys(
            observation["text"]
            for observation in observations
            if observation.get("text")
        ))
        return {
            "combined_text": combined,
            "observations": observations,
            "intervals": self.build_ocr_intervals(observations),
            "processed_frames": len(observations),
            "nonempty_frames": sum(
                1 for observation in observations if observation.get("text")
            ),
            "region_count": sum(
                len(observation.get("regions", []))
                for observation in observations
            ),
        }

    def process_clip_ocr(
        self,
        keyframe_paths: List[str],
        max_frames: int = 4,
    ) -> str:
        keyframes = [
            {
                "frame_id": f"frame_{index}",
                "path": path,
                "timestamp": float(index),
            }
            for index, path in enumerate(keyframe_paths)
        ]
        return self.process_clip_ocr_detailed(
            keyframes,
            max_frames=max_frames,
        )["combined_text"]
