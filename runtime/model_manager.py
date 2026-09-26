"""Central lazy/staged model lifecycle for single or dual T4 runtimes."""
import gc
import os
from typing import Optional

from search.device_detector import (
    configured_devices,
    device_diagnostics,
    release_cuda,
)


class RuntimeModelManager:
    def __init__(self):
        self.devices = configured_devices()
        self.transcriber = None
        self.ocr = None
        self.embedder = None
        self.query_planner = None
        self.reranker = None
        self.verifier = None

    def diagnostics(self):
        return device_diagnostics()

    def get_transcriber(self):
        if self.transcriber is None:
            from ingestion.transcriber import AudioTranscriber
            self.transcriber = AudioTranscriber(
                device=self.devices["ingest"],
                autoload=False,
            )
        return self.transcriber

    def get_ocr(self):
        if self.ocr is None:
            from ingestion.ocr_extractor import OCRExtractor
            languages = [
                item.strip()
                for item in os.environ.get("VIDEO_RAG_OCR_LANGS", "en").split(",")
                if item.strip()
            ]
            # EasyOCR internally uses DataParallel whose source device is
            # cuda:0. OCR is sequentially executed after Whisper is released,
            # therefore the primary ingestion GPU is safe and available.
            ocr_device = os.environ.get(
                "VIDEO_RAG_OCR_DEVICE",
                self.devices["ingest"],
            )
            if str(ocr_device).startswith("cuda") and ocr_device != "cuda:0":
                print(
                    f"[RuntimeModelManager] Remapping OCR {ocr_device} -> cuda:0 "
                    f"for EasyOCR DataParallel compatibility"
                )
                ocr_device = "cuda:0"
            self.ocr = OCRExtractor(
                languages=languages or ["en"],
                device=ocr_device,
                autoload=False,
                max_frames_per_clip=int(
                    os.environ.get("VIDEO_RAG_OCR_FRAMES_PER_CLIP", "6")
                ),
                canvas_size=int(
                    os.environ.get("VIDEO_RAG_OCR_CANVAS_SIZE", "1280")
                ),
            )
        return self.ocr

    def get_embedder(self):
        if self.embedder is None:
            from ingestion.embedder import MultimodalEmbedder
            self.embedder = MultimodalEmbedder(
                device=self.devices["ingest"],
                autoload=False,
            )
        return self.embedder

    def get_query_planner(self):
        if self.query_planner is None:
            from search.query_planner import QueryPlanner
            self.query_planner = QueryPlanner(
                device=self.devices["search"],
            )
        return self.query_planner

    def get_reranker(self):
        if self.reranker is None:
            from search.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(
                device=self.devices["search"],
                autoload=False,
            )
        return self.reranker

    def get_verifier(self):
        if self.verifier is None:
            from search.verification import VideoVerificationLayer
            self.verifier = VideoVerificationLayer(
                enable=True,
                device=self.devices["search"],
                autoload=False,
            )
        return self.verifier

    def release(self, name):
        obj = getattr(self, name, None)
        if obj is not None:
            release = getattr(obj, "release_model", None)
            if release is None:
                release = getattr(obj, "release_models", None)
            if callable(release):
                release()
        setattr(self, name, None)
        gc.collect()
        release_cuda()

    def release_ingestion(self):
        for name in ("transcriber", "ocr", "embedder"):
            self.release(name)

    def release_search(self):
        for name in ("query_planner", "reranker", "verifier", "embedder"):
            self.release(name)

    def release_all(self):
        self.release_ingestion()
        self.release_search()
