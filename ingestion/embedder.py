"""Lazy BGE-M3 + SigLIP2 embedder pinned to the ingestion GPU."""
import gc
from typing import List, Dict, Any, Union
import numpy as np

try:
    import torch
    from PIL import Image
except Exception:
    torch = None

from search.device_detector import configured_devices, best_dtype, release_cuda


class MultimodalEmbedder:
    TEXT_MODEL_NAME = "BAAI/bge-m3"
    VISUAL_MODEL_NAME = "google/siglip2-base-patch16-224"
    TEXT_DIM = 1024
    VISUAL_DIM = 768

    def __init__(self, device: str = None, autoload: bool = False):
        self.device = device or configured_devices()["ingest"]
        self.dtype = best_dtype(self.device)
        self.sbert_model = None
        self.siglip_model = None
        self.siglip_processor = None
        if autoload:
            self.load_models()

    def load_text_model(self):
        if self.sbert_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[MultimodalEmbedder] Loading BGE-M3 on {self.device}...")
        self.sbert_model = SentenceTransformer(
            self.TEXT_MODEL_NAME,
            device=self.device,
        )
        print("[MultimodalEmbedder]  BGE-M3 loaded")

    def load_visual_model(self):
        if self.siglip_model is not None:
            return
        from transformers import AutoProcessor, AutoModel
        print(f"[MultimodalEmbedder] Loading SigLIP2 on {self.device}...")
        self.siglip_processor = AutoProcessor.from_pretrained(
            self.VISUAL_MODEL_NAME,
            use_fast=False,
        )
        try:
            self.siglip_model = AutoModel.from_pretrained(
                self.VISUAL_MODEL_NAME,
                dtype=self.dtype,
            )
        except TypeError:
            self.siglip_model = AutoModel.from_pretrained(
                self.VISUAL_MODEL_NAME,
                torch_dtype=self.dtype,
            )
        self.siglip_model = self.siglip_model.to(
            self.device,
            dtype=self.dtype,
        ).eval()
        print(f"[MultimodalEmbedder]  SigLIP2 loaded ({self.dtype})")

    def load_models(self):
        self.load_text_model()
        self.load_visual_model()
        return self

    def release_models(self):
        for name in ("sbert_model", "siglip_model", "siglip_processor"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
            setattr(self, name, None)
        gc.collect()
        release_cuda(self.device)
        print(f"[MultimodalEmbedder]  Released models from {self.device}")

    @staticmethod
    def _zero(dim):
        return [0.0] * dim

    @staticmethod
    def _normalize(vector, target_dim):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if len(vector) > target_dim:
            vector = vector[:target_dim]
        elif len(vector) < target_dim:
            vector = np.pad(vector, (0, target_dim - len(vector)))
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 1e-8 else [0.0] * target_dim

    @staticmethod
    def _extract(output):
        if torch is not None and isinstance(output, torch.Tensor):
            return output
        for name in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, name, None)
            if value is not None:
                return value.mean(dim=1) if name == "last_hidden_state" else value
        return output[0]

    def _move(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def embed_speech(self, text):
        if not str(text).strip():
            return self._zero(self.TEXT_DIM)
        self.load_text_model()
        vector = self.sbert_model.encode(
            str(text),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._normalize(vector, self.TEXT_DIM)

    def embed_ocr(self, text):
        return self.embed_speech(text)

    def embed_query_text(self, text):
        return self.embed_speech(text)

    def embed_visual_frames(self, frames):
        if not frames:
            return []
        self.load_visual_model()
        output = []
        for frame in frames:
            try:
                image = (
                    Image.open(frame).convert("RGB")
                    if isinstance(frame, str)
                    else Image.fromarray(frame)
                )
                inputs = self._move(
                    self.siglip_processor(images=[image], return_tensors="pt")
                )
                with torch.inference_mode():
                    if hasattr(self.siglip_model, "get_image_features"):
                        feature = self.siglip_model.get_image_features(**inputs)
                    else:
                        feature = self._extract(self.siglip_model(**inputs))
                feature = self._extract(feature).float()
                feature = feature / (feature.norm(dim=-1, keepdim=True) + 1e-8)
                output.append(
                    self._normalize(
                        feature.detach().to("cpu", dtype=torch.float32).numpy(),
                        self.VISUAL_DIM,
                    )
                )
            except Exception as exc:
                print(f"[MultimodalEmbedder] Frame embedding failed: {exc}")
                output.append(self._zero(self.VISUAL_DIM))
        return output

    def embed_visual(self, frames):
        vectors = [
            vector for vector in self.embed_visual_frames(frames)
            if np.linalg.norm(vector) > 1e-8
        ]
        if not vectors:
            return self._zero(self.VISUAL_DIM)
        return self._normalize(np.mean(vectors, axis=0), self.VISUAL_DIM)

    def embed_query_visual(self, text):
        if not str(text).strip():
            return self._zero(self.VISUAL_DIM)
        self.load_visual_model()
        try:
            inputs = self._move(
                self.siglip_processor(
                    text=[str(text)],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
            )
            with torch.inference_mode():
                if hasattr(self.siglip_model, "get_text_features"):
                    feature = self.siglip_model.get_text_features(**inputs)
                else:
                    feature = self._extract(self.siglip_model(**inputs))
            feature = self._extract(feature).float()
            feature = feature / (feature.norm(dim=-1, keepdim=True) + 1e-8)
            return self._normalize(
                feature.detach().to("cpu", dtype=torch.float32).numpy(),
                self.VISUAL_DIM,
            )
        except Exception as exc:
            print(f"[MultimodalEmbedder] Visual-query failure: {exc}")
            return self._zero(self.VISUAL_DIM)

    def embed_clips(self, clips: List[Dict[str, Any]]):
        self.load_models()
        for clip in clips:
            keyframes = clip.get("keyframes", [])
            paths = [
                frame.get("path", "")
                for frame in keyframes
                if frame.get("path")
            ]
            visual_vectors = self.embed_visual_frames(paths)
            observations = {
                item.get("frame_id"): item
                for item in clip.get("ocr_observations", [])
            }
            frame_records = []
            for frame, visual_vector in zip(keyframes, visual_vectors):
                observation = observations.get(frame.get("frame_id"), {})
                frame_text = observation.get("text", "")
                frame_records.append({
                    **frame,
                    "visual_embedding": visual_vector,
                    "ocr_text": frame_text,
                    "ocr_embedding": self.embed_ocr(frame_text),
                    "ocr_regions": observation.get("regions", []),
                })
            clip["frame_records"] = frame_records
            clip["visual_embedding"] = self.embed_visual(paths)
            clip["speech_embedding"] = self.embed_speech(
                clip.get("transcript", "")
            )
            clip["ocr_embedding"] = self.embed_ocr(
                clip.get("ocr_text", "")
            )
        return clips
