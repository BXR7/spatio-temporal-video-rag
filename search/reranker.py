"""Lazy BGE reranker pinned to the search GPU."""
import gc
from typing import List, Dict, Any, Optional

try:
    import torch
except Exception:
    torch = None

from search.device_detector import configured_devices, release_cuda


class CrossEncoderReranker:
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self, device: Optional[str] = None, autoload: bool = False):
        self.device = device or configured_devices()["search"]
        self.model = None
        self.tokenizer = None
        if autoload:
            self.load_model()

    def load_model(self):
        if self.model is not None:
            return self
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            print(f"[Reranker] Loading BGE reranker on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            kwargs = {"low_cpu_mem_usage": True}
            if torch is not None and self.device.startswith("cuda"):
                kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                **kwargs,
            ).to(self.device)
            self.model.eval()
            print("[Reranker]  Loaded")
        except Exception as exc:
            print(f"[Reranker] Retrieval-score fallback: {exc}")
            self.model = None
        return self

    def release_model(self):
        for name in ("model", "tokenizer"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
            setattr(self, name, None)
        gc.collect()
        release_cuda(self.device)
        print(f"[Reranker]  Released from {self.device}")

    @staticmethod
    def _text(candidate):
        payload = candidate.get("payload", {}) or {}
        parts = [
            payload.get("transcript", ""),
            payload.get("ocr_text", ""),
        ]
        return "\n".join(part for part in parts if part).strip()

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 30):
        if not candidates:
            return candidates
        self.load_model()

        if self.model is None or self.tokenizer is None:
            for candidate in candidates:
                candidate["cross_score"] = max(
                    0.0,
                    min(
                        float(candidate.get("retrieval_similarity", 0.0)),
                        1.0,
                    ),
                )
            return sorted(
                candidates,
                key=lambda item: item["cross_score"],
                reverse=True,
            )

        selected = candidates[:top_k]
        texts = [self._text(candidate) for candidate in selected]
        pairs = [[query, text] for text in texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
        if logits.ndim == 2 and logits.shape[-1] > 1:
            scores = torch.softmax(logits.float(), dim=-1)[:, -1]
        else:
            scores = torch.sigmoid(logits.float().reshape(-1))

        for candidate, score in zip(
            selected,
            scores.detach().cpu().tolist(),
        ):
            model_score = float(score)
            lexical_score = float(
                candidate.get("ocr_lexical_score", 0.0)
            )
            candidate["cross_model_score"] = model_score
            candidate["cross_score"] = min(
                1.0,
                0.80 * model_score + 0.20 * lexical_score,
            )
        for candidate in candidates[top_k:]:
            candidate["cross_score"] = 0.0
        return sorted(
            candidates,
            key=lambda item: item.get("cross_score", 0.0),
            reverse=True,
        )
