"""Coarse clip retrieval plus fine frame retrieval and weighted RRF fusion."""
from collections import defaultdict
from typing import List, Dict, Any, Optional


class MultimodalRetriever:
    def __init__(self, indexer: Any, embedder: Any):
        self.indexer = indexer
        self.embedder = embedder

    @staticmethod
    def _identity(result):
        payload = result.get("payload", {}) or {}
        video_id = result.get("video_id") or payload.get("video_id", "")
        clip_id = result.get("clip_id") or payload.get("clip_id", "")
        return f"{video_id}:{clip_id}" if video_id and clip_id else result.get("candidate_id", "")

    def _query_vector(self, modality: str, query: str):
        return (
            self.embedder.embed_query_visual(query)
            if modality == "visual"
            else self.embedder.embed_query_text(query)
        )

    @staticmethod
    def _normalize_ocr_text(text):
        import re
        return " ".join(
            re.findall(r"[a-z0-9._:/-]+", str(text).casefold())
        )

    @classmethod
    def _ocr_lexical_score(cls, query, text):
        import re
        query_norm = cls._normalize_ocr_text(query)
        text_norm = cls._normalize_ocr_text(text)
        if not query_norm or not text_norm:
            return 0.0

        code_targets = []
        for match in re.finditer(
            r"\b[A-Z][A-Z0-9_-]{2,}(?:\s+[:=/-]?\s*\d+(?:[.:/-]\d+)*)?",
            str(query),
        ):
            target = cls._normalize_ocr_text(match.group(0))
            if target:
                code_targets.append(target)

        for target in code_targets:
            if target in text_norm:
                return 1.0

        numbers = set(re.findall(r"\b\d{2,}\b", query_norm))
        if numbers and numbers.issubset(set(text_norm.split())):
            return 0.85

        q_tokens = set(query_norm.split())
        t_tokens = set(text_norm.split())
        return len(q_tokens & t_tokens) / max(len(q_tokens), 1)

    def search(self, plan, video_id: Optional[str] = None, top_k: int = 20):
        hits_per_modality = {}
        for modality in plan.modalities:
            merged = {}
            for query in plan.queries_by_modality.get(modality, plan.queries):
                vector = self._query_vector(modality, query)
                hits = self.indexer.search(
                    collection_name=self.indexer.collection_name,
                    query_vector=(modality, vector),
                    limit=top_k * 3,
                    video_id=video_id,
                )
                for hit in hits:
                    identity = self._identity(hit)
                    if identity and (
                        identity not in merged or hit.get("score", 0.0) > merged[identity].get("score", 0.0)
                    ):
                        merged[identity] = hit
            hits_per_modality[modality] = sorted(
                merged.values(), key=lambda item: item.get("score", 0.0), reverse=True
            )[:top_k * 2]

        print("[Retriever] coarse hits:", {name: len(items) for name, items in hits_per_modality.items()})
        active = {name: items for name, items in hits_per_modality.items() if items}
        if not active:
            return []
        candidates = self.weighted_rrf(
            active,
            plan.modality_weights,
        )

        if "ocr" in plan.modalities:
            ocr_query = plan.query_for("ocr")
            for candidate in candidates:
                payload = candidate.get("payload", {}) or {}
                lexical_score = self._ocr_lexical_score(
                    ocr_query,
                    payload.get("ocr_text", ""),
                )
                candidate["ocr_lexical_score"] = lexical_score
                candidate["retrieval_similarity"] = min(
                    1.0,
                    0.75 * candidate.get(
                        "retrieval_similarity", 0.0
                    )
                    + 0.25 * lexical_score,
                )

            candidates.sort(
                key=lambda item: (
                    item.get("ocr_lexical_score", 0.0),
                    item.get("rrf_score", 0.0),
                ),
                reverse=True,
            )

        return candidates[:top_k]

    def weighted_rrf(self, hits_per_modality, weights, k: int = 60):
        aggregated = {}
        for modality, hits in hits_per_modality.items():
            weight = float(weights.get(modality, 1.0))
            for rank, hit in enumerate(hits):
                identity = self._identity(hit)
                payload = dict(hit.get("payload", {}) or {})
                payload["candidate_id"] = identity
                if identity not in aggregated:
                    aggregated[identity] = {
                        "candidate_id": identity,
                        "clip_id": hit.get("clip_id", payload.get("clip_id", "")),
                        "video_id": hit.get("video_id", payload.get("video_id", "")),
                        "payload": payload,
                        "rrf_score": 0.0,
                        "per_modality_scores": {},
                    }
                item = aggregated[identity]
                item["rrf_score"] += weight / (k + rank + 1)
                item["per_modality_scores"][modality] = max(
                    hit.get("score", 0.0),
                    item["per_modality_scores"].get(modality, -1.0),
                )
                for key, value in payload.items():
                    if value not in (None, "", [], {}):
                        item["payload"][key] = value

        for item in aggregated.values():
            scores = item["per_modality_scores"]
            item["retrieval_similarity"] = sum(
                float(weights.get(name, 0.0)) * max(min(float(score), 1.0), 0.0)
                for name, score in scores.items()
            )
        return sorted(aggregated.values(), key=lambda item: item["rrf_score"], reverse=True)

    def attach_fine_frame_evidence(
        self, plan, candidates: List[Dict[str, Any]], video_id: Optional[str] = None, top_frames: int = 16
    ):
        if not candidates:
            return candidates
        top_clip_ids = [item.get("clip_id", "") for item in candidates[:10]]
        evidence_by_candidate = defaultdict(list)

        for modality in ("visual", "ocr"):
            if modality not in plan.modalities:
                continue
            query = plan.query_for(modality)
            vector = self._query_vector(modality, query)
            hits = self.indexer.search_frames(
                query_vector=(modality, vector),
                video_id=video_id,
                clip_ids=top_clip_ids,
                limit=top_frames,
            )
            for hit in hits:
                payload = hit.get("payload", {}) or {}
                candidate_id = payload.get("candidate_id") or self._identity(hit)
                evidence_by_candidate[candidate_id].append({
                    "modality": modality,
                    "score": float(hit.get("score", 0.0)),
                    "timestamp": float(payload.get("timestamp", 0.0)),
                    "path": payload.get("path", ""),
                    "frame_id": payload.get("frame_id", ""),
                    "width": int(payload.get("width", 0)),
                    "height": int(payload.get("height", 0)),
                    "ocr_text": payload.get("ocr_text", ""),
                    "ocr_regions": payload.get("ocr_regions", []),
                })

        for candidate in candidates:
            evidence = sorted(
                evidence_by_candidate.get(candidate["candidate_id"], []),
                key=lambda item: item["score"],
                reverse=True,
            )
            candidate["frame_evidence"] = evidence
            candidate["payload"]["frame_evidence"] = evidence
            if evidence:
                visual_scores = [item["score"] for item in evidence if item["modality"] == "visual"]
                ocr_scores = [item["score"] for item in evidence if item["modality"] == "ocr"]
                candidate["fine_visual_score"] = max(visual_scores, default=0.0)
                candidate["fine_ocr_score"] = max(ocr_scores, default=0.0)
                candidate["retrieval_similarity"] = min(
                    1.0,
                    0.75 * candidate.get("retrieval_similarity", 0.0)
                    + 0.15 * max(candidate["fine_visual_score"], 0.0)
                    + 0.10 * max(candidate["fine_ocr_score"], 0.0),
                )
        return candidates
