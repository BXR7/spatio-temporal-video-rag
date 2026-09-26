"""Qdrant clip + frame index for coarse-to-fine spatio-temporal retrieval."""
import os
import uuid
import json
from typing import List, Dict, Any, Optional

import numpy as np


class QdrantIndexer:
    COLLECTION_NAME = "spatio_temporal_clips"
    FRAME_COLLECTION_NAME = "spatio_temporal_frames"
    VISUAL_DIM = 768
    TEXT_DIM = 1024

    def __init__(
        self,
        collection_name: str = None,
        location: str = "storage/qdrant_db",
    ):
        self.collection_name = collection_name or self.COLLECTION_NAME
        self.frame_collection_name = self.FRAME_COLLECTION_NAME
        self.client = None
        self.clip_cache_file = os.path.join("storage", f"index_{self.collection_name}.json")
        self.frame_cache_file = os.path.join("storage", f"index_{self.frame_collection_name}.json")
        self.clip_storage = self._load_cache(self.clip_cache_file)
        self.frame_storage = self._load_cache(self.frame_cache_file)
        self.local_storage = self.clip_storage

        try:
            from qdrant_client import QdrantClient

            self.location = location
            if location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                os.makedirs(location, exist_ok=True)
                self.client = QdrantClient(path=location)

            self._ensure_collections()
        except Exception as exc:
            print(f"[QdrantIndexer] Local fallback active: {exc}")
            self.client = None

    @staticmethod
    def _load_cache(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                pass
        return {}

    @staticmethod
    def _save_cache(path: str, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)

    def _save_disk_cache(self):
        self._save_cache(self.clip_cache_file, self.clip_storage)
        self._save_cache(self.frame_cache_file, self.frame_storage)

    def _ensure_collections(self):
        from qdrant_client.models import VectorParams, Distance
        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "visual": VectorParams(size=self.VISUAL_DIM, distance=Distance.COSINE),
                    "speech": VectorParams(size=self.TEXT_DIM, distance=Distance.COSINE),
                    "ocr": VectorParams(size=self.TEXT_DIM, distance=Distance.COSINE),
                },
            )
        if self.frame_collection_name not in existing:
            self.client.create_collection(
                collection_name=self.frame_collection_name,
                vectors_config={
                    "visual": VectorParams(size=self.VISUAL_DIM, distance=Distance.COSINE),
                    "ocr": VectorParams(size=self.TEXT_DIM, distance=Distance.COSINE),
                },
            )

    @staticmethod
    def _candidate_id(video_id: str, clip_id: str) -> str:
        return f"{video_id}:{clip_id}"

    @staticmethod
    def _frame_candidate_id(video_id: str, clip_id: str, frame_id: str) -> str:
        return f"{video_id}:{clip_id}:{frame_id}"

    @staticmethod
    def _point_id(identity: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"video-rag://{identity}"))

    def _build_filter(self, video_id=None, query_filter=None):
        if self.client is None:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if video_id:
            conditions.append(FieldCondition(key="video_id", match=MatchValue(value=video_id)))
        for key, value in (query_filter or {}).items():
            if isinstance(value, (list, tuple, set)):
                try:
                    from qdrant_client.models import MatchAny
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
                except Exception:
                    continue
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def _delete_video_records(self, video_id: str):
        if self.client is not None:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
                selector = FilterSelector(
                    filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
                )
                for collection in (self.collection_name, self.frame_collection_name):
                    self.client.delete(collection_name=collection, points_selector=selector, wait=True)
            except Exception as exc:
                print(f"[QdrantIndexer] Cleanup notice: {exc}")

        for storage in (self.clip_storage, self.frame_storage):
            stale = [
                key for key, item in storage.items()
                if (item.get("payload", {}) or {}).get("video_id") == video_id
            ]
            for key in stale:
                storage.pop(key, None)

    @staticmethod
    def _keywords(clip: Dict[str, Any]) -> List[str]:
        tokens = set()
        for field in ("transcript", "ocr_text"):
            for token in str(clip.get(field, "")).casefold().split():
                token = token.strip(".,;:!?\"'()[]{}")
                if len(token) > 2:
                    tokens.add(token)
        return sorted(tokens)[:80]

    def index_clips(
        self, video_id: str, clips: List[Dict[str, Any]], source_video_path: str = ""
    ) -> int:
        self._delete_video_records(video_id)
        clip_points, frame_points = [], []

        try:
            from qdrant_client.models import PointStruct
        except Exception:
            PointStruct = None

        for clip in clips:
            clip_id = clip.get("clip_id", "")
            candidate_id = self._candidate_id(video_id, clip_id)
            payload = {
                "candidate_id": candidate_id,
                "clip_id": clip_id,
                "video_id": video_id,
                "source_video_path": source_video_path or clip.get("source_video_path", ""),
                "start_ts": float(clip.get("start_ts", 0.0)),
                "end_ts": float(clip.get("end_ts", 0.0)),
                "duration": float(clip.get("duration", 0.0)),
                "scene_id": clip.get("scene_id", ""),
                "transcript": clip.get("transcript", ""),
                "speech_text": clip.get("transcript", ""),
                "ocr_text": clip.get("ocr_text", ""),
                "keywords": self._keywords(clip),
                "word_timestamps": clip.get("word_timestamps", []),
                "ocr_observations": clip.get("ocr_observations", []),
                "ocr_intervals": clip.get("ocr_intervals", []),
                "keyframes": [
                    {key: value for key, value in frame.items()
                     if key not in {"visual_embedding", "ocr_embedding"}}
                    for frame in clip.get("frame_records", clip.get("keyframes", []))
                ],
                "keyframe_paths": clip.get("keyframe_paths", []),
            }
            vectors = {
                "visual": clip.get("visual_embedding", [0.0] * self.VISUAL_DIM),
                "speech": clip.get("speech_embedding", [0.0] * self.TEXT_DIM),
                "ocr": clip.get("ocr_embedding", [0.0] * self.TEXT_DIM),
            }
            self.clip_storage[candidate_id] = {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "clip_id": clip_id,
                "video_id": video_id,
                "vectors": vectors,
                "payload": payload,
            }
            if PointStruct is not None:
                clip_points.append(PointStruct(
                    id=self._point_id(candidate_id), vector=vectors, payload=payload
                ))

            for frame in clip.get("frame_records", []):
                frame_id = frame.get("frame_id", "")
                frame_candidate_id = self._frame_candidate_id(video_id, clip_id, frame_id)
                frame_payload = {
                    "frame_candidate_id": frame_candidate_id,
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "timestamp": float(frame.get("timestamp", 0.0)),
                    "frame_index": int(frame.get("frame_index", 0)),
                    "path": frame.get("path", ""),
                    "width": int(frame.get("width", 0)),
                    "height": int(frame.get("height", 0)),
                    "ocr_text": frame.get("ocr_text", ""),
                    "ocr_regions": frame.get("ocr_regions", []),
                }
                frame_vectors = {
                    "visual": frame.get("visual_embedding", [0.0] * self.VISUAL_DIM),
                    "ocr": frame.get("ocr_embedding", [0.0] * self.TEXT_DIM),
                }
                self.frame_storage[frame_candidate_id] = {
                    "id": frame_candidate_id,
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "vectors": frame_vectors,
                    "payload": frame_payload,
                }
                if PointStruct is not None:
                    frame_points.append(PointStruct(
                        id=self._point_id(frame_candidate_id),
                        vector=frame_vectors,
                        payload=frame_payload,
                    ))

        if self.client is not None and PointStruct is not None:
            try:
                if clip_points:
                    self.client.upsert(self.collection_name, clip_points, wait=True)
                if frame_points:
                    self.client.upsert(self.frame_collection_name, frame_points, wait=True)
            except Exception as exc:
                print(f"[QdrantIndexer] Qdrant upsert failed; local index retained: {exc}")

        self._save_disk_cache()
        return len(clips)

    @staticmethod
    def _map_qdrant_results(results):
        mapped = []
        for result in results:
            payload = result.payload or {}
            mapped.append({
                "candidate_id": payload.get("candidate_id", ""),
                "frame_candidate_id": payload.get("frame_candidate_id", ""),
                "clip_id": payload.get("clip_id", ""),
                "frame_id": payload.get("frame_id", ""),
                "video_id": payload.get("video_id", ""),
                "payload": payload,
                "score": float(result.score),
            })
        return mapped

    def search(
        self,
        collection_name: str,
        query_vector: tuple,
        limit: int = 10,
        query_filter: Optional[Dict] = None,
        video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        vector_name, vector = query_vector
        collection = collection_name or self.collection_name
        if self.client is not None:
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=vector,
                    using=vector_name,
                    query_filter=self._build_filter(video_id, query_filter),
                    with_payload=True,
                    limit=limit,
                )
                return self._map_qdrant_results(response.points)
            except Exception as exc:
                print(f"[QdrantIndexer] Qdrant search fallback: {exc}")
        storage = self.frame_storage if collection == self.frame_collection_name else self.clip_storage
        return self._local_search(storage, vector_name, vector, video_id, query_filter, limit)

    def search_frames(
        self,
        query_vector: tuple,
        video_id: Optional[str] = None,
        clip_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        query_filter = {"clip_id": clip_ids} if clip_ids else None
        results = self.search(
            collection_name=self.frame_collection_name,
            query_vector=query_vector,
            limit=max(limit * 4 if clip_ids else limit, limit),
            query_filter=query_filter,
            video_id=video_id,
        )
        if clip_ids:
            allowed = set(clip_ids)
            results = [item for item in results if item.get("clip_id") in allowed]
        return results[:limit]

    @staticmethod
    def _local_search(storage, vector_name, vector, video_id, query_filter, limit):
        query = np.asarray(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        hits = []
        for item in storage.values():
            payload = item.get("payload", {})
            if video_id and payload.get("video_id") != video_id:
                continue
            valid = True
            for key, expected in (query_filter or {}).items():
                actual = payload.get(key)
                valid = actual in expected if isinstance(expected, (list, tuple, set)) else actual == expected
                if not valid:
                    break
            if not valid:
                continue
            stored = np.asarray(item.get("vectors", {}).get(vector_name, []), dtype=np.float32)
            if stored.shape != query.shape:
                continue
            norm = np.linalg.norm(stored)
            if norm < 1e-8:
                continue
            score = float(np.dot(query, stored) / (query_norm * norm))
            hits.append({
                "candidate_id": item.get("candidate_id", payload.get("candidate_id", "")),
                "frame_candidate_id": item.get("id", payload.get("frame_candidate_id", "")),
                "clip_id": item.get("clip_id", payload.get("clip_id", "")),
                "frame_id": item.get("frame_id", payload.get("frame_id", "")),
                "video_id": item.get("video_id", payload.get("video_id", "")),
                "payload": payload,
                "score": score,
            })
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]

    def indexed_count(self, video_id: Optional[str] = None) -> int:
        return sum(
            1 for item in self.clip_storage.values()
            if not video_id or (item.get("payload", {}) or {}).get("video_id") == video_id
        )

    def indexed_frame_count(self, video_id: Optional[str] = None) -> int:
        return sum(
            1 for item in self.frame_storage.values()
            if not video_id or (item.get("payload", {}) or {}).get("video_id") == video_id
        )

    def index_status(self, video_id: Optional[str] = None) -> Dict[str, Any]:
        clips = self.indexed_count(video_id)
        frames = self.indexed_frame_count(video_id)
        return {
            "clip_collection": self.collection_name,
            "frame_collection": self.frame_collection_name,
            "video_id": video_id or "*",
            "indexed_clips": clips,
            "indexed_frames": frames,
            "qdrant_connected": self.client is not None,
            "ready": clips > 0 and frames > 0,
        }

    def clear_collection(self):
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.client.delete_collection(self.frame_collection_name)
                self._ensure_collections()
            except Exception:
                pass
        self.clip_storage.clear()
        self.frame_storage.clear()
        self._save_disk_cache()
