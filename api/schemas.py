from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="الاستعلام النصي المركب")
    video_id: Optional[str] = Field(None, description="معرف الفيديو")


class IngestRequest(BaseModel):
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    video_id: str = "default_video"


class StrictSearchResponse(BaseModel):
    start_timestamp: str
    end_timestamp: str
    confidence_score: float


class DetailedSearchResponse(StrictSearchResponse):
    search_id: str = ""
    candidate_id: str = ""
    video_id: str = ""
    clip_id: str = ""
    speech_excerpt: str = ""
    ocr_text: str = ""
    retrieval_method: str = ""
    temporal_relation: str = "none"
    relation_satisfied: bool = True
    modality_spans: Dict[str, Any] = Field(default_factory=dict)
    spatial_evidence: Dict[str, Any] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    stage_latencies: Dict[str, float] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    video_id: str
    total_clips: int
    total_frames: int = 0
    segmentation_method: str = "scene_plus_asr"
    metrics: Dict[str, Any] = Field(default_factory=dict)
