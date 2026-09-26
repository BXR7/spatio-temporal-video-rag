"""Spatio-Temporal Multimodal Video RAG application."""
import os
import time
import uuid
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from api.schemas import (
    SearchQueryRequest,
    StrictSearchResponse,
    DetailedSearchResponse,
    IngestResponse,
    IngestRequest,
)
from ingestion.video_processor import VideoProcessor
from ingestion.indexer import QdrantIndexer
from search.retriever import MultimodalRetriever
from search.refiner import TimestampRefiner
from search.confidence import ConfidenceCalculator
from search.temporal_reasoner import TemporalReasoner
from storage.metadata_store import TemporalMetadataStore
from monitoring.performance import PerformanceRegistry
from runtime.model_manager import RuntimeModelManager

app = FastAPI(
    title="Spatio-Temporal Video RAG",
    description=(
        "Visual, speech, OCR, spatial grounding, and temporal reasoning"
    ),
    version="0.1.0",
)

# Lightweight objects only. No neural model is loaded during import.
video_processor = VideoProcessor(frame_sample_fps=1.0)
indexer = QdrantIndexer(
    location=os.environ.get(
        "VIDEO_RAG_QDRANT_PATH",
        "storage/qdrant_db",
    )
)
metadata_store = TemporalMetadataStore()
refiner = TimestampRefiner()
confidence_calculator = ConfidenceCalculator()
temporal_reasoner = TemporalReasoner()
performance_registry = PerformanceRegistry()
model_manager = RuntimeModelManager()



def _derive_video_id(target):
    base = os.path.splitext(os.path.basename(target or ""))[0]
    clean = "".join(
        character if character.isalnum() else "_"
        for character in base
    ).strip("_")
    return clean or f"video_{uuid.uuid4().hex[:8]}"


@app.get("/runtime")
def runtime_status():
    return {
        "models_loaded": {
            name: getattr(model_manager, name) is not None
            for name in (
                "transcriber",
                "ocr",
                "embedder",
                "query_planner",
                "reranker",
                "verifier",
            )
        },
        "devices": model_manager.diagnostics(),
    }


@app.post("/runtime/release")
def release_runtime_models():
    model_manager.release_all()
    return runtime_status()


@app.post("/ingest", response_model=IngestResponse)
def ingest_video_endpoint(request: IngestRequest):
    target = request.video_path or request.video_url
    if not target:
        raise HTTPException(400, "video_path or video_url is required")
    if target.startswith(("http://", "https://")):
        raise HTTPException(400, "Upload the video locally in this notebook")
    if not os.path.exists(target):
        raise HTTPException(400, f"Video does not exist: {target}")

    video_id = (
        request.video_id
        if request.video_id != "default_video"
        else _derive_video_id(target)
    )
    temp_dir = os.path.join("storage", "processed", video_id)
    keyframe_dir = os.path.join(temp_dir, "keyframes")
    audio_path = os.path.join(temp_dir, "audio.wav")
    os.makedirs(temp_dir, exist_ok=True)

    stage_metrics = {}
    process_started = time.perf_counter()
    metadata = video_processor.video_metadata(target)

    try:
        # CPU stage.
        with performance_registry.stage(stage_metrics, "audio_extraction"):
            video_processor.extract_audio(target, audio_path)

        # GPU 0 stage 1: Whisper only.
        transcriber = model_manager.get_transcriber()
        with performance_registry.stage(stage_metrics, "asr"):
            words = transcriber.transcribe_audio(audio_path)
            asr_boundaries = transcriber.detect_sentence_boundaries(words)
        transcriber.release_model()

        # CPU stage.
        with performance_registry.stage(
            stage_metrics,
            "segmentation_and_frames",
        ):
            clips = video_processor.process_video(
                target,
                asr_boundaries,
                keyframe_dir,
            )
            clips = transcriber.align_transcripts_to_chunks(clips, words)
        model_manager.release("transcriber")

        # OCR stage after the transcription model has been released.
        ocr_engine = model_manager.get_ocr()
        print(
            f"[Ingestion] Starting stable OCR on {ocr_engine.device}; "
            f"clips={len(clips)}"
        )
        ocr_engine.load_model()
        ocr_processed_frames = 0
        ocr_nonempty_frames = 0
        ocr_region_count = 0
        with performance_registry.stage(stage_metrics, "spatial_ocr"):
            for clip in clips:
                result = ocr_engine.process_clip_ocr_detailed(
                    clip.get("keyframes", [])
                )
                clip["ocr_text"] = result["combined_text"]
                clip["ocr_observations"] = result["observations"]
                clip["ocr_intervals"] = result["intervals"]
                ocr_processed_frames += result.get("processed_frames", 0)
                ocr_nonempty_frames += result.get("nonempty_frames", 0)
                ocr_region_count += result.get("region_count", 0)
        ocr_backend_name = ocr_engine.backend
        print(
            f"[Ingestion] OCR backend={ocr_backend_name} "
            f"processed={ocr_processed_frames} "
            f"nonempty={ocr_nonempty_frames} "
            f"regions={ocr_region_count}"
        )
        model_manager.release("ocr")

        # GPU 0 stage 2: BGE-M3 + SigLIP2 only.
        embedder = model_manager.get_embedder()
        with performance_registry.stage(
            stage_metrics,
            "multimodal_embeddings",
        ):
            clips = embedder.embed_clips(clips)

        # Index before releasing because vectors live in Python lists.
        with performance_registry.stage(stage_metrics, "qdrant_indexing"):
            metadata_store.clear_video(video_id)
            total_clips = indexer.index_clips(
                video_id,
                clips,
                source_video_path=target,
            )
            for clip in clips:
                metadata_store.store_word_timestamps(
                    clip["clip_id"],
                    video_id,
                    clip.get("word_timestamps", []),
                )
                metadata_store.store_ocr_observations(
                    clip["clip_id"],
                    video_id,
                    clip.get("ocr_observations", []),
                )
        model_manager.release("embedder")

    except Exception:
        model_manager.release_ingestion()
        raise

    elapsed = time.perf_counter() - process_started
    snapshots = [
        value for value in stage_metrics.values()
        if isinstance(value, dict)
    ]
    record = {
        "video_id": video_id,
        "video_duration_sec": round(metadata["duration"], 3),
        "total_clips": total_clips,
        "total_frames": indexer.indexed_frame_count(video_id),
        "latency_sec": round(elapsed, 3),
        "seconds_per_video_minute": round(
            elapsed / max(metadata["duration"] / 60.0, 1e-8),
            3,
        ),
        "max_observed_ram_mb": max(
            (
                item.get("max_observed_ram_mb", 0.0)
                for item in snapshots
            ),
            default=0.0,
        ),
        "peak_vram_mb": max(
            (item.get("peak_vram_mb", 0.0) for item in snapshots),
            default=0.0,
        ),
        "stages": stage_metrics,
        "device_assignment": model_manager.devices,
        "ocr_backend": ocr_backend_name,
        "ocr_processed_frames": ocr_processed_frames,
        "ocr_nonempty_frames": ocr_nonempty_frames,
        "ocr_region_count": ocr_region_count,
    }
    performance_registry.record_ingestion(record)
    return IngestResponse(
        status="success",
        video_id=video_id,
        total_clips=total_clips,
        total_frames=record["total_frames"],
        metrics=record,
    )


def ingest_video(video_path=None, video_url=None, video_id=None):
    return ingest_video_endpoint(
        IngestRequest(
            video_path=video_path,
            video_url=video_url,
            video_id=video_id
            or _derive_video_id(video_path or video_url or ""),
        )
    )


@dataclass(frozen=True)
class SearchExecution:
    search_id: str
    best: dict
    refined: dict
    plan: object
    stage_latencies: dict


def _assert_integrity(candidate):
    payload = candidate.get("payload", {}) or {}
    expected = (
        f"{payload.get('video_id', '')}:"
        f"{payload.get('clip_id', '')}"
    )
    if candidate.get("candidate_id") != expected:
        raise RuntimeError(
            f"Candidate integrity error: "
            f"{candidate.get('candidate_id')} != {expected}"
        )


def execute_search_once(request: SearchQueryRequest) -> SearchExecution:
    query = request.query.strip()
    video_id = request.video_id
    if not query:
        raise HTTPException(400, "Query cannot be empty")
    status = indexer.index_status(video_id)
    if not status["ready"]:
        raise HTTPException(409, f"Index is not ready: {status}")

    stages = {}
    search_started = time.perf_counter()

    try:
        # Query planning stage. The deterministic parser is the default.
        stage = time.perf_counter()
        planner = model_manager.get_query_planner()
        plan = planner.plan(query)
        stages["query_planning"] = time.perf_counter() - stage
        model_manager.release("query_planner")

        # GPU 0 stage: BGE-M3 + SigLIP2 query embeddings.
        stage = time.perf_counter()
        embedder = model_manager.get_embedder()
        retriever = MultimodalRetriever(indexer, embedder)
        candidates = retriever.search(
            plan,
            video_id=video_id,
            top_k=30,
        )
        candidates = retriever.attach_fine_frame_evidence(
            plan,
            candidates,
            video_id=video_id,
        )
        stages["coarse_and_fine_retrieval"] = (
            time.perf_counter() - stage
        )
        model_manager.release("embedder")
        if not candidates:
            raise HTTPException(404, "No candidates returned")

        # GPU 1 stage 2: BGE reranker only.
        stage = time.perf_counter()
        reranker = model_manager.get_reranker()
        candidates = reranker.rerank(query, candidates)
        stages["cross_encoder_rerank"] = (
            time.perf_counter() - stage
        )
        model_manager.release("reranker")

        # Spatial verification stage. Qwen-VL is loaded only when required.
        stage = time.perf_counter()
        verifier = model_manager.get_verifier()
        candidates = verifier.ground_candidates(
            query,
            plan,
            candidates,
            top_k=5,
        )
        stages["spatial_temporal_grounding"] = (
            time.perf_counter() - stage
        )
        model_manager.release("verifier")

        # CPU temporal algebra and confidence.
        stage = time.perf_counter()
        evaluated = []
        for candidate in candidates[:30]:
            _assert_integrity(candidate)
            refined = refiner.refine_timestamps(candidate, plan)
            grounding = candidate.get("grounding", {}) or {}
            spatial_score = (
                grounding.get("confidence", 0.0)
                if grounding.get("bbox_norm")
                else 0.6 * grounding.get("confidence", 0.0)
            )
            confidence, breakdown = confidence_calculator.calculate(
                candidate.get("retrieval_similarity", 0.0),
                candidate.get("cross_score", 0.0),
                candidate.get("visual_verification_score", 0.0),
                refined.get("temporal_score", 0.0),
                spatial_score,
                relation_required=plan.temporal_relation != "none",
                relation_satisfied=refined.get(
                    "relation_satisfied",
                    True,
                ),
            )
            candidate["refined"] = refined
            candidate["final_confidence"] = confidence
            candidate["score_breakdown"] = breakdown
            evaluated.append(candidate)

        ordered = temporal_reasoner.select(
            evaluated,
            plan.temporal_selector,
        )
        if not ordered:
            raise HTTPException(
                404,
                f"No candidate satisfies temporal relation {plan.temporal_relation!r}.",
            )
        best = ordered[0]
        refined = best["refined"]
        stages["temporal_algebra_and_selection"] = (
            time.perf_counter() - stage
        )

    except Exception:
        model_manager.release_search()
        raise

    elapsed = time.perf_counter() - search_started
    search_id = uuid.uuid4().hex
    performance_registry.record_search({
        "search_id": search_id,
        "video_id": video_id,
        "query": query,
        "candidate_id": best["candidate_id"],
        "latency_sec": round(elapsed, 4),
        "confidence": round(best["final_confidence"], 4),
        "relation": plan.temporal_relation,
        "relation_satisfied": refined.get(
            "relation_satisfied",
            True,
        ),
        "stage_latencies": {
            key: round(value, 4)
            for key, value in stages.items()
        },
        "device_assignment": model_manager.devices,
        **performance_registry.memory_snapshot(),
    })
    return SearchExecution(
        search_id,
        best,
        refined,
        plan,
        stages,
    )


def _strict(execution):
    return StrictSearchResponse(
        start_timestamp=execution.refined["formatted_start"],
        end_timestamp=execution.refined["formatted_end"],
        confidence_score=round(
            execution.best["final_confidence"],
            4,
        ),
    )


def _detailed(execution):
    candidate = execution.best
    refined = execution.refined
    plan = execution.plan
    payload = candidate.get("payload", {})
    return DetailedSearchResponse(
        start_timestamp=refined["formatted_start"],
        end_timestamp=refined["formatted_end"],
        confidence_score=round(
            candidate["final_confidence"],
            4,
        ),
        search_id=execution.search_id,
        candidate_id=candidate["candidate_id"],
        video_id=payload.get("video_id", ""),
        clip_id=payload.get("clip_id", ""),
        speech_excerpt=payload.get("transcript", ""),
        ocr_text=payload.get("ocr_text", ""),
        retrieval_method=(
            "coarse_clip_rrf+fine_frame+"
            + str(
                refined.get(
                    "spatial_evidence",
                    {},
                ).get(
                    "evidence_type",
                    "none",
                )
            )
        ),
        temporal_relation=plan.temporal_relation,
        relation_satisfied=refined.get(
            "relation_satisfied",
            True,
        ),
        modality_spans=refined.get("modality_spans", {}),
        spatial_evidence=refined.get(
            "spatial_evidence",
            {},
        ),
        scores=candidate.get("score_breakdown", {}),
        stage_latencies={
            key: round(value, 4)
            for key, value in execution.stage_latencies.items()
        },
    )


@app.post("/search", response_model=StrictSearchResponse)
def search_video_segment(request: SearchQueryRequest):
    return _strict(execute_search_once(request))


@app.post(
    "/search/detailed",
    response_model=DetailedSearchResponse,
)
def search_video_detailed(request: SearchQueryRequest):
    return _detailed(execute_search_once(request))


def search_video_both(request):
    execution = execute_search_once(request)
    return _strict(execution), _detailed(execution)


@app.get("/metrics")
def metrics():
    return performance_registry.summary()


@app.get("/metrics/report", response_class=PlainTextResponse)
def metrics_report():
    return performance_registry.arabic_markdown()
