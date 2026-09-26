# Implementation Audit

## Scope and baseline

This audit records the state of `BXR7/spatio-temporal-video-rag` before source extraction. The starting branch contained five tracked files: `README.md`, one notebook, two DOCX research/design reports, and `docker_in_100s.mp4`. No Python package tree, dependency specification, tests, `.env.example`, `.gitignore`, license, or CI configuration was tracked.

The notebook `spatio-temporal-multimodal-vidoe-rag.ipynb` contains 39 cells: 36 code cells and 3 Markdown cells. It is the implementation source of truth for this upgrade and contains 22 `%%writefile` cells.

## Notebook-generated modules

The following modules are generated in notebook execution order. The source is extracted verbatim from the corresponding `%%writefile` body unless a later documented importability fix is required.

| Target | Source cell | Role |
|---|---:|---|
| `ingestion/scene_detector.py` | 3 | Scene detection and fallback windows |
| `ingestion/video_processor.py` | 4 | Video metadata, audio, keyframes, clip preparation |
| `ingestion/transcriber.py` | 5 | Whisper transcription and word timestamps |
| `ingestion/ocr_extractor.py` | 6 | OCR extraction and spatial regions |
| `ingestion/embedder.py` | 7 | Text and visual embeddings |
| `ingestion/indexer.py` | 8 | Qdrant/local vector indexing and retrieval |
| `search/device_detector.py` | 9 | Device and CUDA lifecycle helpers |
| `search/query_understanding.py` | 10 | Query parsing and optional LLM planning |
| `search/retrieval_planner.py` | 11 | Retrieval-plan construction |
| `search/temporal_reasoner.py` | 12 | Temporal relation reasoning |
| `search/query_planner.py` | 13 | Query-understanding/retrieval-plan composition |
| `search/retriever.py` | 14 | Multimodal candidate retrieval and RRF |
| `search/reranker.py` | 15 | BGE reranking |
| `search/verification.py` | 16 | OCR/visual verification |
| `search/refiner.py` | 17 | Timestamp refinement |
| `search/confidence.py` | 18 | Heuristic confidence calculation |
| `storage/metadata_store.py` | 19 | SQLite temporal and spatial metadata |
| `api/schemas.py` | 20 | Pydantic request/response schemas |
| `monitoring/performance.py` | 21 | Runtime performance registry |
| `evaluation/metrics.py` | 22 | Deterministic interval metrics |
| `runtime/model_manager.py` | 23 | Lazy loading and model release |
| `api/main.py` | 24 | FastAPI application and orchestration |

The notebook must currently run the writefile cells before importing `api.main`.

## Dependencies and environment assumptions

The implementation assumes an already prepared Linux environment. No notebook cell installs dependencies. The code imports FastAPI/Pydantic, Uvicorn, PyTorch, Transformers, Sentence Transformers, Qdrant client, OpenCV, NumPy, Pillow, EasyOCR, faster-whisper, PySceneDetect, Accelerate, bitsandbytes, and related model tooling. FFmpeg is invoked as a system executable to create mono 16 kHz PCM audio.

Models are downloaded lazily from Hugging Face, including Whisper, BGE-M3, SigLIP2, BGE reranker, and optional Qwen planner/verification models. The demonstrated configuration assigns ingestion/OCR to `cuda:0` and search/model-heavy work to `cuda:1`; several notebook validation cells assert CUDA availability. Full CPU or single-GPU end-to-end support was not independently verified.

Documented runtime variables include `PYTORCH_CUDA_ALLOC_CONF`, `TOKENIZERS_PARALLELISM`, `VIDEO_RAG_OCR_DEVICE`, `VIDEO_RAG_OCR_LANGS`, `VIDEO_RAG_OCR_FRAMES_PER_CLIP`, `VIDEO_RAG_OCR_CANVAS_SIZE`, `VIDEO_RAG_QDRANT_PATH`, `VIDEO_RAG_INGEST_GPU`, `VIDEO_RAG_SEARCH_GPU`, `VIDEO_RAG_USE_LLM_PLANNER`, `VIDEO_RAG_VIDEO_ID`, and `VIDEO_RAG_VIDEO_PATH`.

## Storage and state

Qdrant uses persistent local storage under `storage/qdrant_db` with clip/frame collections. The indexer also contains JSON/local fallback paths under `storage/`. SQLite defaults to `storage/temporal_metadata.db` and stores ASR word timestamps and OCR regions. Processed audio, keyframes, metrics, and model caches are runtime artifacts and must not be committed.

The notebook defaults to a Kaggle video path and has a Colab upload fallback. Outside those environments, `VIDEO_RAG_VIDEO_PATH` must point to an existing local file; the tracked MP4 is the available fixture. Relative storage paths assume the repository root is the current working directory.

## Entry points and evidence

The extracted API defines `GET /runtime`, `POST /runtime/release`, `POST /ingest`, `POST /search`, `GET /metrics`, and `GET /metrics/report`. The notebook exercises ingestion, strict and detailed search helpers, API contract checks, model lifecycle checks, and positive/negative temporal relations after ingestion completes.

The notebook's evidence is functional validation on one English video: ingestion/index creation, OCR-grounded queries, a positive `while` relation, a negative unsatisfied relation returning HTTP 404, schema checks, and model cleanup. Existing GPU snapshots are evidence for one run; they are not a general capacity or performance benchmark. Existing retrieval intervals and confidence values must not be used as ground truth.

## Reproducibility blockers

1. A fresh clone cannot import `api.main` until notebook generation cells have run.
2. Dependencies and PyTorch/CUDA are not specified in an installable project configuration.
3. Model downloads are mutable and revision/cache requirements are undocumented.
4. The default video path is environment-specific.
5. Two-device CUDA assumptions are not verified on this host.
6. Relative Qdrant/SQLite paths and runtime artifacts have no ignore policy.
7. The README references a different notebook filename and a `/search/detailed` route that must be checked against source.
8. No independent ground-truth annotations exist, so Temporal IoU, Recall@K, MRR, nDCG, calibration, and accuracy claims remain unavailable.
9. No license is present; no license will be added without owner approval.

## Extraction risks and policy

Source extraction preserves notebook behavior. Importability fixes are limited to package markers, path/configuration hygiene, and testable deterministic seams. No model, retrieval, confidence, or temporal algorithm is being rewritten without evidence. GPU/model-dependent execution will be reported separately from locally verified syntax, deterministic utilities, and schema checks.
