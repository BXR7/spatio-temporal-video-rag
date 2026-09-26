# Spatio-Temporal Multimodal Video RAG

A research prototype that retrieves precise moments from video by combining **speech, OCR, visual embeddings, spatial evidence, and temporal relations**.

> **Status:** the implementation is extracted into committed Python packages, and deterministic components are locally testable. Full ingestion/search requires heavy ML dependencies, model downloads, FFmpeg, and compatible CUDA hardware. Current evidence is fixture-specific functional validation, not a retrieval-accuracy benchmark.

## What it does

The system can answer queries such as:

```text
Where does EXPOSE 8080 first appear in the Dockerfile?
Show screen text PORT FORWARDING while the speaker says port forwarding.
```

It returns a strict interval response (`start_timestamp`, `end_timestamp`, `confidence_score`) or a detailed response with modality spans, OCR/spatial evidence, scores, and stage timings.

## Architecture

```mermaid
flowchart LR
  A[Video] --> B[Scenes + ASR boundaries]
  B --> C[Keyframes / Whisper / OCR / embeddings]
  C --> D[(Qdrant + SQLite)]
  D --> E[Query planning + multimodal retrieval]
  E --> F[RRF + reranking + verification]
  F --> G[Temporal reasoning + confidence]
  G --> H[FastAPI result]
```

See the [verified architecture](docs/architecture.md) for component boundaries.

## Quick start

### Development and deterministic tests

This path installs project dependencies but does not download model weights:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

### Full model-backed runtime

On a fresh environment, use this order. Install PyTorch first using the official selector for the host's CUDA version; no universal CUDA command is assumed here.

```text
1. Create and activate the virtual environment as shown above.
2. Install PyTorch using https://pytorch.org/get-started/locally/.
3. Install the project dependencies once:
   pip install -e '.[dev]'
4. Copy and configure the environment:
   cp .env.example .env
   export VIDEO_RAG_VIDEO_PATH="$PWD/docker_in_100s.mp4"
5. Start the API:
   uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another shell:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"./docker_in_100s.mp4","video_id":"demo_video"}'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where does EXPOSE 8080 first appear in the Dockerfile?","video_id":"demo_video"}'
```

The full workflow and blocker status are in [docs/DEMO.md](docs/DEMO.md). The tracked fixture is an 11-minute, 640×360 H.264/AAC video despite its historical `docker_in_100s.mp4` filename.

## Verified API surface

- `POST /ingest` — local video path, returns `IngestResponse`.
- `POST /search` — strict `StrictSearchResponse`.
- `POST /search/detailed` — `DetailedSearchResponse` with evidence fields.
- `GET /runtime` and `POST /runtime/release` — model lifecycle status/control.
- `GET /metrics` and `GET /metrics/report` — performance registry output.

Request models are defined in [`api/schemas.py`](api/schemas.py). Model-backed route execution is not verified in this CPU-only sandbox.

## Evaluation

The repository preserves notebook evidence of ingestion/indexing, OCR-grounded queries, positive/negative temporal relations, schema checks, and model release behavior on one English video. These outputs are not labels. No independent annotations exist, so this branch reports no Temporal IoU, Recall@K, MRR, nDCG, accuracy, calibration, or general latency claim.

Run deterministic tests:

```bash
pytest -q
```

See [evaluation/README.md](evaluation/README.md) and [evaluation/ground_truth.example.json](evaluation/ground_truth.example.json).

## Repository structure

```text
api/          FastAPI app and schemas
ingestion/    video, ASR, OCR, embedding, and indexing
tests/        deterministic unit tests
search/       parsing, planning, retrieval, reranking, verification
evaluation/   interval utilities and protocol
storage/      SQLite metadata store
runtime/      lazy model lifecycle
monitoring/   performance registry
docs/         audit, architecture, demo, evaluation, release checklist
notebooks/    source notebook
```

The Python modules are extracted from the notebook's `%%writefile` cells; [docs/IMPLEMENTATION_AUDIT.md](docs/IMPLEMENTATION_AUDIT.md) maps every module to its source cell.

## Limitations and release status

Full runtime depends on large Hugging Face models, FFmpeg, and CUDA-capable hardware; CPU and single-GPU end-to-end support remain unverified. Local Qdrant/SQLite indexes and processed media are runtime artifacts and are ignored. The heuristic confidence score is not a calibrated probability. No license is present; see [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) before any public release. `assets/demo.gif` is pending because no truthful model-backed execution was available in the current sandbox.

## Notebook and reports

The notebook remains as an experimentation and provenance artifact in [`notebooks/spatio-temporal-multimodal-video-rag.ipynb`](notebooks/spatio-temporal-multimodal-video-rag.ipynb). The original research documents are under [`docs/reports/`](docs/reports/).
