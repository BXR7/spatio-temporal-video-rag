# Demo

## Status

`assets/demo.gif` is intentionally **pending**. The available sandbox does not have the heavy model/runtime dependencies or CUDA devices required to execute ingestion and search, so a truthful recording cannot be generated here. No fake UI, response, timestamp, or animation is included.

## Prerequisites

- Linux with FFmpeg.
- Python 3.10–3.12 and the dependencies in `pyproject.toml`.
- A CUDA-capable NVIDIA environment for the full model path; the demonstrated notebook configuration uses two device assignments.
- Hugging Face access and sufficient model/cache storage.

## Exact workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# Install a PyTorch build matching the host CUDA version.
export VIDEO_RAG_VIDEO_PATH="$PWD/docker_in_100s.mp4"
export VIDEO_RAG_VIDEO_ID=demo_video
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second shell, ingest the fixture:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"./docker_in_100s.mp4","video_id":"demo_video"}'
```

Then query a real OCR moment:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where does EXPOSE 8080 first appear in the Dockerfile?","video_id":"demo_video"}'
```

The expected shape is the verified `StrictSearchResponse` schema. Exact values must be recorded from the run, not copied from notebook examples.

## Recording instructions

Record: server ready → ingest request/result → query request → strict timestamp response → optional `/search/detailed` evidence → source-video seek to the returned interval. Label the recording as fixture-specific functional validation, not an accuracy benchmark. Keep the raw command output and environment metadata with the recording.
