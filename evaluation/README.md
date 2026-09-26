# Evaluation

## Current status

The repository contains **functional validation evidence**, not a retrieval-quality benchmark. The notebook exercised ingestion, indexing, OCR-grounded searches, a positive `while` relation, a negative unsatisfied relation, API schemas, and model release behavior on one English video.

The observed intervals and confidence scores are outputs of the system. They are **not ground truth** and must not be used as labels.

## Protocol

A future retrieval-quality evaluation should use a held-out set of videos and independently annotated query intervals. Each annotation should record the video ID, query, relevant interval(s), relation constraints, and annotator/version metadata. Report the number of videos, queries, and relevant intervals, plus exact matching rules and uncertainty.

Available deterministic utilities are in `evaluation/metrics.py`: temporal IoU and recall at a chosen IoU threshold. Recall@K, MRR, nDCG, calibration, and cross-video robustness are not implemented or evidenced by the current experiment.

## Evidence classification

- **Verified:** source compilation; deterministic unit tests; schema construction; interval utility behavior.
- **Partially verified:** notebook-level functional ingestion/search/API checks on the tracked fixture, subject to model/GPU availability.
- **Not verified:** general retrieval accuracy, latency distributions, throughput, CPU support, single-GPU end-to-end execution, and quality metrics.

Do not call the heuristic confidence score a probability. Do not report Temporal IoU or retrieval-quality metrics until independent annotations exist.
