"""Persistent ingestion/search latency and RAM/VRAM instrumentation."""
import json
import os
import time
from contextlib import contextmanager

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


class PerformanceRegistry:
    def __init__(self, path="storage/performance_metrics.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                pass
        return {"ingestions": [], "searches": []}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def memory_snapshot():
        rss_mb = 0.0
        if psutil is not None:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        gpu_memory = {}
        total_allocated = 0.0
        if torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(index) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(index) / (1024 ** 2)
                gpu_memory[f"gpu_{index}"] = {
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                }
                total_allocated += allocated
        return {
            "ram_mb": round(rss_mb, 2),
            "vram_mb": round(total_allocated, 2),
            "gpu_memory": gpu_memory,
        }

    @contextmanager
    def stage(self, stage_metrics, name):
        before = self.memory_snapshot()
        if torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            after = self.memory_snapshot()
            peak_by_gpu = {}
            peak_vram = 0.0
            if torch is not None and torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    peak = torch.cuda.max_memory_allocated(index) / (1024 ** 2)
                    peak_by_gpu[f"gpu_{index}"] = round(peak, 2)
                    peak_vram += peak
            stage_metrics[name] = {
                "latency_sec": round(duration, 4),
                "ram_before_mb": before["ram_mb"],
                "ram_after_mb": after["ram_mb"],
                "max_observed_ram_mb": max(
                    before["ram_mb"],
                    after["ram_mb"],
                ),
                "vram_after_mb": after["vram_mb"],
                "peak_vram_mb": round(peak_vram, 2),
                "peak_vram_by_gpu_mb": peak_by_gpu,
            }

    def record_ingestion(self, record):
        self.data["ingestions"].append(record)
        self.data["ingestions"] = self.data["ingestions"][-50:]
        self._save()

    def record_search(self, record):
        self.data["searches"].append(record)
        self.data["searches"] = self.data["searches"][-500:]
        self._save()

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
        return ordered[index]

    def summary(self):
        latencies = [
            float(item.get("latency_sec", 0.0))
            for item in self.data["searches"]
        ]
        enough_samples = len(latencies) >= 5

        return {
            "ingestion_runs": len(self.data["ingestions"]),
            "search_runs": len(self.data["searches"]),
            "search_latency_sample_size": len(latencies),
            "search_latency_p50_sec": (
                round(self._percentile(latencies, 50), 4)
                if enough_samples
                else None
            ),
            "search_latency_p95_sec": (
                round(self._percentile(latencies, 95), 4)
                if enough_samples
                else None
            ),
            "search_latency_max_sec": (
                round(max(latencies), 4)
                if latencies
                else None
            ),
            "latest_ingestion": (
                self.data["ingestions"][-1]
                if self.data["ingestions"]
                else {}
            ),
            "latest_search": (
                self.data["searches"][-1]
                if self.data["searches"]
                else {}
            ),
        }

    def arabic_markdown(self):
        summary = self.summary()
        latest_ingestion = summary["latest_ingestion"]
        return f"""# تقرير أداء Spatio-Temporal Video RAG

| المقياس | القيمة |
|---|---:|
| عدد مرات الفهرسة المقاسة | {summary['ingestion_runs']} |
| عدد استعلامات البحث المقاسة | {summary['search_runs']} |
| Search latency P50 | {summary['search_latency_p50_sec']} ثانية |
| Search latency P95 | {summary['search_latency_p95_sec']} ثانية |
| أعلى Search latency | {summary['search_latency_max_sec']} ثانية |
| آخر عدد Clips | {latest_ingestion.get('total_clips', 0)} |
| آخر عدد Frames | {latest_ingestion.get('total_frames', 0)} |
| مدة آخر فيديو | {latest_ingestion.get('video_duration_sec', 0)} ثانية |
| زمن آخر فهرسة | {latest_ingestion.get('latency_sec', 0)} ثانية |
| Peak VRAM للفهرسة | {latest_ingestion.get('peak_vram_mb', 0)} MB |
| أعلى RAM مرصود للفهرسة | {latest_ingestion.get('max_observed_ram_mb', 0)} MB |

## آخر فهرسة

```json
{json.dumps(latest_ingestion, ensure_ascii=False, indent=2)}
```

## آخر بحث

```json
{json.dumps(summary['latest_search'], ensure_ascii=False, indent=2)}
```
"""
