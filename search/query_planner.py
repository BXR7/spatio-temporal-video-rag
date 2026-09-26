"""Query planner whose Qwen model is loaded only for the planning stage."""
from typing import Optional
from .query_understanding import FastQueryUnderstander
from .retrieval_planner import RetrievalPlanner


class QueryPlanner:
    def __init__(self, device: Optional[str] = None):
        self.understander = FastQueryUnderstander(
            device=device,
            autoload=False,
        )
        self.retrieval_planner = RetrievalPlanner()

    def plan(self, query):
        intent = self.understander.analyze(query)
        active = [m for m, ok in (
            ("visual", intent.needs_visual),
            ("speech", intent.needs_speech),
            ("ocr", intent.needs_ocr),
        ) if ok]
        print(
            f"[QueryPlanner] intent={intent.intent} "
            f"selector={intent.temporal_selector} "
            f"relation={intent.temporal_relation} "
            f"modalities={active}"
        )
        print(
            f"[QueryPlanner] visual={intent.visual_query!r} "
            f"speech={intent.speech_query!r} "
            f"ocr={intent.ocr_query!r}"
        )
        return self.retrieval_planner.plan(
            intent,
            intent.expanded_queries,
        )

    def release_model(self):
        self.understander.release_model()
