from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class RetrievalPlan:
    queries: List[str]
    modalities: List[str]
    modality_weights: Dict[str, float]
    queries_by_modality: Dict[str, List[str]] = field(default_factory=dict)
    fusion: str = "rrf"
    rerank: bool = True
    verify: bool = True
    temporal_selector: str = "any"
    temporal_relation: str = "none"
    primary_modality: str = "visual"
    relation_left_modality: str = "visual"
    relation_right_modality: str = "speech"
    requires_spatial_grounding: bool = True
    original_query: str = ""

    @property
    def active_modalities(self): return self.modalities
    @property
    def temporal_intent(self): return self.temporal_selector
    def query_for(self, modality):
        values = self.queries_by_modality.get(modality) or self.queries
        return values[0] if values else self.original_query

class RetrievalPlanner:
    def plan(self, intent: Any, rewritten_queries: List[str]):
        valid_m = {"visual","speech","ocr"}
        relation = intent.temporal_relation if intent.temporal_relation in {"while","before","after","none"} else "none"
        selector = intent.temporal_selector if intent.temporal_selector in {"first","last","any"} else "any"
        left = intent.relation_left_modality if intent.relation_left_modality in valid_m else "visual"
        right = intent.relation_right_modality if intent.relation_right_modality in valid_m else "speech"
        modalities = [m for m, ok in (("visual",intent.needs_visual),("speech",intent.needs_speech),("ocr",intent.needs_ocr)) if ok]
        if relation != "none":
            for m in (left, right):
                if m not in modalities: modalities.append(m)
        if not modalities: modalities = ["visual","speech","ocr"]

        raw = intent.modality_weights or {}
        weights = {}
        for m in modalities:
            try: weights[m] = max(float(raw.get(m,0)),0)
            except Exception: weights[m] = 0
        if sum(weights.values()) <= 0: weights = {m:1 for m in modalities}
        total = sum(weights.values())
        weights = {k:v/total for k,v in weights.items()}

        qbm = {}
        for m, attr in (("visual","visual_query"),("speech","speech_query"),("ocr","ocr_query")):
            clause = str(getattr(intent, attr, "") or "").strip()
            qbm[m] = list(dict.fromkeys([x for x in ([clause] if clause else []) + list(rewritten_queries) if x]))[:3]

        return RetrievalPlan(
            queries=list(dict.fromkeys(rewritten_queries)), modalities=modalities,
            modality_weights=weights, queries_by_modality=qbm,
            fusion="rrf" if len(modalities)>1 else "single",
            temporal_selector=selector, temporal_relation=relation,
            primary_modality=intent.primary_modality if intent.primary_modality in valid_m else modalities[0],
            relation_left_modality=left, relation_right_modality=right,
            requires_spatial_grounding=bool(intent.requires_spatial_grounding),
            original_query=intent.original_query,
        )
