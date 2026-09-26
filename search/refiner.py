"""Clause-aware temporal algebra for speech, OCR, and visual intervals."""
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional

Span = Optional[Tuple[float, float]]


class TimestampRefiner:
    EXPANSION_MARGIN = 0.35

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis == 1000:
            secs += 1
            millis = 0
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def _tokens(text: str):
        stop = {
            "where", "when", "what", "the", "a", "an", "is", "are", "in", "on", "at",
            "show", "find", "moment", "متى", "أين", "في", "على", "التي", "الذي", "لحظة",
        }
        return [
            token for token in re.findall(r"[\w.-]+", text.casefold())
            if len(token) > 1 and token not in stop
        ]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    def find_speech_span(self, words: List[Dict[str, Any]], clause: str) -> Span:
        query_tokens = self._tokens(clause)
        if not words or not query_tokens:
            return None

        matched = []
        for index, word in enumerate(words):
            token = str(word.get("word", "")).strip(".,;:!?()[]{}\"'").casefold()
            similarity = max((self._similarity(token, query) for query in query_tokens), default=0.0)
            if similarity >= 0.72:
                matched.append((index, similarity))
        if not matched:
            return None

        best_window = None
        for start_pos in range(len(matched)):
            seen = set()
            for end_pos in range(start_pos, min(len(matched), start_pos + len(query_tokens) + 5)):
                word_index, _ = matched[end_pos]
                token = str(words[word_index].get("word", "")).casefold()
                for query_index, query in enumerate(query_tokens):
                    if self._similarity(token, query) >= 0.72:
                        seen.add(query_index)
                coverage = len(seen) / max(len(query_tokens), 1)
                if coverage >= min(0.5, 2 / max(len(query_tokens), 1)):
                    start_index = matched[start_pos][0]
                    end_index = word_index
                    span = (float(words[start_index]["start"]), float(words[end_index]["end"]))
                    score = coverage - 0.01 * (span[1] - span[0])
                    if best_window is None or score > best_window[0]:
                        best_window = (score, span)
        if best_window:
            return best_window[1]

        indices = [item[0] for item in matched]
        return float(words[min(indices)]["start"]), float(words[max(indices)]["end"])

    def find_ocr_span(self, observations, clause: str) -> Span:
        query_tokens = set(self._tokens(clause))
        if not observations or not query_tokens:
            return None
        matched_timestamps = []
        for observation in observations:
            text_tokens = set(self._tokens(observation.get("text", "")))
            if not text_tokens:
                continue
            coverage = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            sequence = SequenceMatcher(
                None, " ".join(sorted(query_tokens)), " ".join(sorted(text_tokens))
            ).ratio()
            if max(coverage, sequence) >= 0.35:
                matched_timestamps.append(float(observation.get("timestamp", 0.0)))
        if not matched_timestamps:
            return None
        return min(matched_timestamps), max(matched_timestamps) + 0.75

    @staticmethod
    def visual_span(candidate) -> Span:
        grounding = candidate.get("grounding", {}) or {}
        if grounding.get("match") and "start_ts" in grounding and "end_ts" in grounding:
            return float(grounding["start_ts"]), float(grounding["end_ts"])
        evidence = candidate.get("frame_evidence", [])
        timestamps = [
            float(item["timestamp"])
            for item in evidence
            if item.get("modality") == "visual" and item.get("score", 0.0) >= 0.2
        ]
        return (min(timestamps), max(timestamps) + 0.75) if timestamps else None

    @staticmethod
    def intersect(*spans: Span) -> Span:
        present = [span for span in spans if span is not None]
        if not present:
            return None
        start = max(span[0] for span in present)
        end = min(span[1] for span in present)
        return (start, end) if start < end else None

    @staticmethod
    def union(*spans: Span) -> Span:
        present = [span for span in spans if span is not None]
        if not present:
            return None
        return min(span[0] for span in present), max(span[1] for span in present)

    @staticmethod
    def iou(a: Span, b: Span) -> float:
        if a is None or b is None:
            return 0.0
        intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return intersection / union if union > 0 else 0.0

    def refine_timestamps(self, candidate: Dict[str, Any], plan) -> Dict[str, Any]:
        payload = candidate.get("payload", {}) or {}
        clip_span = (
            float(payload.get("start_ts", 0.0)),
            float(payload.get("end_ts", 0.0)),
        )
        spans = {
            "speech": self.find_speech_span(
                payload.get("word_timestamps", []),
                plan.query_for("speech"),
            ) if "speech" in plan.modalities else None,
            "ocr": self.find_ocr_span(
                payload.get("ocr_observations", []),
                plan.query_for("ocr"),
            ) if "ocr" in plan.modalities else None,
            "visual": self.visual_span(candidate)
            if "visual" in plan.modalities
            else None,
        }

        grounding = candidate.get("grounding", {}) or {}
        exact_ocr_grounding = bool(
            grounding.get("match")
            and grounding.get("evidence_type") == "ocr_region"
            and grounding.get("exact_target_match")
            and "start_ts" in grounding
            and "end_ts" in grounding
        )
        if exact_ocr_grounding:
            spans["ocr"] = (
                float(grounding["start_ts"]),
                float(grounding["end_ts"]),
            )

        relation = plan.temporal_relation
        left = spans.get(plan.relation_left_modality)
        right = spans.get(plan.relation_right_modality)
        relation_satisfied = True
        relation_score = 1.0

        if relation == "while":
            refined = self.intersect(left, right)
            relation_satisfied = refined is not None and left is not None and right is not None
            relation_score = self.iou(left, right) if relation_satisfied else 0.0
        elif relation == "before":
            relation_satisfied = bool(left and right and left[1] <= right[0] + 0.25)
            refined = left
            gap = max(0.0, right[0] - left[1]) if left and right else 99.0
            relation_score = 1.0 / (1.0 + gap) if relation_satisfied else 0.0
        elif relation == "after":
            relation_satisfied = bool(left and right and left[0] >= right[1] - 0.25)
            refined = left
            gap = max(0.0, left[0] - right[1]) if left and right else 99.0
            relation_score = 1.0 / (1.0 + gap) if relation_satisfied else 0.0
        else:
            primary = spans.get(plan.primary_modality)
            refined = primary or self.intersect(*[spans[m] for m in plan.modalities])
            if refined is None:
                refined = self.union(*[spans[m] for m in plan.modalities])
            available = [span for span in spans.values() if span is not None]
            relation_score = 0.85 if available else 0.35

        if refined is None:
            refined = clip_span
            relation_satisfied = relation == "none"
            relation_score = 0.25 if relation == "none" else 0.0

        start = max(clip_span[0], refined[0] - self.EXPANSION_MARGIN)
        end = min(clip_span[1], refined[1] + self.EXPANSION_MARGIN)
        if end <= start:
            end = min(clip_span[1], start + 0.5)

        return {
            "refined_start": round(start, 3),
            "refined_end": round(end, 3),
            "formatted_start": self.format_timestamp(start),
            "formatted_end": self.format_timestamp(end),
            "temporal_score": max(0.0, min(relation_score, 1.0)),
            "relation_satisfied": relation_satisfied,
            "temporal_relation": relation,
            "modality_spans": {
                key: list(value) if value is not None else None for key, value in spans.items()
            },
            "spatial_evidence": candidate.get("grounding", {}),
        }
