"""Lazy Qwen2.5-VL grounding pinned to the search GPU."""
import gc
import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

try:
    import torch
    from PIL import Image
except Exception:
    torch = None

from search.device_detector import configured_devices, release_cuda


class VideoVerificationLayer:
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        enable: bool = True,
        device: Optional[str] = None,
        autoload: bool = False,
    ):
        self.enable = bool(enable)
        self.device = device or configured_devices()["search"]
        self.model = None
        self.processor = None
        if autoload and self.enable:
            self.load_model()

    def load_model(self):
        if not self.enable or self.model is not None:
            return self
        if torch is None or not self.device.startswith("cuda"):
            return self
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"[VerificationLayer] Loading Qwen2.5-VL on {self.device}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                quantization_config=quantization,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME,
                use_fast=False,
            )
            print("[VerificationLayer]  Loaded")
        except Exception as exc:
            print(f"[VerificationLayer] Indexed-evidence fallback: {exc}")
            self.model = None
        return self

    def release_model(self):
        for name in ("model", "processor"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
            setattr(self, name, None)
        gc.collect()
        release_cuda(self.device)
        print(f"[VerificationLayer]  Released from {self.device}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = str(text).casefold()
        normalized = re.sub(r"[_:/=]+", " ", normalized)
        normalized = re.sub(r"[-]+", " ", normalized)
        normalized = " ".join(
            re.findall(r"[a-z0-9.]+", normalized)
        )
        return normalized.strip()

    @classmethod
    def _extract_ocr_targets(cls, query: str):
        """Return OCR targets ordered by semantic specificity.

        Examples:
        - EXPOSE 8080 -> priority 100
        - Dockerfile -> priority 60
        - 8080 -> priority 20

        Lower-priority numeric targets are never allowed to override a richer
        command-plus-number target present in the same query.
        """
        raw_query = str(query)
        targets = []

        # Highest priority: command/identifier followed by a number.
        for match in re.finditer(
            r"\b([A-Z][A-Z0-9_-]{2,})\s+([:=/-]?\s*\d+(?:[.:/-]\d+)*)",
            raw_query,
        ):
            text = cls._normalize_text(
                f"{match.group(1)} {match.group(2)}"
            )
            if text:
                targets.append({
                    "text": text,
                    "priority": 100,
                    "kind": "command_number",
                })

        # Quoted code/text phrases.
        for match in re.finditer(
            r'["\']([^"\']{2,})["\']',
            raw_query,
        ):
            text = cls._normalize_text(match.group(1))
            if text:
                targets.append({
                    "text": text,
                    "priority": 80,
                    "kind": "quoted_phrase",
                })

        # File names and common technical identifiers.
        for match in re.finditer(
            r"\b(?:Dockerfile|Makefile|[A-Za-z0-9_-]+\.(?:py|js|ts|json|yaml|yml|sh|txt|md|html|css))\b",
            raw_query,
            flags=re.IGNORECASE,
        ):
            text = cls._normalize_text(match.group(0))
            if text:
                targets.append({
                    "text": text,
                    "priority": 60,
                    "kind": "filename",
                })

        # Bare uppercase identifiers, but avoid duplicating richer targets.
        for match in re.finditer(
            r"\b[A-Z][A-Z0-9_-]{2,}\b",
            raw_query,
        ):
            text = cls._normalize_text(match.group(0))
            if text:
                targets.append({
                    "text": text,
                    "priority": 50,
                    "kind": "identifier",
                })

        # Lowest priority: standalone numbers.
        for number in re.findall(r"\b\d{2,}\b", raw_query):
            targets.append({
                "text": cls._normalize_text(number),
                "priority": 20,
                "kind": "number",
            })

        # Deduplicate while preserving the highest priority.
        best = {}
        for target in targets:
            text = target["text"]
            if (
                text not in best
                or target["priority"] > best[text]["priority"]
            ):
                best[text] = target

        return sorted(
            best.values(),
            key=lambda item: (
                item["priority"],
                len(item["text"]),
            ),
            reverse=True,
        )

    @classmethod
    def _match_ocr_region(cls, query: str, region_text: str):
        query_norm = cls._normalize_text(query)
        region_norm = cls._normalize_text(region_text)
        if not query_norm or not region_norm:
            return {
                "score": 0.0,
                "matched_target": "",
                "target_priority": 0,
                "target_kind": "none",
                "exact": False,
            }

        targets = cls._extract_ocr_targets(query)
        if targets:
            highest_priority = targets[0]["priority"]
            primary_targets = [
                target
                for target in targets
                if target["priority"] == highest_priority
            ]

            # Exact match of the highest-specificity target.
            for target in primary_targets:
                if target["text"] in region_norm:
                    return {
                        "score": 1.0,
                        "matched_target": target["text"],
                        "target_priority": target["priority"],
                        "target_kind": target["kind"],
                        "exact": True,
                    }

            # Partial matching is allowed for diagnostics/retrieval, but is
            # capped below the authoritative OCR-match threshold.
            best_partial = 0.0
            best_target = primary_targets[0]
            region_tokens = set(region_norm.split())
            for target in primary_targets:
                target_tokens = set(target["text"].split())
                coverage = len(
                    target_tokens & region_tokens
                ) / max(len(target_tokens), 1)
                sequence = SequenceMatcher(
                    None,
                    target["text"],
                    region_norm,
                ).ratio()
                partial = max(
                    0.70 * coverage + 0.30 * sequence,
                    sequence,
                )
                if partial > best_partial:
                    best_partial = partial
                    best_target = target

            return {
                "score": min(best_partial, 0.49),
                "matched_target": best_target["text"],
                "target_priority": best_target["priority"],
                "target_kind": best_target["kind"],
                "exact": False,
            }

        # Generic fallback only when the query has no structured OCR target.
        stop_words = {
            "where", "when", "what", "which", "is", "are", "was", "were",
            "the", "a", "an", "in", "on", "at", "to", "of", "for",
            "written", "write", "appear", "appears", "display", "displays",
            "show", "shows", "find", "line", "file",
            "أين", "متى", "ما", "في", "على", "الذي", "التي", "مكتوب", "يظهر",
        }
        query_tokens = {
            token
            for token in query_norm.split()
            if token not in stop_words and len(token) > 1
        }
        region_tokens = set(region_norm.split())
        if not query_tokens or not region_tokens:
            score = SequenceMatcher(
                None,
                query_norm,
                region_norm,
            ).ratio()
        else:
            intersection = query_tokens & region_tokens
            query_coverage = len(
                intersection
            ) / max(len(query_tokens), 1)
            region_precision = len(
                intersection
            ) / max(len(region_tokens), 1)
            score = max(
                SequenceMatcher(
                    None,
                    query_norm,
                    region_norm,
                ).ratio(),
                0.65 * region_precision
                + 0.35 * query_coverage,
            )

        return {
            "score": score,
            "matched_target": "",
            "target_priority": 0,
            "target_kind": "generic",
            "exact": False,
        }

    @classmethod
    def _token_similarity(cls, query: str, text: str) -> float:
        return float(
            cls._match_ocr_region(query, text)["score"]
        )

    def _ocr_grounding(self, plan, candidate):
        query = plan.query_for("ocr")
        observations = candidate.get("payload", {}).get(
            "ocr_observations", []
        )
        best = None

        for observation in observations:
            for region in observation.get("regions", []):
                region_text = region.get("text", "")
                match_info = self._match_ocr_region(
                    query,
                    region_text,
                )
                semantic_score = float(
                    match_info["score"]
                )
                detector_score = max(
                    0.0,
                    min(
                        float(region.get("confidence", 0.0)),
                        1.0,
                    ),
                )

                combined_score = min(
                    1.0,
                    0.90 * semantic_score
                    + 0.10 * detector_score,
                )
                bbox_norm = region.get("bbox_norm", []) or []
                bbox = region.get("bbox", []) or []

                # A structured target is authoritative only when the
                # highest-specificity phrase itself matched exactly.
                exact_target_match = bool(
                    match_info["exact"]
                    and match_info["target_priority"] >= 50
                )
                generic_match = bool(
                    match_info["target_kind"] == "generic"
                    and semantic_score >= 0.70
                )
                match = bool(
                    (exact_target_match or generic_match)
                    and (bbox_norm or bbox)
                )

                item = {
                    "match": match,
                    "confidence": combined_score,
                    "semantic_score": semantic_score,
                    "ocr_detector_confidence": detector_score,
                    "best_timestamp": float(
                        observation.get("timestamp", 0.0)
                    ),
                    "start_ts": float(
                        observation.get("timestamp", 0.0)
                    ),
                    "end_ts": float(
                        observation.get("timestamp", 0.0)
                    ) + 0.75,
                    "bbox_2d": bbox,
                    "bbox_norm": bbox_norm,
                    "frame_path": observation.get("path", ""),
                    "evidence_type": "ocr_region",
                    "reason": f"OCR region: {region_text}",
                    "matched_text": region_text,
                    "matched_target": match_info[
                        "matched_target"
                    ],
                    "target_priority": match_info[
                        "target_priority"
                    ],
                    "target_kind": match_info[
                        "target_kind"
                    ],
                    "exact_target_match": exact_target_match,
                }

                if best is None:
                    best = item
                elif item["match"] and not best.get("match"):
                    best = item
                elif (
                    item["match"] == best.get("match")
                    and item["confidence"]
                    > best.get("confidence", 0.0)
                ):
                    best = item

        return best

    @staticmethod
    def _select_ordered_frames(candidate, max_frames: int = 4):
        payload = candidate.get("payload", {})
        evidence = candidate.get("frame_evidence") or payload.get("frame_evidence", [])
        unique = {}
        for item in evidence:
            path = item.get("path", "")
            if path and os.path.exists(path):
                unique[path] = {
                    "path": path,
                    "timestamp": float(item.get("timestamp", 0.0)),
                    "width": int(item.get("width", 0)),
                    "height": int(item.get("height", 0)),
                }
        for frame in payload.get("keyframes", []):
            path = frame.get("path", "")
            if path and os.path.exists(path) and path not in unique:
                unique[path] = {
                    "path": path,
                    "timestamp": float(frame.get("timestamp", 0.0)),
                    "width": int(frame.get("width", 0)),
                    "height": int(frame.get("height", 0)),
                }
        frames = sorted(unique.values(), key=lambda item: item["timestamp"])
        if len(frames) > max_frames:
            indices = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
            frames = [frames[i] for i in indices]
        return frames

    @staticmethod
    def _parse_json(text: str):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            try:
                return json.loads(text[start:end + 1].replace("'", '"'))
            except Exception:
                return {}

    def _vlm_grounding(self, query, candidate):
        self.load_model()
        frames = self._select_ordered_frames(candidate)
        if not frames or self.model is None or self.processor is None:
            return None

        images = [Image.open(frame["path"]).convert("RGB") for frame in frames]
        timeline = [{"index": idx, "timestamp": frame["timestamp"]} for idx, frame in enumerate(frames)]
        content = []
        for idx, image in enumerate(images):
            content.append({"type": "text", "text": f"FRAME {idx}, timestamp={frames[idx]['timestamp']:.3f}s"})
            content.append({"type": "image", "image": image})
        content.append({
            "type": "text",
            "text": (
                f"Query: {query}\n"
                f"Transcript: {candidate.get('payload', {}).get('transcript', '')}\n"
                f"OCR: {candidate.get('payload', {}).get('ocr_text', '')}\n"
                f"Timeline: {json.dumps(timeline)}\n"
                "Return ONLY JSON with keys: match(boolean), event_start_index(integer), "
                "event_end_index(integer), best_frame_index(integer), bbox_2d([x1,y1,x2,y2] "
                "in normalized 0-1000 coordinates, or []), confidence(0-1), reason(string). "
                "Do not invent an event outside the ordered frames."
            ),
        })
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        moved = {}
        for key, value in inputs.items():
            if torch is not None and isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    moved[key] = value.to(self.device, dtype=torch.float16)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        inputs = moved
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        input_length = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
        data = self._parse_json(decoded)
        if not data:
            return None

        def safe_index(value, default=0):
            try:
                return min(max(int(value), 0), len(frames) - 1)
            except Exception:
                return default

        start_index = safe_index(data.get("event_start_index", 0))
        end_index = safe_index(data.get("event_end_index", start_index), start_index)
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        best_index = safe_index(data.get("best_frame_index", start_index), start_index)
        bbox_norm = data.get("bbox_2d", []) or []
        width, height = frames[best_index]["width"], frames[best_index]["height"]
        bbox = []
        if len(bbox_norm) == 4 and width > 0 and height > 0:
            bbox = [
                int(round(bbox_norm[0] * width / 1000)),
                int(round(bbox_norm[1] * height / 1000)),
                int(round(bbox_norm[2] * width / 1000)),
                int(round(bbox_norm[3] * height / 1000)),
            ]
        return {
            "match": bool(data.get("match", False)),
            "confidence": max(0.0, min(float(data.get("confidence", 0.0)), 1.0)),
            "start_ts": frames[start_index]["timestamp"],
            "end_ts": frames[end_index]["timestamp"] + 0.75,
            "best_timestamp": frames[best_index]["timestamp"],
            "bbox_2d": bbox,
            "bbox_norm": bbox_norm if len(bbox_norm) == 4 else [],
            "frame_path": frames[best_index]["path"],
            "evidence_type": "qwen_vl_grounding",
            "reason": str(data.get("reason", "")),
            "raw_output": decoded,
        }

    def ground_candidates(
        self,
        query,
        plan,
        candidates,
        top_k: int = 5,
    ):
        if not candidates:
            return candidates

        if torch is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        precomputed_ocr = {}
        for candidate in candidates:
            if "ocr" in plan.modalities:
                precomputed_ocr[candidate["candidate_id"]] = (
                    self._ocr_grounding(plan, candidate)
                )

        exact_ocr_available = any(
            evidence
            and evidence.get("match")
            and evidence.get("exact_target_match")
            and evidence.get("bbox_norm")
            for evidence in precomputed_ocr.values()
        )
        ocr_dominant_query = bool(
            plan.primary_modality == "ocr"
            or plan.modality_weights.get("ocr", 0.0)
            >= max(plan.modality_weights.values() or [0.0])
        )
        skip_vlm_globally = bool(
            exact_ocr_available
            and ocr_dominant_query
        )

        selected_log_count = 0

        for candidate_index, candidate in enumerate(candidates):
            ocr_grounding = precomputed_ocr.get(
                candidate["candidate_id"]
            )

            ocr_is_authoritative = bool(
                ocr_grounding
                and ocr_grounding.get("match")
                and ocr_grounding.get("exact_target_match")
                and ocr_grounding.get("bbox_norm")
            )

            vlm_grounding = None
            needs_vlm = bool(
                candidate_index < top_k
                and plan.requires_spatial_grounding
                and "visual" in plan.modalities
                and not ocr_is_authoritative
                and not skip_vlm_globally
            )

            if (
                ocr_is_authoritative
                and selected_log_count < 5
            ):
                print(
                    "[VerificationLayer] Exact OCR target selected "
                    f"target={ocr_grounding.get('matched_target')!r}, "
                    f"region={ocr_grounding.get('matched_text')!r}"
                )
                selected_log_count += 1

            if needs_vlm:
                try:
                    vlm_grounding = self._vlm_grounding(
                        plan.query_for("visual") or query,
                        candidate,
                    )
                except Exception as exc:
                    print(
                        f"[VerificationLayer] Grounding failed: {exc}"
                    )
                    if (
                        torch is not None
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

            if ocr_is_authoritative:
                grounding = ocr_grounding
            elif (
                ocr_grounding
                and ocr_grounding.get("match")
                and ocr_grounding.get("bbox_norm")
                and (
                    vlm_grounding is None
                    or not vlm_grounding.get("bbox_norm")
                )
            ):
                grounding = ocr_grounding
            elif vlm_grounding is not None:
                grounding = vlm_grounding
            elif ocr_grounding is not None:
                grounding = ocr_grounding
            else:
                grounding = None

            if grounding is None:
                evidence = candidate.get(
                    "frame_evidence",
                    [],
                )
                if evidence:
                    best = max(
                        evidence,
                        key=lambda item: item.get(
                            "score",
                            0.0,
                        ),
                    )
                    grounding = {
                        "match": True,
                        "confidence": max(
                            0.0,
                            min(
                                best.get("score", 0.0),
                                1.0,
                            ),
                        ),
                        "start_ts": best["timestamp"],
                        "end_ts": best["timestamp"] + 0.75,
                        "best_timestamp": best["timestamp"],
                        "bbox_2d": [],
                        "bbox_norm": [],
                        "frame_path": best.get("path", ""),
                        "evidence_type": "frame_vector",
                        "reason": "Fine frame-vector evidence",
                    }
                else:
                    grounding = {
                        "match": False,
                        "confidence": 0.0,
                        "bbox_2d": [],
                        "bbox_norm": [],
                        "evidence_type": "none",
                        "reason": "No spatial evidence",
                    }

            candidate["grounding"] = grounding
            candidate["ocr_grounding_score"] = (
                ocr_grounding.get("confidence", 0.0)
                if ocr_grounding
                else 0.0
            )
            candidate["visual_verification_score"] = (
                grounding.get("confidence", 0.0)
            )
            candidate["verified"] = grounding.get(
                "match",
                False,
            )

        return candidates

    def verify_candidates(self, query, candidates, top_k: int = 3):
        class LegacyPlan:
            modalities = ["visual"]
            requires_spatial_grounding = True
            def query_for(self, modality):
                return query
        return self.ground_candidates(query, LegacyPlan(), candidates, top_k)
