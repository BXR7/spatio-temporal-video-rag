import gc, json, os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
try:
    import torch
except Exception:
    torch = None
from search.device_detector import configured_devices, release_cuda

VALID_MODALITIES = {"visual", "speech", "ocr"}

@dataclass
class QueryIntent:
    original_query: str
    intent: str = "content_search"
    entities: List[str] = field(default_factory=list)
    temporal_selector: str = "any"
    temporal_relation: str = "none"
    visual_query: str = ""
    speech_query: str = ""
    ocr_query: str = ""
    primary_modality: str = "visual"
    relation_left_modality: str = "visual"
    relation_right_modality: str = "speech"
    needs_visual: bool = True
    needs_speech: bool = True
    needs_ocr: bool = False
    requires_spatial_grounding: bool = True
    modality_weights: Dict[str, float] = field(default_factory=dict)
    expanded_queries: List[str] = field(default_factory=list)

class FastQueryUnderstander:
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, device: Optional[str] = None, autoload: bool = False, use_llm: Optional[bool] = None):
        self.device = device or configured_devices()["search"]
        self.use_llm = bool(use_llm) if use_llm is not None else os.environ.get("VIDEO_RAG_USE_LLM_PLANNER", "0") == "1"
        self.model = None
        self.tokenizer = None
        if autoload and self.use_llm:
            self.load_model()

    def load_model(self):
        if not self.use_llm or self.model is not None or torch is None or not self.device.startswith("cuda"):
            return self
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME, quantization_config=q, device_map={"": self.device},
                low_cpu_mem_usage=True).eval()
        except Exception as exc:
            print(f"[QueryUnderstander] deterministic fallback: {exc}")
            self.model = None
        return self

    def release_model(self):
        for name in ("model", "tokenizer"):
            obj = getattr(self, name, None)
            if obj is not None:
                try: del obj
                except Exception: pass
            setattr(self, name, None)
        gc.collect()
        release_cuda(self.device)

    @staticmethod
    def _contains(text, terms):
        t = text.casefold()
        return any(x in t for x in terms)

    @staticmethod
    def _relation(value, query):
        text = f"{value or ''} {query}".casefold()
        if any(x in text for x in ("while", "during", "at the same time", "بينما", "أثناء")): return "while"
        if any(x in text for x in ("before", "قبل")): return "before"
        if any(x in text for x in ("after", "بعد")): return "after"
        return "none"

    @staticmethod
    def _selector(value, query):
        text = f"{value or ''} {query}".casefold()
        if any(x in text for x in ("first", "earliest", "أول")): return "first"
        if any(x in text for x in ("last", "latest", "آخر")): return "last"
        return "any"

    @staticmethod
    def _modality(value, default):
        text = str(value or "").casefold()
        if "speech" in text or "spoken" in text: return "speech"
        if "ocr" in text or "screen text" in text: return "ocr"
        if "visual" in text or "image" in text: return "visual"
        return default

    @staticmethod
    def _weights(raw, active):
        vals = {}
        for m in active:
            try: vals[m] = max(float((raw or {}).get(m, 0)), 0)
            except Exception: vals[m] = 0
        if sum(vals.values()) <= 0: vals = {m: 1 for m in active}
        total = sum(vals.values())
        return {k: v / total for k, v in vals.items()}

    def analyze(self, query):
        query = str(query).strip()
        deterministic = self._fallback(query)
        if not self.use_llm:
            print("[QueryUnderstander] deterministic strict parser")
            return deterministic
        self.load_model()
        if self.model is None: return deterministic
        try:
            data = self._llm(query)
            return self._repair_llm(query, data, deterministic)
        except Exception as exc:
            print(f"[QueryUnderstander] invalid LLM output, fallback: {exc}")
            return deterministic

    def _llm(self, query):
        system = ("Return ONLY JSON. Allowed temporal_selector: first,last,any. "
                  "Allowed temporal_relation: while,before,after,none. "
                  "Allowed modalities: visual,speech,ocr. Include visual_query,speech_query,ocr_query, "
                  "needs_visual,needs_speech,needs_ocr,requires_spatial_grounding,modality_weights.")
        messages = [{"role":"system","content":system},{"role":"user","content":query}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=260, do_sample=False)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a: raise ValueError("no JSON")
        return json.loads(text[a:b+1])

    def _repair_llm(self, query, data, base):
        relation = self._relation(data.get("temporal_relation"), query)
        selector = self._selector(data.get("temporal_selector"), query)
        nv = base.needs_visual or bool(data.get("needs_visual", False))
        ns = base.needs_speech or bool(data.get("needs_speech", False))
        no = base.needs_ocr or bool(data.get("needs_ocr", False))
        active = [m for m, ok in (("visual",nv),("speech",ns),("ocr",no)) if ok] or ["visual","speech","ocr"]
        return QueryIntent(
            original_query=query,
            intent=(
                "temporal_search"
                if relation != "none" or selector != "any"
                else "content_search"
            ),
            entities=list(data.get("entities") or base.entities),
            temporal_selector=selector,
            temporal_relation=relation,
            visual_query=str(data.get("visual_query") or base.visual_query),
            speech_query=str(data.get("speech_query") or base.speech_query),
            ocr_query=str(data.get("ocr_query") or base.ocr_query),
            primary_modality=self._modality(data.get("primary_modality"), base.primary_modality),
            relation_left_modality=self._modality(data.get("relation_left_modality"), base.relation_left_modality),
            relation_right_modality=self._modality(data.get("relation_right_modality"), base.relation_right_modality),
            needs_visual=nv, needs_speech=ns, needs_ocr=no,
            requires_spatial_grounding=base.requires_spatial_grounding or bool(data.get("requires_spatial_grounding", False)),
            modality_weights=self._weights(data.get("modality_weights"), active),
            expanded_queries=list(dict.fromkeys([query] + list(data.get("expanded_queries") or [])))[:3],
        )

    def _fallback(self, query):
        q = f" {query.casefold()} "
        relation = self._relation("", query)
        selector = self._selector("", query)
        ocr_terms = ["written","text","code","command","command-line","terminal","dockerfile","screen","port","expose",
                     "مكتوب","كود","أمر","شاشة","سطر","منفذ"]
        speech_terms = [
            "say", "says", "said",
            "mention", "mentions", "mentioned",
            "speaker", "speaks", "spoken",
            "explain", "explains", "explained",
            "discuss", "discusses", "narrate", "describes",
            "شرح", "يشرح", "يتحدث", "يقول",
        ]
        visual_terms = ["show","shows","appear","display","displays","interface","terminal","screen","person","object",
                        "واجهة","يظهر","يعرض"]
        no = self._contains(q, ocr_terms)
        ns = self._contains(q, speech_terms) or relation != "none"
        nv = self._contains(q, visual_terms) or no or relation != "none"

        connector = next((x.strip() for x in (" while "," during "," at the same time "," بينما "," أثناء ",
                                               " before "," قبل "," after "," بعد ") if x in q), None)
        left, right = query, ""
        if connector:
            parts = re.split(re.escape(connector), query, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2: left, right = parts[0].strip(), parts[1].strip()

        def mod(clause):
            c = clause.casefold()
            if self._contains(c, speech_terms): return "speech"
            if self._contains(c, ocr_terms): return "ocr"
            return "visual"

        lm = mod(left)
        rm = mod(right) if right else ("speech" if lm != "speech" else "visual")
        clauses = {
            "visual": "",
            "speech": "",
            "ocr": "",
        }
        clauses[lm] = left

        if right:
            if rm == lm:
                if self._contains(right, speech_terms):
                    rm = "speech"
                elif self._contains(right, ocr_terms):
                    rm = "ocr"
                else:
                    rm = "visual"
            clauses[rm] = right

        if no and not clauses["ocr"]:
            clauses["ocr"] = left if relation != "none" else query
        if nv and not clauses["visual"]:
            clauses["visual"] = left if relation != "none" else query
        if ns and not clauses["speech"]:
            clauses["speech"] = right or query

        active = [m for m, ok in (("visual",nv),("speech",ns),("ocr",no)) if ok] or ["visual","speech","ocr"]
        if nv and ns and no: weights = {"visual":0.30,"speech":0.30,"ocr":0.40}
        elif nv and no: weights = {"visual":0.35,"ocr":0.65}
        else: weights = {m:1/len(active) for m in active}

        return QueryIntent(
            original_query=query,
            intent="temporal_search" if relation != "none" or selector != "any" else "content_search",
            entities=[x for x in re.findall(r"[\w.-]+", query) if len(x)>2][:16],
            temporal_selector=selector, temporal_relation=relation,
            visual_query=clauses["visual"], speech_query=clauses["speech"], ocr_query=clauses["ocr"],
            primary_modality=lm, relation_left_modality=lm, relation_right_modality=rm,
            needs_visual=nv, needs_speech=ns, needs_ocr=no,
            requires_spatial_grounding=nv or no,
            modality_weights=weights, expanded_queries=[query],
        )
