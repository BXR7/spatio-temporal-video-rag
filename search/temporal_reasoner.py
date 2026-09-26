class TemporalReasoner:
    @staticmethod
    def _start(c):
        return float(c.get("refined",{}).get("refined_start", c.get("payload",{}).get("start_ts",0)))

    def select(self, candidates, selector="any"):
        if not candidates: return []
        relation_required = any(
            c.get("refined",{}).get("temporal_relation","none") in {"while","before","after"}
            for c in candidates
        )
        valid = [c for c in candidates if c.get("refined",{}).get("relation_satisfied",True)]
        if relation_required and not valid: return []
        pool = valid if relation_required else candidates
        if selector == "first": return sorted(pool, key=self._start)
        if selector == "last": return sorted(pool, key=self._start, reverse=True)
        return sorted(pool, key=lambda c:c.get("final_confidence",0), reverse=True)

    def apply(self, candidates, temporal_intent="any"):
        return self.select(candidates, temporal_intent)
