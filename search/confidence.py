"""Five-signal composite confidence."""
class ConfidenceCalculator:
    def __init__(
        self,
        w_retrieval=0.25,
        w_cross=0.25,
        w_verification=0.20,
        w_temporal=0.20,
        w_spatial=0.10,
    ):
        weights = [w_retrieval, w_cross, w_verification, w_temporal, w_spatial]
        total = max(sum(weights), 1e-8)
        (
            self.w_retrieval,
            self.w_cross,
            self.w_verification,
            self.w_temporal,
            self.w_spatial,
        ) = [value / total for value in weights]

    @staticmethod
    def _clip(value):
        return max(0.0, min(float(value), 1.0))

    def calculate(
        self,
        retrieval_similarity,
        cross_encoder_score,
        verification_score,
        temporal_score,
        spatial_score,
        relation_required=False,
        relation_satisfied=True,
    ):
        values = {
            "retrieval_similarity": self._clip(retrieval_similarity),
            "cross_encoder_score": self._clip(cross_encoder_score),
            "verification_score": self._clip(verification_score),
            "temporal_score": self._clip(temporal_score),
            "spatial_score": self._clip(spatial_score),
        }
        final = (
            self.w_retrieval * values["retrieval_similarity"]
            + self.w_cross * values["cross_encoder_score"]
            + self.w_verification * values["verification_score"]
            + self.w_temporal * values["temporal_score"]
            + self.w_spatial * values["spatial_score"]
        )
        if relation_required and not relation_satisfied:
            final = min(final, 0.30)
        return self._clip(final), values
