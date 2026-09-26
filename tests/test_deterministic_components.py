import pytest

from api.schemas import IngestRequest, SearchQueryRequest, StrictSearchResponse
from evaluation.metrics import evaluate_predictions, temporal_iou
from search.confidence import ConfidenceCalculator
from search.query_understanding import FastQueryUnderstander
from search.temporal_reasoner import TemporalReasoner


def candidate(start, confidence=0.5, relation="none", satisfied=True):
    return {
        "refined": {
            "refined_start": start,
            "temporal_relation": relation,
            "relation_satisfied": satisfied,
        },
        "final_confidence": confidence,
    }


def test_temporal_iou_and_evaluation():
    assert temporal_iou((1, 3), (2, 4)) == pytest.approx(1 / 3)
    result = evaluate_predictions([
        {"id": "a", "predicted": [1, 3], "truth": [1, 3]},
        {"id": "b", "predicted": [0, 1], "truth": [2, 3]},
    ])
    assert result["mean_temporal_iou"] == pytest.approx(0.5)
    assert result["recall_at_iou_0_5"] == pytest.approx(0.5)


def test_temporal_selector_respects_relations():
    reasoner = TemporalReasoner()
    candidates = [candidate(4, 0.9), candidate(2, 0.4)]
    assert reasoner.select(candidates, "first")[0]["refined"]["refined_start"] == 2
    assert reasoner.select(candidates, "last")[0]["refined"]["refined_start"] == 4
    assert reasoner.select([candidate(1, relation="while", satisfied=False)], "any") == []


def test_confidence_clips_inputs_and_rejects_failed_relation():
    calculator = ConfidenceCalculator()
    score, values = calculator.calculate(2, -1, 0.5, 0.5, 0.5)
    assert score == pytest.approx(0.50)
    assert values["retrieval_similarity"] == 1.0
    failed, _ = calculator.calculate(1, 1, 1, 1, 1, relation_required=True, relation_satisfied=False)
    assert failed == 0.30


def test_deterministic_query_parser():
    intent = FastQueryUnderstander(use_llm=False).analyze(
        "Show screen text EXPOSE 8080 while the speaker says port forwarding"
    )
    assert intent.temporal_relation == "while"
    assert intent.needs_ocr is True
    assert intent.needs_speech is True
    assert intent.intent == "temporal_search"


def test_schema_validation():
    request = SearchQueryRequest(query="find the first command", video_id="demo")
    assert request.video_id == "demo"
    assert IngestRequest(video_path="clip.mp4").video_id == "default_video"
    with pytest.raises(Exception):
        StrictSearchResponse(start_timestamp="bad", end_timestamp="bad")
