"""Temporal grounding evaluation metrics."""
def temporal_iou(predicted, truth):
    intersection = max(0.0, min(predicted[1], truth[1]) - max(predicted[0], truth[0]))
    union = max(predicted[1], truth[1]) - min(predicted[0], truth[0])
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(cases):
    rows = []
    for case in cases:
        score = temporal_iou(tuple(case["predicted"]), tuple(case["truth"]))
        rows.append({**case, "temporal_iou": score, "iou_at_0_5": score >= 0.5})
    mean_iou = sum(row["temporal_iou"] for row in rows) / max(len(rows), 1)
    recall = sum(row["iou_at_0_5"] for row in rows) / max(len(rows), 1)
    return {
        "cases": rows,
        "mean_temporal_iou": mean_iou,
        "recall_at_iou_0_5": recall,
    }
