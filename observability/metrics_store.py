import json
import os
from datetime import datetime
from typing import List

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics_log.json")


# ── SAVE SCORES ───────────────────────────────────────────────────
def save_scores(scores: dict):
    """Append evaluation scores to local JSON log"""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

    existing = load_all_scores()
    existing.append(scores)

    with open(METRICS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"💾 Scores saved to {METRICS_FILE}")


# ── LOAD SCORES ───────────────────────────────────────────────────
def load_all_scores() -> List[dict]:
    """Load all historical evaluation scores"""
    if not os.path.exists(METRICS_FILE):
        return []
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


# ── SUMMARY STATS ─────────────────────────────────────────────────
def get_summary_stats() -> dict:
    """Calculate aggregate stats across all evaluations"""
    scores = load_all_scores()
    if not scores:
        return {}

    metrics = [
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "overall_score"
    ]

    summary = {"total_evaluations": len(scores)}
    for metric in metrics:
        values = [s[metric] for s in scores if metric in s]
        if values:
            summary[f"avg_{metric}"] = round(sum(values) / len(values), 3)
            summary[f"min_{metric}"] = round(min(values), 3)
            summary[f"max_{metric}"] = round(max(values), 3)

    return summary