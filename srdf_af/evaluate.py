"""Language quality metrics for navigation instructions.

Wraps pycocoevalcap (BLEU, METEOR, ROUGE-L, CIDEr) and adds
navigation-specific quality signals (direction density, landmark density).
"""

import json
from pathlib import Path

# ── Navigation-Specific Metrics ──────────────────────────────────────

_DIR_WORDS = frozenset({
    "left", "right", "forward", "straight", "turn", "walk",
    "go", "head", "stop", "proceed", "continue",
})

_LAND_WORDS = frozenset({
    "door", "room", "hallway", "stairs", "kitchen", "bathroom", "bedroom",
    "table", "chair", "couch", "bed", "desk", "counter", "window",
    "wall", "light", "shelf", "rug", "plant", "refrigerator", "sink", "toilet",
})


def direction_density(texts: list[str]) -> float:
    """Mean fraction of direction words per instruction."""
    total = 0.0
    for t in texts:
        words = t.lower().split()
        total += sum(1 for w in words if w in _DIR_WORDS) / max(len(words), 1)
    return total / max(len(texts), 1)


def landmark_density(texts: list[str]) -> float:
    """Mean fraction of landmark words per instruction."""
    total = 0.0
    for t in texts:
        words = t.lower().split()
        total += sum(1 for w in words if w in _LAND_WORDS) / max(len(words), 1)
    return total / max(len(texts), 1)


# ── Standard Language Metrics ─────────────────────────────────────────


def compute_metrics(
    predictions: list[str], references: list[list[str]]
) -> dict[str, float]:
    """BLEU-4, METEOR, ROUGE-L, CIDEr via pycocoevalcap."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    scores: dict[str, float] = {}
    for name, scorer in [
        ("BLEU-4", Bleu(4)),
        ("METEOR", Meteor()),
        ("ROUGE-L", Rouge()),
        ("CIDEr", Cider()),
    ]:
        s, _ = scorer.compute_score(gts, res)
        scores[name] = s[-1] if isinstance(s, list) else s  # type: ignore[index]

    return scores


# ── File-Level Evaluation ─────────────────────────────────────────────


def evaluate_file(predictions_path: str, r2r_path: str) -> dict[str, float | int]:
    """Evaluate a predictions JSONL against R2R ground truth.

    Predictions file: one JSON per line with ``path_id`` and ``prediction``.
    """
    with open(predictions_path) as f:
        preds_data = [json.loads(l) for l in f if l.strip()]
    with open(r2r_path) as f:
        r2r = {item["path_id"]: item["instructions"] for item in json.load(f)}

    predictions, references = [], []
    for entry in preds_data:
        pid = entry["path_id"]
        if pid not in r2r:
            continue
        predictions.append(entry["prediction"])
        references.append(r2r[pid])

    if not predictions:
        return {}

    metrics = compute_metrics(predictions, references)
    metrics["direction_density"] = direction_density(predictions)
    metrics["landmark_density"] = landmark_density(predictions)
    metrics["n_samples"] = len(predictions)
    return metrics
