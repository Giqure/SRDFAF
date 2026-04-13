"""GRPO reward functions for navigation instruction quality.

Each function has the signature ``(completions, **kwargs) -> list[float]``
expected by TRL's ``GRPOTrainer``.  Dataset columns (e.g. ``ground_truth``)
are passed as keyword arguments.
"""

import re

DIRECTION_WORDS = frozenset({
    "left", "right", "forward", "straight", "turn", "walk", "go",
    "head", "proceed", "continue", "stop", "enter", "exit",
})

LANDMARK_WORDS = frozenset({
    "door", "room", "hallway", "corridor", "stairs", "staircase",
    "kitchen", "bathroom", "bedroom", "living", "dining", "table",
    "chair", "couch", "sofa", "bed", "desk", "counter", "cabinet",
    "window", "wall", "floor", "ceiling", "light", "lamp", "picture",
    "painting", "mirror", "shelf", "rug", "carpet", "plant", "tv",
    "refrigerator", "oven", "sink", "toilet", "shower", "bathtub",
})


def direction_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for including direction / action words."""
    rewards = []
    for text in completions:
        words = set(text.lower().split())
        rewards.append(min(len(words & DIRECTION_WORDS) / 5.0, 1.0))
    return rewards


def landmark_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for referencing indoor landmark objects."""
    rewards = []
    for text in completions:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        rewards.append(min(len(words & LANDMARK_WORDS) / 4.0, 1.0))
    return rewards


def length_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward reasonable instruction length, penalise extremes."""
    rewards = []
    for text in completions:
        n = len(text.split())
        if n < 10:
            rewards.append(n / 10.0)
        elif n <= 80:
            rewards.append(1.0)
        else:
            rewards.append(max(0.0, 1.0 - (n - 80) / 80.0))
    return rewards


def structure_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward step-like sentence structure."""
    rewards = []
    for text in completions:
        sents = [
            s.strip()
            for s in re.split(r"[.!,;]|\band\b|\bthen\b", text)
            if s.strip()
        ]
        n = len(sents)
        if n < 2:
            rewards.append(0.3)
        elif n <= 8:
            rewards.append(1.0)
        else:
            rewards.append(max(0.5, 1.0 - (n - 8) / 8.0))
    return rewards


def meteor_reward(
    completions: list[str], ground_truth: list | None = None, **kwargs
) -> list[float]:
    """METEOR score against ground-truth reference instructions.

    ``ground_truth`` is broadcast by TRL's GRPOTrainer: for batch_size=B and
    num_generations=G, it contains B*G items (each a list of reference strings).
    """
    if not ground_truth:
        return [0.5] * len(completions)

    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
    except ImportError:
        return [0.5] * len(completions)

    rewards = []
    for i, text in enumerate(completions):
        refs = ground_truth[i] if i < len(ground_truth) else ground_truth[-1]
        # refs may be a list of reference strings or a single string
        if isinstance(refs, str):
            refs = [refs]
        ref_tokens = [r.lower().split() for r in refs]
        hyp_tokens = text.lower().split()
        try:
            score = _meteor(ref_tokens, hyp_tokens)
        except Exception:
            score = 0.0
        rewards.append(score)
    return rewards


# Registry consumed by GRPOTrainer
REWARD_FUNCTIONS = [
    direction_reward,   # 0.15
    landmark_reward,    # 0.15
    length_reward,      # 0.10
    structure_reward,   # 0.10
    meteor_reward,      # 0.50
]
REWARD_WEIGHTS = [0.15, 0.15, 0.10, 0.10, 0.50]
