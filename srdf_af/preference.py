"""Build preference pairs (chosen / rejected) from judge-scored candidates.

Each R2R trajectory has N candidate instructions.  The judge scores each one,
then we pair the best and worst candidates whose weighted-score gap exceeds
a threshold — this is the signal-to-noise filter that ensures DPO only
trains on meaningful contrasts.
"""

import json
from pathlib import Path

from tqdm import tqdm

from srdf_af.judge import Scores


def build_preferences(
    candidates_path: str,
    judge,
    output_path: str,
    score_gap: float = 0.15,
) -> int:
    """Score all candidates and construct preference pairs.

    Args:
        candidates_path: JSONL with ``candidates`` and ``image_paths`` per trajectory.
        judge: A ``LocalJudge`` or ``APIJudge`` instance.
        output_path: Where to write the preference JSONL.
        score_gap: Minimum weighted-score difference between chosen and rejected.

    Returns:
        Number of valid preference pairs written.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(candidates_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc="Building preferences"):
            entry = json.loads(line)
            image_paths: list[str] = entry["image_paths"]
            candidates: list[str] = entry["candidates"]

            if len(candidates) < 2:
                continue

            # Score every candidate
            scored: list[tuple[str, Scores]] = []
            for cand in candidates:
                s = judge.score(image_paths, cand)
                if s is not None:
                    scored.append((cand, s))

            if len(scored) < 2:
                continue

            # Pick best and worst by weighted score
            scored.sort(key=lambda x: x[1].weighted, reverse=True)
            best_text, best_s = scored[0]
            worst_text, worst_s = scored[-1]
            gap = best_s.weighted - worst_s.weighted

            if gap < score_gap:
                continue

            record = {
                "path_id": entry["path_id"],
                "scan": entry["scan"],
                "path": entry["path"],
                "ground_truth": entry["ground_truth"],
                "image_paths": image_paths,
                "chosen": best_text,
                "rejected": worst_text,
                "chosen_score": round(best_s.weighted, 4),
                "rejected_score": round(worst_s.weighted, 4),
                "score_gap": round(gap, 4),
                "chosen_detail": {
                    "spatial": best_s.spatial,
                    "landmark": best_s.landmark,
                    "completeness": best_s.completeness,
                    "executability": best_s.executability,
                },
                "rejected_detail": {
                    "spatial": worst_s.spatial,
                    "landmark": worst_s.landmark,
                    "completeness": worst_s.completeness,
                    "executability": worst_s.executability,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count
