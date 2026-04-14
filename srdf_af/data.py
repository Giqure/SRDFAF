"""R2R data loading, connectivity graph, heading computation, and format conversion."""

import json
import math
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert indoor navigation instructor. Given a sequence of "
    "panoramic images along a navigation path, generate a clear, step-by-step "
    "instruction that would allow a person to follow the exact same route."
)

USER_PROMPT = (
    "These images show consecutive viewpoints along an indoor navigation path, "
    "ordered from start to end. Generate a detailed navigation instruction."
)


# ── R2R Data ──────────────────────────────────────────────────────────


def load_r2r(data_dir: str, split: str = "train") -> list[dict]:
    """Load R2R navigation annotations."""
    with open(Path(data_dir) / f"R2R_{split}.json") as f:
        return json.load(f)


# ── Connectivity & Heading ────────────────────────────────────────────


def load_connectivity(conn_dir: str, scan: str) -> dict[str, list[float]]:
    """Load viewpoint positions {image_id: [x, y, z]} from connectivity JSON."""
    with open(Path(conn_dir) / f"{scan}_connectivity.json") as f:
        data = json.load(f)
    positions = {}
    for node in data:
        pose = node["pose"]  # 4x4 column-major flat list
        positions[node["image_id"]] = [pose[3], pose[7], pose[11]]
    return positions


def trajectory_headings(
    positions: dict[str, list[float]],
    path: list[str],
    initial_heading: float = 0.0,
) -> list[float]:
    """Compute viewing heading at each viewpoint along a trajectory.

    For the first viewpoint, look toward the second.  For subsequent viewpoints,
    look in the direction of travel (from previous to current).  At the last
    viewpoint, keep the previous heading.
    """
    headings: list[float] = []
    for i, vid in enumerate(path):
        if vid not in positions:
            headings.append(headings[-1] if headings else initial_heading)
            continue
        if i < len(path) - 1 and path[i + 1] in positions:
            a, b = positions[vid], positions[path[i + 1]]
            headings.append(math.atan2(b[0] - a[0], b[2] - a[2]))
        elif headings:
            headings.append(headings[-1])
        else:
            headings.append(initial_heading)
    return headings


# ── Image Path Resolution ─────────────────────────────────────────────


def _subsample_indices(n: int, k: int) -> list[int]:
    """Return k evenly-spaced indices from 0..n-1, always including first & last."""
    if n <= k:
        return list(range(n))
    return list(dict.fromkeys(round(i * (n - 1) / (k - 1)) for i in range(k)))


def resolve_images(
    image_dir: str,
    scan: str,
    path: list[str],
    headings: list[float],
    max_images: int = 12,
) -> list[str]:
    """Find rendered perspective image files for a trajectory.

    Images are expected at ``{image_dir}/{scan}/{viewpoint}_{heading_deg}.jpg``.
    Falls back to any available heading for the viewpoint if the exact one is
    missing.
    """
    if len(path) > max_images:
        idx = _subsample_indices(len(path), max_images)
        path = [path[i] for i in idx]
        headings = [headings[i] for i in idx]

    base = Path(image_dir) / scan
    result: list[str] = []
    for vid, h in zip(path, headings):
        deg = round(math.degrees(h)) % 360
        exact = base / f"{vid}_{deg}.jpg"
        if exact.exists():
            result.append(str(exact.resolve()))
            continue
        # fallback: any heading for this viewpoint
        candidates = sorted(base.glob(f"{vid}_*.jpg"))
        if candidates:
            result.append(str(candidates[0].resolve()))
    return result


# ── VLM Message Formatting ────────────────────────────────────────


def _user_content(image_paths: list[str]) -> list[dict]:
    """Build multimodal user content list for VLM."""
    parts: list[dict] = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
    parts.append({"type": "text", "text": USER_PROMPT})
    return parts


def _user_content_trl(image_paths: list[str]) -> list[dict]:
    """Build user content with image placeholders (for TRL datasets)."""
    parts: list[dict] = [{"type": "image"} for _ in image_paths]
    parts.append({"type": "text", "text": USER_PROMPT})
    return parts


def to_sft(image_paths: list[str], instruction: str) -> dict:
    """Format one (trajectory, instruction) pair for SFT training."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content_trl(image_paths)},
            {"role": "assistant", "content": instruction.strip()},
        ],
        "images": image_paths,
    }


def to_dpo(image_paths: list[str], chosen: str, rejected: str) -> dict:
    """Format a preference pair for DPO training."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content_trl(image_paths)},
        ],
        "chosen": [{"role": "assistant", "content": chosen.strip()}],
        "rejected": [{"role": "assistant", "content": rejected.strip()}],
        "images": image_paths,
    }


def to_grpo(image_paths: list[str], ground_truth: list[str]) -> dict:
    """Format a trajectory prompt for GRPO training."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content_trl(image_paths)},
        ],
        "ground_truth": ground_truth,
        "images": image_paths,
    }
