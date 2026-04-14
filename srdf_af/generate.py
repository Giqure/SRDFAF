"""Candidate navigation instruction generation from a Speaker model.

Supports two backends:
  - Local: loads model via transformers (for fine-tuned speakers)
  - API: calls an OpenAI-compatible VLM API (DashScope, SiliconFlow, etc.)
"""

import base64
import json
import random
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from srdf_af.data import SYSTEM_PROMPT, USER_PROMPT

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _build_prompt(image_paths: list[str]) -> list[dict]:
    """Build VLM chat messages for instruction generation."""
    content: list[dict] = [
        {"type": "image", "image": f"file://{p}"} for p in image_paths
    ]
    content.append({"type": "text", "text": USER_PROMPT})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def generate_candidates(
    model,
    processor,
    image_paths: list[str],
    n: int = 8,
    temp_range: tuple[float, float] = (0.7, 1.2),
    max_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Generate *n* diverse candidate instructions for a single trajectory.

    Each candidate uses a different temperature sampled uniformly from
    *temp_range* to encourage diversity.
    """
    messages = _build_prompt(image_paths)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    ).to(device)

    candidates: list[str] = []
    for _ in range(n):
        t = random.uniform(*temp_range)
        ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=t,
            do_sample=True,
            top_p=0.95,
        )
        out = processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        out = _THINK_RE.sub("", out)  # strip <think> blocks if present
        candidates.append(out.strip())
    return candidates


def generate_all(
    model,
    processor,
    r2r_data: list[dict],
    conn_dir: str,
    image_dir: str,
    output_path: str,
    n: int = 8,
    max_images: int = 12,
    temp_range: tuple[float, float] = (0.7, 1.2),
    max_tokens: int = 512,
    device: str = "cuda",
) -> int:
    """Generate candidate instructions for all R2R trajectories.

    Writes one JSONL record per trajectory to *output_path*, each containing
    ``path_id``, ``scan``, ``path``, ``ground_truth``, ``candidates``,
    and ``image_paths``.

    Returns the number of trajectories processed.
    """
    from srdf_af.data import load_connectivity, trajectory_headings, resolve_images

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    conn_cache: dict[str, dict] = {}
    count = 0

    with open(output_path, "w") as f:
        for entry in tqdm(r2r_data, desc="Generating candidates"):
            scan = entry["scan"]
            if scan not in conn_cache:
                try:
                    conn_cache[scan] = load_connectivity(conn_dir, scan)
                except FileNotFoundError:
                    continue

            headings = trajectory_headings(
                conn_cache[scan], entry["path"], entry.get("heading", 0.0)
            )
            imgs = resolve_images(
                image_dir, scan, entry["path"], headings, max_images
            )
            if not imgs:
                continue

            cands = generate_candidates(
                model, processor, imgs, n, temp_range, max_tokens, device
            )

            record = {
                "path_id": entry["path_id"],
                "scan": scan,
                "path": entry["path"],
                "ground_truth": entry["instructions"],
                "candidates": cands,
                "image_paths": imgs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


# ── API-based Generation ──────────────────────────────────────────────


class APIGenerator:
    """Generate instructions via an OpenAI-compatible VLM API."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def generate_candidates(
        self,
        image_paths: list[str],
        n: int = 8,
        temp_range: tuple[float, float] = (0.7, 1.2),
        max_tokens: int = 512,
    ) -> list[str]:
        import requests

        content: list[dict] = []
        for p in image_paths:
            b64 = self._encode_image(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": USER_PROMPT})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        candidates: list[str] = []
        for _ in range(n):
            t = random.uniform(*temp_range)
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": t,
                    "top_p": 0.95,
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            candidates.append(text.strip())
        return candidates


def generate_all_api(
    generator: "APIGenerator",
    r2r_data: list[dict],
    conn_dir: str,
    image_dir: str,
    output_path: str,
    n: int = 8,
    max_images: int = 12,
    temp_range: tuple[float, float] = (0.7, 1.2),
    max_tokens: int = 512,
) -> int:
    """Generate candidate instructions for all R2R trajectories via API."""
    from srdf_af.data import load_connectivity, trajectory_headings, resolve_images

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    conn_cache: dict[str, dict] = {}
    count = 0

    with open(output_path, "w") as f:
        for entry in tqdm(r2r_data, desc="Generating candidates (API)"):
            scan = entry["scan"]
            if scan not in conn_cache:
                try:
                    conn_cache[scan] = load_connectivity(conn_dir, scan)
                except FileNotFoundError:
                    continue

            headings = trajectory_headings(
                conn_cache[scan], entry["path"], entry.get("heading", 0.0)
            )
            imgs = resolve_images(
                image_dir, scan, entry["path"], headings, max_images
            )
            if not imgs:
                continue

            cands = generator.generate_candidates(
                imgs, n, temp_range, max_tokens
            )

            record = {
                "path_id": entry["path_id"],
                "scan": scan,
                "path": entry["path"],
                "ground_truth": entry["instructions"],
                "candidates": cands,
                "image_paths": imgs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count
