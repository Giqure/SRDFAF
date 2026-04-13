"""VLM-based navigation instruction quality judge.

Supports two backends:
  - LocalJudge: loads a VLM via transformers (for A100 or same-GPU judging)
  - APIJudge: calls an OpenAI-compatible API (vLLM serve, OpenRouter, etc.)

The judge uses a 4-dimension constitutional rubric:
  spatial (30%), landmark (25%), completeness (25%), executability (20%)
"""

import base64
import json
import os
import re
from dataclasses import dataclass

# ── Constitution ──────────────────────────────────────────────────────

CONSTITUTION = {
    "spatial":       {"weight": 0.30, "desc": "Direction words match actual trajectory actions"},
    "landmark":      {"weight": 0.25, "desc": "References specific, actually visible landmarks"},
    "completeness":  {"weight": 0.25, "desc": "Covers all trajectory steps and turning points"},
    "executability": {"weight": 0.20, "desc": "Unambiguously determines the exact path"},
}

JUDGE_PROMPT = """\
Score this navigation instruction on four criteria (0.0-1.0 each).

Criteria:
1. Spatial Accuracy (30%): Direction words (left/right/straight) match the path shown
2. Landmark Quality (25%): References specific, actually visible objects
3. Completeness (25%): Covers all steps, no skipped turns
4. Executability (20%): A follower could unambiguously follow this

Instruction: {instruction}

Respond with ONLY valid JSON:
{{"spatial": 0.0, "landmark": 0.0, "completeness": 0.0, "executability": 0.0}}"""


# ── Scores Dataclass ──────────────────────────────────────────────────


@dataclass
class Scores:
    spatial: float = 0.0
    landmark: float = 0.0
    completeness: float = 0.0
    executability: float = 0.0

    @property
    def weighted(self) -> float:
        return sum(
            getattr(self, k) * v["weight"]
            for k, v in CONSTITUTION.items()
        )


# ── Response Parsing ──────────────────────────────────────────────────


def parse_scores(text: str) -> Scores | None:
    """Extract a Scores object from possibly messy VLM output."""
    # Try markdown code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return _dict_to_scores(d)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try bare JSON objects (find the one with 'spatial' key)
    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            d = json.loads(m.group(0))
            if "spatial" in d:
                return _dict_to_scores(d)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _dict_to_scores(d: dict) -> Scores:
    return Scores(**{
        k: max(0.0, min(1.0, float(d[k])))
        for k in Scores.__dataclass_fields__
        if k in d
    })


# ── Local Judge (transformers) ────────────────────────────────────────


class LocalJudge:
    """Score instructions using a locally-loaded VLM."""

    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def score(self, image_paths: list[str], instruction: str) -> Scores | None:
        from PIL import Image

        content: list[dict] = [
            {"type": "image", "image": f"file://{p}"} for p in image_paths
        ]
        content.append({
            "type": "text",
            "text": JUDGE_PROMPT.format(instruction=instruction),
        })
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        ).to(self.device)

        ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = self.processor.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return parse_scores(response)

    def score_batch(
        self, items: list[tuple[list[str], str]]
    ) -> list[Scores | None]:
        return [self.score(imgs, instr) for imgs, instr in items]


# ── API Judge (OpenAI-compatible) ──────────────────────────────────


class APIJudge:
    """Score instructions via an OpenAI-compatible vision API.

    Works with vLLM serve, OpenRouter, or any compatible endpoint.
    """

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def score(self, image_paths: list[str], instruction: str) -> Scores | None:
        import requests

        content: list[dict] = []
        for p in image_paths:
            b64 = self._encode_image(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({
            "type": "text",
            "text": JUDGE_PROMPT.format(instruction=instruction),
        })

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return parse_scores(text)

    def score_batch(
        self, items: list[tuple[list[str], str]]
    ) -> list[Scores | None]:
        return [self.score(imgs, instr) for imgs, instr in items]
