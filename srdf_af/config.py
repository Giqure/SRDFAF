"""Configuration management via flat dataclass + YAML."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- Paths ---
    r2r_dir: str = "../Matterport3DSimulator/tasks/R2R/data"
    connectivity_dir: str = "../Matterport3DSimulator/connectivity"
    skybox_dir: str = "../RLAIF-NIG/data/mp3d/v1/scans"
    image_dir: str = "./data/images"
    output_dir: str = "./output"

    # --- Data ---
    split: str = "train"
    max_images: int = 12
    fov: int = 90
    image_size: int = 480

    # --- Models ---
    speaker: str = "Qwen/Qwen3-VL-2B-Instruct"
    judge_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    speaker_api: str = ""  # OpenAI-compatible URL for generation (empty = local)
    speaker_api_key: str = ""  # API key for speaker API
    judge_api: str = ""  # OpenAI-compatible URL for judge (empty = local)
    judge_api_key: str = ""  # API key for judge API
    lora_r: int = 64
    lora_alpha: int = 128
    qlora: bool = True

    # --- Generation ---
    n_candidates: int = 8
    temp_low: float = 0.7
    temp_high: float = 1.2
    max_gen_tokens: int = 512

    # --- Judge ---
    judge_temperature: float = 0.0
    score_gap: float = 0.15  # min weighted-score gap for valid preference pair

    # --- Training ---
    sft_lr: float = 2e-4
    sft_epochs: int = 3
    dpo_lr: float = 5e-7
    dpo_beta: float = 0.1
    dpo_epochs: int = 1
    grpo_lr: float = 1e-6
    grpo_epochs: int = 1
    grpo_n_gen: int = 4
    batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 2048

    # --- Flywheel ---
    n_rounds: int = 3
    optim: str = "dpo"  # "dpo" or "grpo"

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)
