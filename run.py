#!/usr/bin/env python3
"""SRDF-AF CLI — Self-Refining Data Flywheel with AI Feedback.

Usage:
    python run.py render   [--config configs/default.yaml]
    python run.py sft      [--config configs/default.yaml]
    python run.py flywheel [--config configs/default.yaml]
    python run.py evaluate --predictions PATH --r2r PATH
"""

import logging

import fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


def render(config: str = "configs/default.yaml"):
    """Render perspective images from MP3D skybox data for all R2R trajectories."""
    from srdf_af.config import Config
    from srdf_af.data import load_r2r
    from srdf_af.render import render_all

    cfg = Config.load(config)
    r2r = load_r2r(cfg.r2r_dir, cfg.split)
    render_all(
        r2r, cfg.skybox_dir, cfg.connectivity_dir, cfg.image_dir,
        fov=cfg.fov, size=cfg.image_size,
    )


def sft(config: str = "configs/default.yaml"):
    """Run SFT training only (Round 0)."""
    from pathlib import Path

    from srdf_af.config import Config
    from srdf_af.flywheel import _prepare_sft_data
    from srdf_af.train import train_sft

    cfg = Config.load(config)
    base = Path(cfg.output_dir)

    sft_data = str(base / "sft_data.jsonl")
    n = _prepare_sft_data(cfg, sft_data)
    logging.info(f"SFT data: {n} items")

    train_sft(cfg, sft_data, str(base / "sft_speaker"))


def flywheel(config: str = "configs/default.yaml"):
    """Run the full multi-round self-improving flywheel."""
    from srdf_af.config import Config
    from srdf_af.flywheel import run_flywheel

    cfg = Config.load(config)
    final = run_flywheel(cfg)
    logging.info(f"Final speaker: {final}")


def evaluate(predictions: str, r2r: str):
    """Compute language metrics (BLEU/METEOR/ROUGE-L/CIDEr + nav quality)."""
    from srdf_af.evaluate import evaluate_file

    metrics = evaluate_file(predictions, r2r)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:20s}  {v:.4f}")
        else:
            print(f"  {k:20s}  {v}")


if __name__ == "__main__":
    fire.Fire({
        "render": render,
        "sft": sft,
        "flywheel": flywheel,
        "evaluate": evaluate,
    })
