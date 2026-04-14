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


def generate(
    config: str = "configs/4080.yaml",
    speaker: str = "output/round_0/speaker_v2/checkpoint-878",
    output: str = "output/round_1/candidates.jsonl",
    max_traj: int = 0,
):
    """Generate candidate instructions from a finetuned LoRA speaker.

    Args:
        speaker: Path to LoRA adapter checkpoint (from SFT).
        max_traj: Limit number of trajectories (0 = all).
    """
    import random

    import torch
    from peft import PeftModel

    from srdf_af.config import Config
    from srdf_af.data import load_r2r
    from srdf_af.generate import generate_all
    from srdf_af.train import load_model_and_processor

    cfg = Config.load(config)
    model, processor = load_model_and_processor(cfg.speaker, cfg.qlora)
    model = PeftModel.from_pretrained(model, speaker)
    model.eval()

    r2r = load_r2r(cfg.r2r_dir, cfg.split)
    if max_traj > 0:
        random.seed(42)
        random.shuffle(r2r)
        r2r = r2r[:max_traj]
    logging.info(f"Generating {cfg.n_candidates} candidates for {len(r2r)} trajectories")

    n = generate_all(
        model, processor, r2r, cfg.connectivity_dir, cfg.image_dir,
        output, n=cfg.n_candidates, max_images=cfg.max_images,
        temp_range=(cfg.temp_low, cfg.temp_high), max_tokens=cfg.max_gen_tokens,
    )
    logging.info(f"Generated candidates for {n} trajectories → {output}")

    del model, processor
    torch.cuda.empty_cache()


def judge(
    config: str = "configs/4080.yaml",
    candidates: str = "output/round_1/candidates.jsonl",
    output: str = "output/round_1/preferences.jsonl",
):
    """Judge candidates via VLM API and build preference pairs."""
    from srdf_af.config import Config
    from srdf_af.judge import APIJudge
    from srdf_af.preference import build_preferences

    cfg = Config.load(config)
    j = APIJudge(cfg.judge_api, cfg.judge_model, cfg.judge_api_key)
    n = build_preferences(candidates, j, output, cfg.score_gap)
    logging.info(f"Built {n} preference pairs → {output}")


def dpo(
    config: str = "configs/4080.yaml",
    preferences: str = "output/round_1/preferences.jsonl",
    speaker: str = "output/round_0/speaker_v2/checkpoint-878",
    output: str = "output/round_1/speaker",
):
    """DPO training on VLM-judged preference pairs.

    Loads base model + SFT LoRA adapter, then continues DPO training
    on the same adapter weights.
    """
    import json
    from pathlib import Path

    from datasets import load_dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    from srdf_af.config import Config
    from srdf_af.data import to_dpo
    from srdf_af.flywheel import _convert_pref_to_dpo
    from srdf_af.train import load_model_and_processor

    cfg = Config.load(config)

    # Convert preferences → DPO format
    dpo_path = str(Path(output).parent / "dpo_data.jsonl")
    n = _convert_pref_to_dpo(preferences, dpo_path)
    logging.info(f"DPO data: {n} pairs")

    # Load base model + SFT adapter (train existing LoRA)
    model, processor = load_model_and_processor(cfg.speaker, cfg.qlora)
    model = PeftModel.from_pretrained(model, speaker, is_trainable=True)

    dataset = load_dataset("json", data_files=dpo_path, split="train")

    training_args = DPOConfig(
        output_dir=output,
        num_train_epochs=cfg.dpo_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum * 2,
        learning_rate=cfg.dpo_lr,
        beta=cfg.dpo_beta,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        max_length=cfg.max_seq_len,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(output)
    processor.save_pretrained(output)


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
        "generate": generate,
        "judge": judge,
        "dpo": dpo,
        "evaluate": evaluate,
    })
