"""Unified SFT / DPO / GRPO training using TRL + LoRA/QLoRA.

All three stages share the same model-loading and LoRA configuration logic.
Only the Trainer class, hyperparameters, and dataset format differ.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from srdf_af.config import Config


# ── Helpers ──────────────────────────────────────────────────────────


def load_model_and_processor(
    model_name: str, qlora: bool = True
):
    """Load a VLM with optional 4-bit QLoRA quantisation."""
    kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )
    return model, processor


def _lora_config(cfg: Config) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── SFT ──────────────────────────────────────────────────────────────


def train_sft(cfg: Config, data_path: str, output_dir: str):
    """Stage 1: Supervised fine-tuning on R2R human instructions."""
    from trl import SFTConfig, SFTTrainer

    model, processor = load_model_and_processor(cfg.speaker, cfg.qlora)
    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.sft_lr,
        max_length=cfg.max_seq_len,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=_lora_config(cfg),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


# ── DPO ──────────────────────────────────────────────────────────────


def train_dpo(
    cfg: Config, data_path: str, model_path: str, output_dir: str
):
    """DPO training on VLM-judged preference pairs."""
    from trl import DPOConfig, DPOTrainer

    model, processor = load_model_and_processor(model_path, cfg.qlora)
    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = DPOConfig(
        output_dir=output_dir,
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
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=_lora_config(cfg),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


# ── GRPO ─────────────────────────────────────────────────────────────


def train_grpo(
    cfg: Config, data_path: str, model_path: str, output_dir: str
):
    """GRPO training with multi-dimensional reward functions."""
    from trl import GRPOConfig, GRPOTrainer

    from srdf_af.rewards import REWARD_FUNCTIONS, REWARD_WEIGHTS

    model, processor = load_model_and_processor(model_path, cfg.qlora)
    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.grpo_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum * 2,
        learning_rate=cfg.grpo_lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        num_generations=cfg.grpo_n_gen,
        max_completion_length=cfg.max_gen_tokens,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=_lora_config(cfg),
        processing_class=processor,
        reward_funcs=REWARD_FUNCTIONS,
        reward_weights=REWARD_WEIGHTS,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
