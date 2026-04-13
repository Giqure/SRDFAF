# SRDF-AF Project Instructions

## Project Overview
SRDF-AF (Self-Refining Data Flywheel with AI Feedback) is a research project for generating high-quality Vision-Language Navigation (VLN) instructions using RLAIF (Reinforcement Learning from AI Feedback).

**Key idea**: Use a VLM speaker to generate navigation instructions from panoramic images, then a separate VLM judge scores them on 4 constitutional dimensions (spatial grounding, temporal coherence, landmark fidelity, action completeness). High/low score pairs feed DPO/GRPO training in a self-improving flywheel.

## Architecture
- **Speaker**: Qwen2.5-VL-3B-Instruct (QLoRA fine-tuned)
- **Judge**: Qwen3-VL-32B-Instruct (local on 48GB GPU or via DashScope API)
- **Dataset**: R2R (Room-to-Room) on Matterport3D scans
- **Pipeline**: Render skybox views → SFT warm-start → Generate candidates → Judge score → Build preferences → DPO/GRPO → repeat

## Code Structure
```
srdf_af/
├── config.py      # Flat @dataclass Config, loads from YAML
├── data.py        # R2R loading, trajectory image resolution, TRL format conversion
├── render.py      # MP3D skybox → perspective image rendering
├── generate.py    # Local + API candidate generation
├── judge.py       # 4-dimension constitutional AI judge (Local + API)
├── preference.py  # Score-gap filtered preference pairs
├── rewards.py     # 5 GRPO reward functions
├── train.py       # SFT/DPO/GRPO training with TRL + QLoRA
├── flywheel.py    # Multi-round self-improvement orchestration
└── evaluate.py    # BLEU/METEOR/CIDEr + nav quality metrics
```

## Key Technical Details
- TRL SFTConfig uses `max_length` (NOT `max_seq_length`)
- Image format for TRL: separate `images` column with plain file paths, `{"type": "image"}` placeholders in messages
- Use `load_dataset("json", data_files=path)` (NOT `Dataset.from_list()`) to avoid Arrow serialization errors
- Qwen3-VL needs `trust_remote_code=True` with AutoModelForImageTextToText
- MP3D skybox files are 6 flat JPEGs per viewpoint (not cubemap EXR)

## Current Progress
- SFT v1: Completed (780 steps, 5h37m, loss 12.28→7.14) on 4,153 items
- SFT v2: Completed (2,634 steps, ~19h) on 14,039 items from 61 scans
- 11,381 rendered perspective images across 61 scans
- 90 MP3D scans downloaded locally (61 with R2R coverage)
- Speaker v1 generates reasonable navigation instructions

## Next Steps (on server)
1. Test SFT v2 speaker quality
2. Run flywheel Round 1: generate candidates → judge → build preferences → DPO
3. Evaluate with BLEU/METEOR/CIDEr metrics
4. Iterate flywheel rounds

## Configs
- `configs/4080.yaml`: Conservative for 12GB VRAM laptop (batch=1, max_images=4)
- `configs/pro6000.yaml`: Aggressive for 48GB server (batch=4, max_images=12, local judge)
- `configs/default.yaml`: Reference defaults

## CLI
```bash
python run.py render --config configs/pro6000.yaml
python run.py sft --config configs/pro6000.yaml
python run.py flywheel --config configs/pro6000.yaml
python run.py evaluate --config configs/pro6000.yaml
```

## Dependencies
torch, transformers, peft, bitsandbytes, trl, datasets, accelerate,
qwen-vl-utils, Pillow, numpy, opencv-python-headless, tqdm, fire, pyyaml

## Environment Setup (Server)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft bitsandbytes trl datasets
pip install qwen-vl-utils Pillow numpy opencv-python-headless tqdm fire pyyaml
export HF_ENDPOINT=https://hf-mirror.com  # AutoDL mirror
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct
```
