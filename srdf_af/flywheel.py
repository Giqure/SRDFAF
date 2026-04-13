"""Multi-round self-improving flywheel orchestrator.

    Round 0:  SFT on R2R human annotations  →  Speaker_0
    Round k:  Speaker_{k-1} generates candidates
              → VLM Judge scores them
              → Build preference pairs  (DPO)  /  prompt set  (GRPO)
              → Train Speaker_k
    Repeat for n_rounds.
"""

import json
import logging
from pathlib import Path

from srdf_af.config import Config
from srdf_af.data import (
    load_connectivity,
    load_r2r,
    resolve_images,
    to_dpo,
    to_grpo,
    to_sft,
    trajectory_headings,
)

log = logging.getLogger(__name__)


# ── Data Preparation Helpers ──────────────────────────────────────────


def _prepare_sft_data(cfg: Config, output_path: str) -> int:
    """Convert R2R annotations → SFT JSONL."""
    r2r = load_r2r(cfg.r2r_dir, cfg.split)
    conn_cache: dict[str, dict] = {}
    items: list[dict] = []

    for entry in r2r:
        scan = entry["scan"]
        if scan not in conn_cache:
            try:
                conn_cache[scan] = load_connectivity(cfg.connectivity_dir, scan)
            except FileNotFoundError:
                continue

        headings = trajectory_headings(
            conn_cache[scan], entry["path"], entry.get("heading", 0.0)
        )
        imgs = resolve_images(
            cfg.image_dir, scan, entry["path"], headings, cfg.max_images
        )
        if not imgs:
            continue
        for instr in entry["instructions"]:
            items.append(to_sft(imgs, instr))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def _prepare_grpo_data(cfg: Config, output_path: str) -> int:
    """Convert R2R annotations → GRPO prompt JSONL."""
    r2r = load_r2r(cfg.r2r_dir, cfg.split)
    conn_cache: dict[str, dict] = {}
    items: list[dict] = []

    for entry in r2r:
        scan = entry["scan"]
        if scan not in conn_cache:
            try:
                conn_cache[scan] = load_connectivity(cfg.connectivity_dir, scan)
            except FileNotFoundError:
                continue

        headings = trajectory_headings(
            conn_cache[scan], entry["path"], entry.get("heading", 0.0)
        )
        imgs = resolve_images(
            cfg.image_dir, scan, entry["path"], headings, cfg.max_images
        )
        if not imgs:
            continue
        items.append(to_grpo(imgs, entry["instructions"]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def _convert_pref_to_dpo(pref_path: str, dpo_path: str) -> int:
    """Convert preference JSONL → DPO training format."""
    count = 0
    with open(pref_path) as fin, open(dpo_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            item = to_dpo(entry["image_paths"], entry["chosen"], entry["rejected"])
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


# ── Main Flywheel ─────────────────────────────────────────────────────


def run_flywheel(cfg: Config) -> str:
    """Execute the full self-improving flywheel and return the final speaker path."""
    import torch

    from srdf_af.generate import APIGenerator, generate_all, generate_all_api
    from srdf_af.judge import APIJudge, LocalJudge
    from srdf_af.preference import build_preferences
    from srdf_af.train import load_model_and_processor, train_dpo, train_grpo, train_sft

    base = Path(cfg.output_dir)
    r2r = load_r2r(cfg.r2r_dir, cfg.split)

    # ── Round 0: SFT ─────────────────────────────────────────────
    log.info("═══ Round 0: SFT ═══")
    sft_data = str(base / "round_0" / "sft_data.jsonl")
    sft_model = str(base / "round_0" / "speaker")

    n = _prepare_sft_data(cfg, sft_data)
    log.info(f"SFT data: {n} items → {sft_data}")

    train_sft(cfg, sft_data, sft_model)
    log.info(f"SFT speaker → {sft_model}")

    current_speaker = sft_model

    # ── Rounds 1..K ──────────────────────────────────────────────
    for rnd in range(1, cfg.n_rounds + 1):
        tag = "DPO" if cfg.optim == "dpo" else "GRPO"
        log.info(f"═══ Round {rnd}/{cfg.n_rounds}: {tag} ═══")
        rd = base / f"round_{rnd}"
        rd.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate candidates with current speaker
        cand_path = str(rd / "candidates.jsonl")
        if cfg.speaker_api:
            api_gen = APIGenerator(
                cfg.speaker_api, cfg.speaker, cfg.speaker_api_key
            )
            n_gen = generate_all_api(
                api_gen,
                r2r,
                cfg.connectivity_dir,
                cfg.image_dir,
                cand_path,
                n=cfg.n_candidates,
                max_images=cfg.max_images,
                temp_range=(cfg.temp_low, cfg.temp_high),
                max_tokens=cfg.max_gen_tokens,
            )
        else:
            model, processor = load_model_and_processor(current_speaker, cfg.qlora)
            n_gen = generate_all(
                model,
                processor,
                r2r,
                cfg.connectivity_dir,
                cfg.image_dir,
                cand_path,
                n=cfg.n_candidates,
                max_images=cfg.max_images,
                temp_range=(cfg.temp_low, cfg.temp_high),
                max_tokens=cfg.max_gen_tokens,
            )
            del model, processor
            torch.cuda.empty_cache()
        log.info(f"Generated candidates for {n_gen} trajectories")

        new_speaker = str(rd / "speaker")

        if cfg.optim == "dpo":
            # Step 2: Judge candidates
            if cfg.judge_api:
                judge = APIJudge(cfg.judge_api, cfg.judge_model, cfg.judge_api_key)
            else:
                jm, jp = load_model_and_processor(cfg.judge_model, qlora=True)
                judge = LocalJudge(jm, jp)

            pref_path = str(rd / "preferences.jsonl")
            n_pref = build_preferences(
                cand_path, judge, pref_path, cfg.score_gap
            )
            log.info(f"Preference pairs: {n_pref}")

            if hasattr(judge, "model"):
                del judge, jm, jp  # type: ignore[possibly-undefined]
                torch.cuda.empty_cache()

            # Step 3: Convert → DPO format and train
            dpo_path = str(rd / "dpo_data.jsonl")
            _convert_pref_to_dpo(pref_path, dpo_path)
            train_dpo(cfg, dpo_path, current_speaker, new_speaker)

        elif cfg.optim == "grpo":
            grpo_path = str(rd / "grpo_data.jsonl")
            n_grpo = _prepare_grpo_data(cfg, grpo_path)
            log.info(f"GRPO prompts: {n_grpo}")
            train_grpo(cfg, grpo_path, current_speaker, new_speaker)

        current_speaker = new_speaker
        log.info(f"Round {rnd} done → {current_speaker}")

    log.info(f"Flywheel complete. Final speaker: {current_speaker}")
    return current_speaker
