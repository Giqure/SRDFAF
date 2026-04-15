#!/usr/bin/env python3
"""Evaluate base Qwen3.5-9B vs server SFT on held-out R2R splits.

Same protocol as eval_heldout.py but for the server-trained model.
Compares: base model (no LoRA) vs SFT checkpoint.
Judge: DashScope API (qwen3.6-plus).
"""

import json
import os
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from srdf_af.config import Config
from srdf_af.data import load_connectivity, load_r2r, resolve_images, trajectory_headings
from srdf_af.generate import generate_candidates
from srdf_af.judge import APIJudge
from srdf_af.train import load_model_and_processor


# ── Config ────────────────────────────────────────────────────────────
CFG_PATH = os.environ.get("EVAL_CONFIG", "configs/pro6000.yaml")
SFT_CKPT = os.environ.get("EVAL_SFT", "/root/autodl-fs/output/srdf-af/sft_speaker")
JUDGE_API = os.environ.get("JUDGE_API", "https://dashscope.aliyuncs.com/compatible-mode/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3.6-plus")
JUDGE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MAX_IMAGES = 4
N_PER_SPLIT = 20


def prepare_trajectories(cfg, split: str, n: int):
    r2r = load_r2r(cfg.r2r_dir, split)
    conn_cache = {}
    usable = []
    for entry in r2r:
        scan = entry["scan"]
        if not os.path.isdir(f"{cfg.image_dir}/{scan}"):
            continue
        if scan not in conn_cache:
            try:
                conn_cache[scan] = load_connectivity(cfg.connectivity_dir, scan)
            except FileNotFoundError:
                continue
        headings = trajectory_headings(
            conn_cache[scan], entry["path"], entry.get("heading", 0.0)
        )
        imgs = resolve_images(cfg.image_dir, scan, entry["path"], headings, MAX_IMAGES)
        if not imgs:
            continue
        usable.append({
            "path_id": entry["path_id"],
            "scan": scan,
            "image_paths": imgs,
        })
    random.seed(42)
    random.shuffle(usable)
    selected = usable[:n]
    print(f"  {split}: {len(usable)} usable, selected {len(selected)}")
    return selected


def generate_for_all(model, processor, trajectories, label):
    preds = []
    for tr in tqdm(trajectories, desc=f"{label} gen"):
        cands = generate_candidates(
            model, processor, tr["image_paths"],
            n=1, temp_range=(0.7, 0.7), max_tokens=512,
        )
        preds.append(cands[0])
    return preds


def judge_and_compare(judge, trajectories, base_preds, sft_preds, split, out_path):
    results = []
    base_wins = sft_wins = ties = 0
    base_total = sft_total = 0.0

    for i, (tr, bp, sp) in enumerate(
        tqdm(zip(trajectories, base_preds, sft_preds),
             total=len(trajectories), desc=f"Judge {split}")
    ):
        bs = judge.score(tr["image_paths"], bp)
        ss = judge.score(tr["image_paths"], sp)
        if bs is None or ss is None:
            print(f"  [{i}] SKIP judge None")
            continue
        bw, sw = bs.weighted, ss.weighted
        base_total += bw
        sft_total += sw
        if sw > bw:
            sft_wins += 1
        elif bw > sw:
            base_wins += 1
        else:
            ties += 1
        results.append({
            "idx": i, "path_id": tr["path_id"],
            "base_pred": bp, "sft_pred": sp,
            "base_score": bw, "sft_score": sw,
            "delta": round(sw - bw, 4),
        })
        print(f"  [{i}] Base={bw:.3f}  SFT={sw:.3f}  Δ={sw-bw:+.3f}")

    n = base_wins + sft_wins + ties
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "split": split, "n": n,
        "avg_base": round(base_total / max(n, 1), 4),
        "avg_sft": round(sft_total / max(n, 1), 4),
        "base_wins": base_wins, "sft_wins": sft_wins, "ties": ties,
    }


def main():
    cfg = Config.load(CFG_PATH)
    print(f"Speaker: {cfg.speaker}")
    print(f"SFT ckpt: {SFT_CKPT}")
    print(f"Judge: {JUDGE_MODEL} via {JUDGE_API}")

    # Data
    print("\n=== Preparing data ===")
    val_seen = prepare_trajectories(cfg, "val_seen", N_PER_SPLIT)
    val_unseen = prepare_trajectories(cfg, "val_unseen", N_PER_SPLIT)
    all_traj = val_seen + val_unseen
    print(f"Total: {len(all_traj)} trajectories\n")

    # Base model (no LoRA)
    print("=== Loading base model ===")
    model, processor = load_model_and_processor(cfg.speaker, cfg.qlora)
    model.eval()
    base_preds = generate_for_all(model, processor, all_traj, "Base")
    del model
    torch.cuda.empty_cache()

    # SFT model
    print("\n=== Loading SFT model ===")
    model, processor = load_model_and_processor(cfg.speaker, cfg.qlora)
    model = PeftModel.from_pretrained(model, SFT_CKPT)
    model.eval()
    sft_preds = generate_for_all(model, processor, all_traj, "SFT")
    del model, processor
    torch.cuda.empty_cache()

    # Judge
    print("\n=== Judging ===")
    judge = APIJudge(JUDGE_API, JUDGE_MODEL, JUDGE_KEY)
    n_seen = len(val_seen)
    summaries = []

    if val_seen:
        s = judge_and_compare(
            judge, val_seen, base_preds[:n_seen], sft_preds[:n_seen],
            "val_seen", "/root/autodl-fs/output/srdf-af/eval_base_vs_sft_seen.jsonl",
        )
        summaries.append(s)
    if val_unseen:
        s = judge_and_compare(
            judge, val_unseen, base_preds[n_seen:], sft_preds[n_seen:],
            "val_unseen", "/root/autodl-fs/output/srdf-af/eval_base_vs_sft_unseen.jsonl",
        )
        summaries.append(s)

    # Report
    print("\n" + "=" * 65)
    print(f"{'Split':<15} {'Base avg':>9} {'SFT avg':>9} {'Base W':>7} {'SFT W':>6} {'Ties':>5}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['split']:<15} {s['avg_base']:>9.4f} {s['avg_sft']:>9.4f} {s['base_wins']:>7} {s['sft_wins']:>6} {s['ties']:>5}")
    total_n = sum(s["n"] for s in summaries)
    tb = sum(s["avg_base"] * s["n"] for s in summaries) / max(total_n, 1)
    ts = sum(s["avg_sft"] * s["n"] for s in summaries) / max(total_n, 1)
    tbw = sum(s["base_wins"] for s in summaries)
    tsw = sum(s["sft_wins"] for s in summaries)
    tt = sum(s["ties"] for s in summaries)
    print("-" * 65)
    print(f"{'Combined':<15} {tb:>9.4f} {ts:>9.4f} {tbw:>7} {tsw:>6} {tt:>5}")
    print("=" * 65)

    # Save summary
    summary = {"summaries": summaries, "combined": {
        "avg_base": round(tb, 4), "avg_sft": round(ts, 4),
        "base_wins": tbw, "sft_wins": tsw, "ties": tt,
    }}
    with open("/root/autodl-fs/output/srdf-af/eval_base_vs_sft_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
