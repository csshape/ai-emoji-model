"""Measure whether the model names what a message is about.

recall@5 over all labels answers "did it agree with the human", which on Danish
mostly rewards answering 😂. This scores only the concrete emoji in the gold
label -- the food, places, transport and things -- so it answers the question
the product actually asks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .emoji_utils import is_concrete_emoji
from .model import EmojiEncoder, ModelConfig
from .tokenizer import Tokenizer
from .vocab import EmojiVocab


@torch.no_grad()
def score(name: str, checkpoint: Path, data_dir: Path, rows: list[dict],
          prior_alpha: float) -> dict:
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()

    tok = Tokenizer(data_dir / "spm.model")
    vocab = EmojiVocab.load(data_dir / "emoji_vocab.json")
    prior_path = data_dir / "emoji_prior.pt"
    prior = torch.load(prior_path) if prior_path.exists() else None

    keep, targets = [], []
    for r in rows:
        want = [e for e in r["labels"] if is_concrete_emoji(e)]
        ids = vocab.encode(want)
        if ids:
            keep.append(r)
            targets.append(set(ids))
    if not keep:
        return {"name": name, "n": 0}

    ids = torch.tensor([tok.encode(r["text"], cfg.max_len) for r in keep],
                       dtype=torch.long)
    logits = torch.cat([model(ids[i:i + 512]) for i in range(0, len(ids), 512)])
    probs = torch.sigmoid(logits)
    if prior is not None and prior_alpha > 0:
        probs = probs / prior.clamp(min=1e-6).pow(prior_alpha)

    top5 = probs.topk(5, dim=-1).indices.tolist()
    hit1 = hit5 = 0
    produced_concrete = 0
    for row_top, want in zip(top5, targets):
        if row_top[0] in want:
            hit1 += 1
        if want & set(row_top):
            hit5 += 1
        if is_concrete_emoji(vocab.emojis[row_top[0]]):
            produced_concrete += 1
    n = len(keep)
    return {"name": name, "n": n,
            "top1": hit1 / n, "recall@5": hit5 / n,
            "concrete_rate": produced_concrete / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/test_da_content.jsonl"))
    ap.add_argument("--models", nargs="+", required=True, help="name:checkpoint:datadir")
    ap.add_argument("--prior-alpha", type=float, default=0.25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.test.open(encoding="utf-8") if l.strip()]
    print(f"{args.test}: {len(rows):,} messages (alpha={args.prior_alpha})\n")
    print(f"{'model':<10}{'n':>6}{'top1':>9}{'recall@5':>11}{'svarer ting':>13}")
    print("-" * 49)
    for spec in args.models:
        name, ckpt, ddir = spec.split(":")
        s = score(name, Path(ckpt), Path(ddir), rows, args.prior_alpha)
        if not s["n"]:
            print(f"{name:<10}{'-':>6}")
            continue
        print(f"{s['name']:<10}{s['n']:>6}{s['top1']:>9.3f}"
              f"{s['recall@5']:>11.3f}{s['concrete_rate']:>12.0%}")


if __name__ == "__main__":
    main()
