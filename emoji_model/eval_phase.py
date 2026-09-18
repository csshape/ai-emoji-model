"""Evaluate a model on the diagnostic coverage feeds, broken down by phase.

The feeds carry a `metadata.phase` stratum per message, so we can see whether a
failure is spread out or concentrated in one category (objects, anger, ...).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

from .emoji_utils import split_emoji
from .model import EmojiEncoder, ModelConfig
from .tokenizer import Tokenizer
from .vocab import EmojiVocab


def load_feeds(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for msg in data["messages"]:
            text, emoji = split_emoji(msg["text"])
            if not emoji or not text.strip():
                continue
            rows.append({
                "text": text,
                "labels": emoji,
                "phase": msg.get("metadata", {}).get("phase", "?"),
            })
    return rows


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("feeds", nargs="+", type=Path)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt", type=Path)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--prior-alpha", type=float, default=0.25)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()

    tok = Tokenizer(args.data / "spm.model")
    vocab = EmojiVocab.load(args.data / "emoji_vocab.json")
    prior_path = args.data / "emoji_prior.pt"
    prior = torch.load(prior_path) if prior_path.exists() else None

    rows = load_feeds(args.feeds)
    keep = [r for r in rows if vocab.encode(r["labels"])]
    dropped = len(rows) - len(keep)

    ids = torch.tensor([tok.encode(r["text"], cfg.max_len) for r in keep], dtype=torch.long)
    tgt = torch.zeros((len(keep), len(vocab)))
    for i, r in enumerate(keep):
        for j in vocab.encode(r["labels"]):
            tgt[i, j] = 1.0

    logits = torch.cat([model(ids[i:i + 512]) for i in range(0, len(ids), 512)])
    probs = torch.sigmoid(logits)
    if prior is not None and args.prior_alpha > 0:
        probs = probs / prior.clamp(min=1e-6).pow(args.prior_alpha)

    hits = tgt.gather(1, probs.topk(5, dim=-1).indices)
    top1, rec5 = hits[:, 0], hits.sum(1) / tgt.sum(1).clamp(min=1)

    print(f"{args.checkpoint}  alpha={args.prior_alpha}")
    print(f"{len(keep)} eksempler ({dropped} kasseret: label uden for vokabular)\n")
    print(f"{'kategori':<18}{'n':>5}{'top1':>8}{'recall@5':>10}")
    print("-" * 41)
    print(f"{'ALLE':<18}{len(keep):>5}{top1.mean():>8.3f}{rec5.mean():>10.3f}")

    by = collections.defaultdict(list)
    for i, r in enumerate(keep):
        by[r["phase"]].append(i)
    for phase, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        sel = torch.tensor(idx)
        print(f"{phase:<18}{len(idx):>5}{top1[sel].mean():>8.3f}{rec5[sel].mean():>10.3f}")


if __name__ == "__main__":
    main()
