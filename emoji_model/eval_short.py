"""Evaluate models on the held-out short-message test set.

Takes (checkpoint, data dir) pairs so models with different tokenizers can be
compared on the same texts -- the emoji vocabulary is shared, the subword one
is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import EmojiEncoder, ModelConfig
from .tokenizer import Tokenizer
from .vocab import EmojiVocab

BANDS = [(1, 2, "1-2 ord"), (3, 4, "3-4 ord"), (5, 8, "5-8 ord")]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@torch.no_grad()
def score(checkpoint: Path, data_dir: Path, rows: list[dict], prior_alpha: float) -> dict:
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()

    tok = Tokenizer(data_dir / "spm.model")
    vocab = EmojiVocab.load(data_dir / "emoji_vocab.json")
    prior_path = data_dir / "emoji_prior.pt"
    prior = torch.load(prior_path) if prior_path.exists() else None

    keep = [r for r in rows if vocab.encode(r["labels"])]
    ids = torch.tensor([tok.encode(r["text"], cfg.max_len) for r in keep], dtype=torch.long)
    tgt = torch.zeros((len(keep), len(vocab)))
    for i, r in enumerate(keep):
        for j in vocab.encode(r["labels"]):
            tgt[i, j] = 1.0

    logits = torch.cat([model(ids[i:i + 512]) for i in range(0, len(ids), 512)])
    probs = torch.sigmoid(logits)
    if prior is not None and prior_alpha > 0:
        probs = probs / prior.clamp(min=1e-6).pow(prior_alpha)

    top5 = probs.topk(5, dim=-1).indices
    hits = tgt.gather(1, top5)
    top1 = hits[:, 0]
    rec5 = hits.sum(1) / tgt.sum(1).clamp(min=1)

    words = torch.tensor([r["n_words"] for r in keep])
    langs = [r["lang"] for r in keep]
    out = {"n": len(keep), "top1": top1.mean().item(), "recall@5": rec5.mean().item()}
    for lo, hi, label in BANDS:
        m = (words >= lo) & (words <= hi)
        out[label] = (rec5[m].mean().item(), int(m.sum())) if m.sum() >= 10 else (float("nan"), int(m.sum()))
    for code in ("en", "da"):
        m = torch.tensor([l == code for l in langs])
        if m.sum() >= 10:
            out[code] = (rec5[m].mean().item(), int(m.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/short_test.jsonl")
    ap.add_argument("--models", nargs="+",
                    default=["current:checkpoints_clean/best.pt:data_clean",
                             "previous:checkpoints/best.pt:data"],
                    help="name:checkpoint:datadir")
    ap.add_argument("--prior-alpha", type=float, default=0.25)
    args = ap.parse_args()

    rows = load_rows(Path(args.test))
    print(f"short-message test set: {len(rows):,} examples "
          f"(alpha={args.prior_alpha})\n")

    results = {}
    for spec in args.models:
        name, ckpt, ddir = spec.split(":")
        results[name] = score(Path(ckpt), Path(ddir), rows, args.prior_alpha)

    cols = list(results)
    hdr = f"{'metrik':<16}" + "".join(f"{c:>12}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for key in ["n", "top1", "recall@5"]:
        vals = "".join(f"{results[c][key]:>12.3f}" if key != "n" else f"{results[c][key]:>12,}"
                       for c in cols)
        print(f"{key:<16}{vals}")
    print()
    for lo, hi, label in BANDS:
        n = results[cols[0]][label][1]
        vals = "".join(f"{results[c][label][0]:>12.3f}" for c in cols)
        print(f"{label + f' (n={n})':<16}{vals}")
    print()
    for code in ("en", "da"):
        if code in results[cols[0]]:
            n = results[cols[0]][code][1]
            vals = "".join(f"{results[c][code][0]:>12.3f}" for c in cols)
            print(f"{code + f' (n={n})':<16}{vals}")


if __name__ == "__main__":
    main()
