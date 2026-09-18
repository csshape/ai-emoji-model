"""Evaluate the trained model, reported separately per language.

Danish is the scarce half of the training data, so an aggregate number would
hide a weak Danish model behind a strong English one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .infer import EmojiPredictor
from .model import EmojiEncoder, ModelConfig

PROBES_DA = [
    "det var en fantastisk dag",
    "jeg er så træt af det her",
    "tillykke med fødselsdagen",
    "skal vi spise pizza i aften?",
    "jeg elsker dig",
    "min hund døde i går",
    "hold da op hvor er det sjovt",
    "vi vandt kampen!",
    "jeg er nervøs for eksamen",
    "god morgen, sov du godt?",
]
PROBES_EN = [
    "that was an amazing day",
    "i am so tired of this",
    "happy birthday!",
    "want to grab pizza tonight?",
    "i love you",
    "my dog died yesterday",
    "this is hilarious",
    "we won the game!",
    "i'm nervous about the exam",
    "good morning, sleep well?",
]


@torch.no_grad()
def metrics_by_lang(checkpoint: Path, data_dir: Path) -> None:
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()

    val = torch.load(data_dir / "val.pt")
    ids, tgt, langs = val["input_ids"].long(), val["targets"].float(), val["langs"]

    print(f"{'lang':<8}{'n':>8}{'top1':>9}{'recall@5':>11}")
    for code, name in ((1, "da"), (0, "en")):
        mask = langs == code
        if not mask.any():
            continue
        sub_ids, sub_tgt = ids[mask], tgt[mask]
        top1 = 0.0
        rec5 = 0.0
        for i in range(0, len(sub_ids), 512):
            logits = model(sub_ids[i:i + 512])
            batch_tgt = sub_tgt[i:i + 512]
            top5 = logits.topk(5, dim=-1).indices
            hits = batch_tgt.gather(1, top5)
            top1 += hits[:, 0].sum().item()
            rec5 += (hits.sum(1) / batch_tgt.sum(1).clamp(min=1)).sum().item()
        n = len(sub_ids)
        print(f"{name:<8}{n:>8,}{top1/n:>9.3f}{rec5/n:>11.3f}")


def qualitative(checkpoint: Path, data_dir: Path) -> None:
    p = EmojiPredictor(checkpoint, data_dir)
    for title, probes in (("DANSK", PROBES_DA), ("ENGLISH", PROBES_EN)):
        print(f"\n--- {title} ---")
        for text in probes:
            preds = p.predict(text)
            emojis = "".join(e for e, _ in preds)
            print(f"  {emojis:<12} {text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    metrics_by_lang(Path(args.checkpoint), Path(args.data))
    qualitative(Path(args.checkpoint), Path(args.data))


if __name__ == "__main__":
    main()
