"""Predict 1-5 emoji for a piece of text.

How many emoji come back is driven by the model's own confidence rather than a
fixed k: always return the best one, then keep adding while the next candidate
is both plausible in absolute terms and close to the leader.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .emoji_utils import clean_text, split_emoji
from .model import EmojiEncoder, ModelConfig
from .tokenizer import Tokenizer
from .vocab import EmojiVocab

MIN_EMOJI, MAX_EMOJI = 1, 5

# How hard to discount emoji that are simply common. Without this the model
# leans on 😂 (109x the median frequency in training data) whenever it is
# unsure. Kept mild: it costs ~1 point of recall@5 because the metric rewards
# matching the original author, who often did just use 😂. 0 disables it.
DEFAULT_PRIOR_ALPHA = 0.25


class EmojiPredictor:
    def __init__(self, checkpoint: str | Path = "checkpoints/best.pt",
                 data_dir: str | Path = "data",
                 device: str | None = None) -> None:
        data_dir = Path(data_dir)
        blob = torch.load(checkpoint, map_location="cpu")
        self.cfg = ModelConfig(**blob["config"])
        self.model = EmojiEncoder(self.cfg)
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.tok = Tokenizer(data_dir / "spm.model")
        self.vocab = EmojiVocab.load(data_dir / "emoji_vocab.json")
        prior_path = data_dir / "emoji_prior.pt"
        self.prior = torch.load(prior_path) if prior_path.exists() else None

    @torch.no_grad()
    def probs(self, text: str) -> torch.Tensor:
        body, _ = split_emoji(text)
        ids = torch.tensor([self.tok.encode(clean_text(body), self.cfg.max_len)],
                           dtype=torch.long, device=self.device)
        return torch.sigmoid(self.model(ids))[0]

    def predict(self, text: str, abs_threshold: float = 0.08,
                rel_ratio: float = 0.35,
                prior_alpha: float | None = None) -> list[tuple[str, float]]:
        p = self.probs(text)
        alpha = DEFAULT_PRIOR_ALPHA if prior_alpha is None else prior_alpha
        if self.prior is not None and alpha > 0:
            # Rank by how much this emoji beats its base rate, then rescale so
            # the reported numbers stay on the original probability scale.
            adjusted = p / self.prior.clamp(min=1e-6).pow(alpha)
            order = adjusted.argsort(descending=True)[:MAX_EMOJI]
            scores = p[order].tolist()
            idx = order.tolist()
            out = [(self.vocab.emojis[idx[0]], scores[0])]
            for s_, i_ in zip(scores[1:], idx[1:]):
                if s_ >= abs_threshold and s_ >= rel_ratio * scores[0]:
                    out.append((self.vocab.emojis[i_], s_))
            return out[:MAX_EMOJI]
        top = p.topk(MAX_EMOJI)
        scores, idx = top.values.tolist(), top.indices.tolist()
        out = [(self.vocab.emojis[idx[0]], scores[0])]
        for s, i in zip(scores[1:], idx[1:]):
            if s >= abs_threshold and s >= rel_ratio * scores[0]:
                out.append((self.vocab.emojis[i], s))
        return out[:MAX_EMOJI]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reply to text with 1-5 emoji")
    ap.add_argument("text", nargs="*", help="text to react to; omit for interactive mode")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--scores", action="store_true", help="show probabilities")
    ap.add_argument("--prior-alpha", type=float, default=None,
                    help="discount common emoji (0 disables; default 0.25)")
    args = ap.parse_args()

    predictor = EmojiPredictor(args.checkpoint, args.data)

    def show(text: str) -> None:
        preds = predictor.predict(text, prior_alpha=args.prior_alpha)
        emojis = "".join(e for e, _ in preds)
        if args.scores:
            detail = "  ".join(f"{e} {s:.2f}" for e, s in preds)
            print(f"{emojis}   [{detail}]")
        else:
            print(emojis)

    if args.text:
        show(" ".join(args.text))
        return

    print("Skriv en tekst, få emoji tilbage. Ctrl-C for at stoppe.\n")
    try:
        while True:
            line = input("> ").strip()
            if line:
                show(line)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
