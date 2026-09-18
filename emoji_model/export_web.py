"""Export a checkpoint to flat files a browser can run without any ML runtime.

Writes, per model:  <name>.bin   float32 weights, concatenated in manifest order
                    <name>.json  config, tensor manifest, tokenizer, emoji, prior

The JS side (report/model/emoji.js) reimplements the forward pass; keeping the
export dumb means the two stay easy to diff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from .model import EmojiEncoder, ModelConfig


def export(checkpoint: Path, data_dir: Path, out_dir: Path, name: str) -> None:
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    if cfg.label_head:
        model.bake_label_head()

    manifest, chunks, offset = [], [], 0
    for key, tensor in model.state_dict().items():
        if key == "label_tokens":
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32).ravel()
        manifest.append({"name": key, "shape": list(tensor.shape), "offset": offset})
        chunks.append(arr)
        offset += arr.size

    out_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).tofile(out_dir / f"{name}.bin")

    sp = spm.SentencePieceProcessor(model_file=str(data_dir / "spm.model"))
    pieces = [sp.id_to_piece(i) for i in range(len(sp))]
    scores = [round(sp.get_score(i), 4) for i in range(len(sp))]

    emoji = json.loads((data_dir / "emoji_vocab.json").read_text(encoding="utf-8"))["emojis"]
    prior_path = data_dir / "emoji_prior.pt"
    prior = torch.load(prior_path).tolist() if prior_path.exists() else None

    (out_dir / f"{name}.json").write_text(json.dumps({
        "config": cfg.to_dict(),
        "n_floats": offset,
        "tensors": manifest,
        "pieces": pieces,
        "scores": scores,
        "emoji": emoji,
        "prior": [round(p, 9) for p in prior] if prior else None,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    mb = (out_dir / f"{name}.bin").stat().st_size / 1e6
    print(f"{name}: {offset:,} floats ({mb:.1f} MB) · {len(pieces):,} pieces · {len(emoji)} emoji")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("report/model"))
    ap.add_argument("--model", action="append", required=True,
                    help="name:checkpoint:datadir")
    args = ap.parse_args()
    for spec in args.model:
        name, ckpt, ddir = spec.split(":")
        export(Path(ckpt), Path(ddir), args.out, name)


if __name__ == "__main__":
    main()
