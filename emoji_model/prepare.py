"""Turn mined JSONL into a tokenizer, an emoji vocabulary and packed tensors."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .model import ModelConfig
from .labels import tokenize_descriptions
from .tokenizer import Tokenizer, build_word_list, train_tokenizer
from .vocab import EmojiVocab


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="data/mined.jsonl")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--emoji-vocab", type=int, default=512)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--text-vocab", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="jsonl whose texts must be kept out of training")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="fixed emoji vocabulary json; default is built from frequency")
    ap.add_argument("--boost", action="append", default=None, metavar="PATH[:N]",
                    help="jsonl mixed into TRAIN only, never val; repeat N times. "
                         "May be given more than once with different weights.")
    ap.add_argument("--boost-repeat", type=int, default=1,
                    help="default repeat for --boost entries without :N")
    ap.add_argument("--dictionary", help="word list whose entries stay whole tokens")
    ap.add_argument("--protect-words", type=int, default=2000)
    ap.add_argument("--cap-per-emoji", type=int,
                    help="keep at most N examples per emoji, so 😂 (56k) stops "
                         "drowning 📝 (676)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.mined))
    print(f"loaded {len(rows):,} mined examples")
    print("  by language:", dict(Counter(r["lang"] for r in rows)))
    print("  by source:  ", dict(Counter(r["src"] for r in rows)))

    if args.exclude:
        # Held-out test rows. Without this the test set is inside the training
        # set and every number measured on it is meaningless.
        blocked = {json.loads(l)["text"]
                   for l in args.exclude.open(encoding="utf-8") if l.strip()}
        before = len(rows)
        rows = [r for r in rows if r["text"] not in blocked]
        print(f"\nexcluded {before - len(rows):,} held-out rows "
              f"({args.exclude})")

    # --- emoji label space -------------------------------------------------
    if args.vocab:
        # A vocabulary chosen elsewhere (see build_vocab.py), so the label space
        # can be steered rather than inherited from raw frequency.
        vocab = EmojiVocab.load(args.vocab)
        print(f"\nusing supplied vocabulary: {args.vocab}")
    else:
        vocab = EmojiVocab.build((r["labels"] for r in rows),
                                 size=args.emoji_vocab, min_count=args.min_count)
    vocab.save(outdir / "emoji_vocab.json")
    print(f"\nemoji vocabulary: {len(vocab)} labels")
    print("  most common:", " ".join(vocab.emojis[:20]))

    rows = [r for r in rows if vocab.encode(r["labels"])]
    print(f"examples with >=1 in-vocab label: {len(rows):,}")

    if args.cap_per_emoji:
        # Multi-label makes this a judgement call: an example carrying both 😂
        # and 📝 must survive for 📝's sake. So keep a row if ANY of its emoji
        # is still under quota -- common emoji get trimmed, rare ones keep
        # every example they appear in.
        random.seed(0)
        random.shuffle(rows)
        seen: Counter = Counter()
        capped = []
        for r in rows:
            labels = [vocab.emojis[i] for i in vocab.encode(r["labels"])]
            if any(seen[e] < args.cap_per_emoji for e in labels):
                capped.append(r)
                seen.update(labels)
        before = Counter(e for r in rows for e in
                         (vocab.emojis[i] for i in vocab.encode(r["labels"])))
        print(f"capped at {args.cap_per_emoji}/emoji: {len(rows):,} -> {len(capped):,} examples")
        top = before.most_common(3)
        print("   " + "  ".join(f"{e} {before[e]:,}->{seen[e]:,}" for e, _ in top))
        rows = capped

    # --- subword tokenizer -------------------------------------------------
    random.seed(0)
    random.shuffle(rows)
    corpus = outdir / "corpus.txt"
    corpus.write_text("\n".join(r["text"] for r in rows), encoding="utf-8")
    protected = None
    if args.dictionary:
        protected = build_word_list(corpus, args.dictionary, max_words=args.protect_words)
        print(f"\nprotecting {len(protected)} dictionary words from splitting, "
              f"e.g. {', '.join(protected[:8])}")
    print("\ntraining sentencepiece...")
    model_path = train_tokenizer(corpus, outdir / "spm", vocab_size=args.text_vocab,
                                 protected_words=protected)
    tok = Tokenizer(model_path)
    print(f"subword vocabulary: {len(tok)}")

    # --- pack tensors ------------------------------------------------------
    n_val = max(1, int(len(rows) * args.val_frac))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    # Boost rows join AFTER the split, so a repeated example can never appear
    # in validation -- that would make the val score meaningless.
    if args.boost:
        val_texts = {r["text"] for r in splits["val"]}
        held_out = set()
        if args.exclude:
            held_out = {json.loads(l)["text"]
                        for l in args.exclude.open(encoding="utf-8") if l.strip()}
        for spec in args.boost:
            path, _, times = spec.partition(":")
            repeat = int(times) if times else args.boost_repeat
            rows_b = [json.loads(l) for l in Path(path).open(encoding="utf-8")
                      if l.strip()]
            before = len(rows_b)
            rows_b = [r for r in rows_b
                      if r["text"] not in val_texts and r["text"] not in held_out]
            splits["train"] = splits["train"] + rows_b * repeat
            dropped = before - len(rows_b)
            print(f"\nboost {path}: {len(rows_b):,} x{repeat} = "
                  f"{len(rows_b) * repeat:,} rows into train"
                  + (f" ({dropped} dropped as val/held-out)" if dropped else ""))
    for name, subset in splits.items():
        ids = torch.zeros((len(subset), args.max_len), dtype=torch.int32)
        targets = torch.zeros((len(subset), len(vocab)), dtype=torch.bool)
        langs = torch.zeros(len(subset), dtype=torch.uint8)
        for i, r in enumerate(subset):
            ids[i] = torch.tensor(tok.encode(r["text"], args.max_len), dtype=torch.int32)
            for j in vocab.encode(r["labels"]):
                targets[i, j] = True
            langs[i] = 1 if r["lang"] == "da" else 0
        torch.save({"input_ids": ids, "targets": targets, "langs": langs},
                   outdir / f"{name}.pt")
        if name == "train":
            counts = targets.sum(0).float()
            torch.save(counts / counts.sum(), outdir / "emoji_prior.pt")
        print(f"  {name}: {len(subset):,} examples -> {name}.pt")

    label_tokens, descriptions = tokenize_descriptions(
        vocab.emojis, tok, ModelConfig().label_len, cache_dir=outdir)
    torch.save(label_tokens, outdir / "label_tokens.pt")
    print(f"\nemoji descriptions: {descriptions[0][:60]}...")

    cfg = ModelConfig(vocab_size=len(tok), n_emoji=len(vocab), max_len=args.max_len)
    (outdir / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    print(f"\nmodel config written; label density "
          f"{sum(len(vocab.encode(r['labels'])) for r in rows)/len(rows):.2f} labels/example")


if __name__ == "__main__":
    main()
