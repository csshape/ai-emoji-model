"""Carve a Danish test set whose gold labels describe content, not punctuation.

The existing short-message set cannot measure what we are trying to build: 49%
of its Danish gold answers are generic smileys, so a model that replies 🍻 to
"Øl" is scored wrong because the human wrote 😂.

Rather than have an LLM write the labels -- which would only measure agreement
with that LLM -- this keeps messages where a Dane *did* reach for an object
emoji. The label is human, and it is the behaviour we want reproduced.

Rows written here must be excluded from training: see prepare.py --exclude.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .emoji_utils import is_concrete_emoji

PLACEHOLDER = re.compile(r"<(url|user|sub)>")
WORD = re.compile(r"[a-zA-ZæøåÆØÅ]{2,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, default=Path("data/mined.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/test_da_content.jsonl"))
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--max-words", type=int, default=20)
    ap.add_argument("--min-words", type=int, default=2)
    args = ap.parse_args()

    pool, seen = [], set()
    with args.mined.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["lang"] != "da" or r["text"] in seen:
                continue
            if not any(is_concrete_emoji(e) for e in r["labels"]):
                continue
            body = PLACEHOLDER.sub(" ", r["text"])
            words = WORD.findall(body)
            if not (args.min_words <= len(words) <= args.max_words):
                continue
            seen.add(r["text"])
            pool.append({"text": r["text"], "labels": r["labels"], "lang": "da",
                         "n_words": len(words)})

    random.seed(0)
    random.shuffle(pool)
    kept = pool[:args.size]
    with args.out.open("w", encoding="utf-8") as w:
        for r in kept:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    objs = sum(1 for r in kept if any(is_concrete_emoji(e) for e in r["labels"]))
    avg = sum(r["n_words"] for r in kept) / len(kept)
    print(f"pool: {len(pool):,} candidates -> kept {len(kept):,} -> {args.out}")
    print(f"  every row has a concrete emoji ({objs}/{len(kept)}), "
          f"{avg:.1f} words on average")


if __name__ == "__main__":
    main()
