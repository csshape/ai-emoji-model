"""Collect Danish messages that carry NO emoji, ready for LLM annotation.

Our training data only kept the ~6% of Danish Reddit comments that happened to
contain an emoji. The other 94% is perfectly good Danish conversation that we
threw away for lack of a label -- an LLM can supply that label, and it picks
content-bearing emoji (🍕 for "en god rulle") where humans default to 😊.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .emoji_utils import clean_text, split_emoji

MIN_WORDS, MAX_WORDS = 3, 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/unlabeled_da.jsonl")
    ap.add_argument("--target", type=int, default=60_000)
    ap.add_argument("--scan-limit", type=int, default=3_000_000)
    ap.add_argument("--exclude", default="data/mined.jsonl",
                    help="skip texts already in the training set")
    args = ap.parse_args()

    seen: set[str] = set()
    exclude = Path(args.exclude)
    if exclude.exists():
        with exclude.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)["text"]
                    seen.add(hashlib.blake2b(t.lower().encode(), digest_size=12).hexdigest())
        print(f"excluding {len(seen):,} texts already used for training")

    ds = load_dataset("alexandrainst/scandi-reddit", "da", split="train", streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = scanned = 0
    bar = tqdm(total=args.target, unit="msg")
    with out.open("w", encoding="utf-8") as w:
        for row in ds:
            scanned += 1
            if scanned > args.scan_limit or kept >= args.target:
                break
            raw = row.get("doc")
            if not raw or not isinstance(raw, str) or raw in ("[deleted]", "[removed]"):
                continue
            body, emojis = split_emoji(raw)
            if emojis:
                continue          # already labelled -- those are in mined.jsonl
            text = clean_text(body)
            n = len(text.split())
            if n < MIN_WORDS or n > MAX_WORDS:
                continue
            key = hashlib.blake2b(text.lower().encode(), digest_size=12).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            w.write(json.dumps({"text": text, "lang": "da"}, ensure_ascii=False) + "\n")
            kept += 1
            bar.update(1)
    bar.close()
    print(f"scanned {scanned:,} | wrote {kept:,} unlabeled Danish messages -> {out}")


if __name__ == "__main__":
    main()
