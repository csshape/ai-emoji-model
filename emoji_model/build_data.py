"""Stream social-media corpora and mine (text, emoji-labels) pairs.

No manual annotation is needed: text that already contains emoji is its own
label source. We remove the emoji to form the input and keep them as targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .emoji_utils import clean_text, dedup_preserve_order, split_emoji

SPAM_RE = re.compile(
    r"\b(giveaway|winner|retweet|rt|airdrop|whitelist|nft|presale|"
    r"follow me|link in bio|insight by)\b", re.I)
SHOUT_RE = re.compile(r"\b[A-Z]{4,}\b")


def is_conversational(text: str) -> bool:
    """Keep only text that reads like a message, not a broadcast.

    An earlier, stricter version also dropped every text containing a link.
    That lifted recall@5 on short messages by 0.045, but it gutted the concrete
    emoji -- 📱 fell 71%, 🍔 63%, ☕ 27% in the training data -- because tweets
    that mention products are exactly the ones carrying links. "just had the
    best coffee ☕ [link]" is a fine message. So links are allowed now, and the
    filter targets what actually marks a broadcast: hashtag stuffing, mention
    chains, sales phrases and shouting.
    """
    if text.count("#") >= 2 or text.count("<user>") >= 3:
        return False
    if SPAM_RE.search(text) or SHOUT_RE.search(text):
        return False
    return True


MIN_WORDS = 3
MAX_WORDS = 48
MAX_LABELS = 5


@dataclass
class Source:
    name: str
    dataset: str
    config: str | None
    split: str
    column: str
    lang: str
    target: int          # how many mined examples we want from this source
    scan_limit: int      # give up after this many rows
    conversational: bool = False   # drop broadcast/marketing text


SOURCES: list[Source] = [
    # Tweets are by far the richest emoji source (~26% of rows carry one) and
    # they match the target use case: short, conversational messages.
    Source("twitter_en", "enryu43/twitter100m_tweets", None, "train", "tweet", "en",
           300_000, 12_000_000, conversational=True),
    # Danish Reddit is sparser (~2-3%) but it is the only sizeable Danish
    # corpus with natural emoji use, so we take everything it has.
    Source("reddit_da", "alexandrainst/scandi-reddit", "da", "train", "doc", "da", 250_000, 5_000_000),
]


def mine_source(src: Source, writer, seen: set[str]) -> dict:
    """Stream one corpus, yielding filtered examples into `writer`."""
    stats = {"scanned": 0, "with_emoji": 0, "kept": 0}
    try:
        ds = load_dataset(src.dataset, src.config, split=src.split, streaming=True)
    except Exception as exc:  # a source being unavailable must not kill the run
        print(f"  !! {src.name}: could not load ({type(exc).__name__}: {exc})")
        return stats

    bar = tqdm(total=src.target, desc=f"  {src.name}", unit="ex")
    for row in ds:
        stats["scanned"] += 1
        if stats["scanned"] > src.scan_limit or stats["kept"] >= src.target:
            break
        raw = row.get(src.column)
        if not raw or not isinstance(raw, str):
            continue
        if raw in ("[deleted]", "[removed]"):
            continue

        body, emojis = split_emoji(raw)
        if not emojis:
            continue
        stats["with_emoji"] += 1

        text = clean_text(body)
        if src.conversational and not is_conversational(text):
            continue
        n_words = text.count(" ") + 1
        if n_words < MIN_WORDS or n_words > MAX_WORDS or len(text) < 8:
            continue

        labels = dedup_preserve_order(emojis)[:MAX_LABELS]
        key = hashlib.blake2b(text.lower().encode(), digest_size=12).digest()
        key_hex = key.hex()
        if key_hex in seen:
            continue
        seen.add(key_hex)

        writer.write(json.dumps(
            {"text": text, "labels": labels, "lang": src.lang, "src": src.name},
            ensure_ascii=False) + "\n")
        stats["kept"] += 1
        bar.update(1)
    bar.close()

    pct = 100 * stats["with_emoji"] / max(stats["scanned"], 1)
    print(f"  {src.name}: scanned {stats['scanned']:,} | {pct:.1f}% had emoji | kept {stats['kept']:,}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mined.jsonl")
    ap.add_argument("--only", nargs="*", help="restrict to these source names")
    ap.add_argument("--target", type=int, help="override per-source example target")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [s for s in SOURCES if not args.only or s.name in args.only]
    if args.target:
        for s in sources:
            s.target = args.target

    seen: set[str] = set()
    with out_path.open("w", encoding="utf-8") as writer:
        for src in sources:
            print(f"\n=== {src.name} ({src.dataset}) ===")
            mine_source(src, writer, seen)

    print(f"\nWrote {len(seen):,} unique examples -> {out_path}")


if __name__ == "__main__":
    main()
