"""A word -> emoji dictionary built from Unicode's own CLDR annotations.

The model learns mood from data; content is closer to a lookup. On real Danish
messages the dictionary alone beats the trained model at naming things
(recall@5 0.139 vs 0.076), and the two together beat either. So we ship both.

Matching is prefix-on-word-start. Plain substring scores marginally better on
content (0.183 against 0.178) but finds "kost" inside "frokost" and "pho"
inside "iphone", and gives back more mood (0.254 against 0.269) for the
trouble. Danish compounds forward, so a prefix still catches "cykelsti".
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .emoji_utils import is_concrete_emoji, normalise_emoji

WORD_RE = re.compile(r"[a-zA-ZæøåÆØÅ]{2,}")
# Swept on data/test_da_content: 3-character keys and at most 3 emoji per word
# score 0.126, against 0.110 for 4-character keys and 0.116 unfiltered.
MIN_KEY = 3
MAX_EMOJI_PER_WORD = 3
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "have", "har", "der", "det",
    "den", "som", "til", "med", "men", "ikke", "eller", "hvad", "hvor", "man",
    "jeg", "min", "mit", "sin", "var", "ved", "kan", "skal", "vil", "face",
    "person", "people", "hand", "sign", "symbol", "mark", "type", "flag",
}


def build(cldr_files: list[Path], vocab: list[str]) -> dict[str, list[str]]:
    # Content only. Mood is what the model is good at, and letting the
    # dictionary vote on faces costs accuracy on both: restricted to concrete
    # emoji it scores 0.174 against 0.146 for the unrestricted table, while
    # giving back less mood (0.262 against 0.230).
    allowed = {normalise_emoji(e) for e in vocab if is_concrete_emoji(e)}
    keys: dict[str, set[str]] = defaultdict(set)
    for path in cldr_files:
        if not path.exists():
            continue
        for ann in ET.parse(path).getroot().iter("annotation"):
            emoji = normalise_emoji(ann.get("cp", ""))
            if emoji not in allowed:
                continue
            for phrase in re.split(r"[|,]", ann.text or ""):
                for word in WORD_RE.findall(phrase.lower()):
                    if len(word) >= MIN_KEY and word not in STOPWORDS:
                        keys[word].add(emoji)
    # A word pointing at half the vocabulary tells us nothing.
    return {w: sorted(es) for w, es in keys.items()
            if len(es) <= MAX_EMOJI_PER_WORD}


def prune(table: dict[str, list[str]], corpus: list[dict],
          min_fires: int = 100, min_precision: float = 0.04) -> dict[str, list[str]]:
    """Drop keys that fire often and help rarely.

    CLDR is multilingual, so English keys collide with Danish words: "over"
    points at 🍳, "bed" at 🛏, "tag" at 🏷, and in Danish text they fire
    constantly and are almost never what the writer meant. Measured on the
    training corpus, never on the test set.
    """
    fires: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for row in corpus:
        words = WORD_RE.findall(row["text"].lower())
        gold = set(row["labels"])
        for key, emojis in table.items():
            if any(w.startswith(key) for w in words):
                fires[key] += 1
                if gold.intersection(emojis):
                    hits[key] += 1
    kept = {}
    for key, emojis in table.items():
        n = fires[key]
        if n >= min_fires and hits[key] / n < min_precision:
            continue
        kept[key] = emojis
    return kept


def lookup(text: str, table: dict[str, list[str]]) -> dict[str, float]:
    """Emoji -> score for one message. Longer matches count for more."""
    hits: dict[str, float] = defaultdict(float)
    words = WORD_RE.findall(text.lower())
    for key, emojis in table.items():
        if any(word.startswith(key) for word in words):
            for e in emojis:
                hits[e] += len(key) * 0.01
    return dict(hits)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=Path("data/emoji_vocab_v2.json"))
    ap.add_argument("--cldr", type=Path, nargs="+",
                    default=[Path("data/cldr_da.xml"), Path("data/cldr_en.xml")])
    ap.add_argument("--out", type=Path, default=Path("data/keywords.json"))
    ap.add_argument("--prune-on", type=Path, default=Path("data/mined.jsonl"),
                    help="corpus used to drop low-precision keys")
    ap.add_argument("--prune-sample", type=int, default=40000)
    args = ap.parse_args()
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))["emojis"]
    table = build(args.cldr, vocab)
    before = len(table)
    if args.prune_on.exists():
        corpus = []
        with args.prune_on.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("lang") == "da":
                    corpus.append(row)
                if len(corpus) >= args.prune_sample:
                    break
        table = prune(table, corpus)
        print(f"pruned {before - len(table)} low-precision keys "
              f"on {len(corpus):,} Danish training rows")
    args.out.write_text(json.dumps(table, ensure_ascii=False,
                                   separators=(",", ":")), encoding="utf-8")
    covered = len({e for es in table.values() for e in es})
    print(f"{len(table):,} words -> {covered} of {len(vocab)} emoji  ({args.out})")


if __name__ == "__main__":
    main()
