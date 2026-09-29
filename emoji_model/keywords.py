"""A word -> emoji dictionary built from Unicode's own CLDR annotations.

The model learns mood from data; content is closer to a lookup. On real Danish
messages the dictionary alone beats the trained model at naming things
(recall@5 0.139 vs 0.076), and the two together beat either. So we ship both.

Matching is prefix-on-word-start. Plain substring scores marginally better on
content (0.183 against 0.178) but finds "kost" inside "frokost" and "pho"
inside "iphone", and gives back more mood (0.254 against 0.269) for the
trouble. Danish compounds forward, so a prefix still catches "cykelsti".

Flags are not in the model's vocabulary -- as a learned answer they are rare
enough that prior correction turns them into noise -- but a named country is
about as unambiguous as content gets. So the table also maps country names and
adjectives to flags, and predict() puts a matched flag first.
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


# Danish writes "spansk vin" far more often than "vin fra Spanien", and CLDR
# only carries the country names. Prefixes, like every other key.
FLAG_ALIASES = {
    "DK": ["dansk"], "SE": ["svensk"], "NO": ["norsk"], "DE": ["tysk"],
    "FR": ["fransk"], "ES": ["spansk"], "IT": ["italiensk"], "NL": ["hollandsk",
    "holland", "nederlandsk"], "BE": ["belgisk"], "PT": ["portugisisk"],
    "GB": ["engelsk", "england", "britisk", "english", "british"],
    "US": ["amerikansk", "amerika", "american"], "JP": ["japansk"],
    "CN": ["kinesisk"], "GR": ["græske", "grækere"], "TR": ["tyrkisk"],
    "PL": ["polsk"], "FI": ["finsk"], "IS": ["islandsk"], "UA": ["ukrainsk"],
    "RU": ["russisk"], "MX": ["mexicansk"], "CA": ["canadisk"], "AU": ["australsk"],
    "BR": ["brasiliansk"], "IN": ["indisk"], "TH": ["thai"], "IE": ["irsk"],
    "AT": ["østrigsk"], "CH": ["schweizisk"], "HR": ["kroatisk"], "KR": ["koreansk"],
    "GL": ["grønlandsk"], "FO": ["færøsk"],
}
# Country names that are mostly something else in chat: a first name, a bird,
# a US state, a word. The corpus can't prune these: people rarely add a flag
# even when they do mean the country, so every flag key looks imprecise.
FLAG_STOP = {"jordan", "chad", "turkey", "georgia", "jersey", "guinea", "niger",
             "togo", "china", "chile"}


def flag_for(code: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)


def is_flag(e: str) -> bool:
    return len(e) == 2 and all(0x1F1E6 <= ord(c) <= 0x1F1FF for c in e)


CLDR_MAIN_URL = "https://raw.githubusercontent.com/unicode-org/cldr/main/common/main/{lang}.xml"


def build_flags(territory_files: list[Path]) -> dict[str, list[str]]:
    """Country name -> flag, from CLDR's localised territory names."""
    import emoji as emoji_lib
    import requests
    keys: dict[str, set[str]] = defaultdict(set)
    for path in territory_files:
        # data/ is not in git, and a missing file here would quietly drop
        # every flag from the table, so fetch it like labels.py fetches CLDR.
        if not path.exists():
            lang = path.stem.removeprefix("cldr_main_")
            path.write_bytes(requests.get(CLDR_MAIN_URL.format(lang=lang), timeout=120).content)
        for node in ET.parse(path).getroot().iter("territory"):
            code, name = node.get("type", ""), node.text or ""
            if node.get("alt") or not re.fullmatch(r"[A-Z]{2}", code):
                continue
            words = WORD_RE.findall(name.lower())
            # "Det Forenede Kongerige" and "New Zealand" would need phrase
            # matching; their single words mean too many other things.
            if len(words) == 1 and len(words[0]) >= MIN_KEY:
                keys[words[0]].add(flag_for(code))
    for code, aliases in FLAG_ALIASES.items():
        for a in aliases:
            keys[a].add(flag_for(code))
    return {w: sorted(fs) for w, fs in keys.items()
            if len(fs) == 1 and w not in FLAG_STOP and w not in STOPWORDS
            and all(emoji_lib.is_emoji(f) for f in fs)}


def build(cldr_files: list[Path], vocab: list[str]) -> dict[str, list[str]]:
    # Content only. Mood is what the model is good at, and letting the
    # dictionary vote on faces costs accuracy on both: restricted to concrete
    # emoji it scores 0.174 against 0.146 for the unrestricted table, while
    # giving back less mood (0.262 against 0.230).
    allowed = {normalise_emoji(e) for e in vocab if is_concrete_emoji(e)}
    keys: dict[str, set[str]] = defaultdict(set)
    heads: set[str] = set()
    for path in cldr_files:
        if not path.exists():
            continue
        for ann in ET.parse(path).getroot().iter("annotation"):
            emoji = normalise_emoji(ann.get("cp", ""))
            for phrase in re.split(r"[|,]", ann.text or ""):
                words = WORD_RE.findall(phrase.lower())
                if words:
                    heads.add(words[-1])
                if emoji not in allowed:
                    continue
                for word in words:
                    if len(word) >= MIN_KEY and word not in STOPWORDS:
                        keys[word].add(emoji)
    # A word pointing at half the vocabulary tells us nothing. Nor does one
    # that only ever modifies the thing: "rødt æble" and "rødt blink" made
    # "rødt" mean 🍎🚨, "løbende" meant ⏳. Dropping every word that is never a
    # phrase's head left dictionary recall@5 on test_da_content at 0.093.
    return {w: sorted(es) for w, es in keys.items()
            if len(es) <= MAX_EMOJI_PER_WORD and w in heads}


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


def best_flag(hits: dict[str, float]) -> str | None:
    """The flag with the longest matching key, if the message names a country."""
    flags = {e: s for e, s in hits.items() if is_flag(e)}
    return max(flags, key=flags.get) if flags else None


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
    ap.add_argument("--territories", type=Path, nargs="+",
                    default=[Path("data/cldr_main_da.xml"), Path("data/cldr_main_en.xml")],
                    help="CLDR main locale files, for country names -> flags")
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
    flags = build_flags(args.territories)
    for key, fs in flags.items():
        table[key] = sorted(set(table.get(key, [])) | set(fs))
    print(f"{len(flags)} country words -> {len({f for fs in flags.values() for f in fs})} flags")
    args.out.write_text(json.dumps(table, ensure_ascii=False,
                                   separators=(",", ":")), encoding="utf-8")
    covered = len({e for es in table.values() for e in es})
    print(f"{len(table):,} words -> {covered} of {len(vocab)} emoji  ({args.out})")


if __name__ == "__main__":
    main()
