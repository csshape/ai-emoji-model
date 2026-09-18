"""Build semantic descriptions for each emoji from its name and keywords.

The plain model treats emoji as arbitrary output indices, so a rare emoji can
only be learned from its own training examples. Describing each emoji in words
("pizza cheese food hungry slice") lets it share representation with the text
encoder: the token "pizza" in a message and in 🍕's description are the same
embedding, so the link is there before a single example is seen.

Source is Unicode CLDR, which ships curated keywords -- real synonyms, not just
the canonical name.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import emoji as emoji_lib
import requests

from .emoji_utils import clean_text

CLDR_URL = "https://raw.githubusercontent.com/unicode-org/cldr/main/common/annotations/{lang}.xml"
VARIATION_SELECTOR = "️"
MIN_WORDS_BEFORE_PADDING = 3


def fetch_annotations(lang: str = "en", cache_dir: Path | None = None) -> dict[str, dict]:
    """Fetch CLDR emoji annotations, caching the XML locally."""
    cache = (cache_dir or Path("data")) / f"cldr_{lang}.xml"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        raw = requests.get(CLDR_URL.format(lang=lang), timeout=120).content
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)

    out: dict[str, dict] = {}
    for node in ET.fromstring(raw).iter("annotation"):
        key = "name" if node.get("type") == "tts" else "keywords"
        out.setdefault(node.get("cp"), {})[key] = node.text
    return out


def describe(emoji: str, annotations: dict[str, dict]) -> str:
    """One bag of words describing an emoji: CLDR name + keywords + alias."""
    entry = annotations.get(emoji) or annotations.get(emoji + VARIATION_SELECTOR) or {}
    parts: list[str] = []
    if entry.get("name"):
        parts.append(entry["name"])
    if entry.get("keywords"):
        parts.append(entry["keywords"].replace("|", " "))
    parts.append(emoji_lib.demojize(emoji).strip(":").replace("_", " "))

    text = " ".join(parts)
    # Flags have no CLDR keywords; a bare "Denmark" is too short to embed
    # stably and drifts toward arbitrary text.
    if len(text.split()) <= MIN_WORDS_BEFORE_PADDING:
        text = f"flag of {text} country nation"
    return clean_text(text)


def build_descriptions(emojis: list[str], lang: str = "en",
                       cache_dir: Path | None = None) -> list[str]:
    annotations = fetch_annotations(lang, cache_dir)
    return [describe(e, annotations) for e in emojis]


def tokenize_descriptions(emojis: list[str], tokenizer, label_len: int,
                          lang: str = "en", cache_dir: Path | None = None):
    """Tokenise each emoji description into a fixed-width id matrix."""
    import torch

    descriptions = build_descriptions(emojis, lang=lang, cache_dir=cache_dir)
    rows = [tokenizer.encode(d, label_len) for d in descriptions]
    return torch.tensor(rows, dtype=torch.long), descriptions
