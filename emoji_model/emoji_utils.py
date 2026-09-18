"""Emoji extraction and normalisation.

The training signal comes from text that already contains emoji: we strip the
emoji out to form the input, and use them as multi-label targets. Normalisation
matters a lot for vocabulary size -- skin-tone and presentation variants would
otherwise triple the label space for no semantic gain.
"""
from __future__ import annotations

import re
import unicodedata

import emoji as emoji_lib

SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
VARIATION_SELECTORS = {"︎", "️"}
ZWJ = "‍"

# Unicode classifies these as emoji, but nobody replies to a message with "©".
NON_EXPRESSIVE = {
    "\u00a9", "\u00ae", "\u2122",          # © ® ™
    "\u203c", "\u2049",                    # ‼ ⁉
    "\u2640", "\u2642", "\u26a7",          # ♀ ♂ ⚧ (gender signs, not replies)
    "\u0023", "\u002a",                    # # *
}
_KEYCAP_RE = re.compile(r"^[0-9#*]")


def is_expressive(seq: str) -> bool:
    """Filter out symbols that are technically emoji but never used as a reply."""
    if seq in NON_EXPRESSIVE or _KEYCAP_RE.match(seq):
        return False
    return True


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"(?<![\w/])@\w{2,30}")
_SUBREDDIT_RE = re.compile(r"(?<![\w/])/?r/\w{2,30}")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")


def normalise_emoji(seq: str) -> str:
    """Collapse an emoji sequence to its canonical base form.

    ``👍🏽`` -> ``👍`` and ``👨🏽‍💻`` -> ``👨‍💻``. If stripping the modifiers
    produces something no longer recognised as an emoji (keycaps, some flags),
    the original sequence is kept.
    """
    stripped = "".join(c for c in seq if c not in SKIN_TONES and c not in VARIATION_SELECTORS)
    if stripped and stripped in emoji_lib.EMOJI_DATA:
        return stripped
    # ZWJ sequences lose their joiner-less form; re-check with selectors removed only.
    if ZWJ in seq:
        no_tone = "".join(c for c in seq if c not in SKIN_TONES)
        if no_tone in emoji_lib.EMOJI_DATA:
            return no_tone
    return seq


def split_emoji(text: str) -> tuple[str, list[str]]:
    """Return (text without emoji, normalised emoji in order of appearance)."""
    found = emoji_lib.emoji_list(text)
    if not found:
        return text, []
    out: list[str] = []
    cursor = 0
    pieces: list[str] = []
    for match in found:
        pieces.append(text[cursor:match["match_start"]])
        cursor = match["match_end"]
        norm = normalise_emoji(match["emoji"])
        if is_expressive(norm):
            out.append(norm)
    pieces.append(text[cursor:])
    return " ".join(pieces), out


def clean_text(text: str) -> str:
    """Normalise the emoji-free text into the form the model sees at inference."""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" <url> ", text)
    text = _MENTION_RE.sub(" <user> ", text)
    text = _SUBREDDIT_RE.sub(" <sub> ", text)
    text = _REPEAT_RE.sub(r"\1\1\1", text)  # "sååååååå" -> "sååå"
    text = _WS_RE.sub(" ", text)
    return text.strip()


def dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
