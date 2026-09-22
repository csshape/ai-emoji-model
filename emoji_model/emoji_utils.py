"""Emoji extraction and normalisation.

The training signal comes from text that already contains emoji: we strip the
emoji out to form the input, and use them as multi-label targets. Normalisation
matters a lot for vocabulary size -- skin-tone and presentation variants would
otherwise triple the label space for no semantic gain.
"""
from __future__ import annotations

import re
from pathlib import Path
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


# --- what kind of thing is this emoji? --------------------------------------
# Unicode groups these itself in emoji-test.txt, which beats any heuristic:
# name matching misses 🤯 (exploding head) and 🤣 (rolling on the floor
# laughing) because neither name contains "face", and codepoint ranges miss
# ☺ and 🥲 because they sit outside the Emoticons block.
_GROUP_FILE = Path(__file__).resolve().parent.parent / "data" / "emoji-test.txt"
_SOCIAL_GROUPS = {"Smileys & Emotion", "People & Body"}
_OBJECT_GROUPS = {"Animals & Nature", "Food & Drink", "Travel & Places",
                  "Activities", "Objects"}
# Unicode calls ✨ and 🔥 objects (Activities / Travel & Places), but in Danish
# chat they are decoration, not description -- ✨ is the single most common
# "object" emoji in the corpus and it turns up on messages about nothing in
# particular. These subgroups are the ones that actually name a thing.
_CONCRETE_SUBGROUPS = {
    "animal-mammal", "animal-bird", "animal-amphibian", "animal-reptile",
    "animal-marine", "animal-bug", "plant-flower", "plant-other",
    "food-fruit", "food-vegetable", "food-prepared", "food-asian",
    "food-sweet", "drink", "dishware",
    "place-map", "place-geographic", "place-building", "place-religious",
    "place-other", "transport-ground", "transport-water", "transport-air",
    "hotel", "sky & weather", "sport", "game", "arts & crafts", "clothing",
    "music", "musical-instrument", "phone", "computer", "light & video",
    "book-paper", "money", "mail", "writing", "office", "lock", "tool",
    "science", "medical", "household", "other-object", "time", "event",
}
# Inside those subgroups sit a handful that are used as emphasis rather than
# reference: ✨ is the most common "object" emoji in the Danish corpus and lands
# on messages about nothing in particular. Their subgroup-mates (🎂 🎁 ☀ 🌧)
# stay, so this is a deny-list, not a whole subgroup.
_DECORATIVE = set("✨🔥💫⭐🌟💥🎇🎆🚀💸💦💨🌈💐🌸🌹🌺🌻🌼🍀")
_KIND: dict[str, str] | None = None
_CONCRETE: set[str] | None = None


def _load_kinds() -> tuple[dict[str, str], set[str]]:
    kinds: dict[str, str] = {}
    concrete: set[str] = set()
    if not _GROUP_FILE.exists():
        return kinds, concrete
    group = sub = ""
    for line in _GROUP_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif line.startswith("# subgroup:"):
            sub = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#") and ";" in line:
            codes, _, rest = line.partition(";")
            if "fully-qualified" not in rest:
                continue
            char = "".join(chr(int(c, 16)) for c in codes.split())
            kind = ("social" if group in _SOCIAL_GROUPS else
                    "object" if group in _OBJECT_GROUPS else
                    "flag" if group == "Flags" else "symbol")
            key = normalise_emoji(char)
            kinds[key] = kind
            if sub in _CONCRETE_SUBGROUPS:
                concrete.add(key)
    return kinds, concrete


def emoji_kind(e: str) -> str:
    """Unicode's own grouping, collapsed to social / object / symbol / flag."""
    global _KIND, _CONCRETE
    if _KIND is None:
        _KIND, _CONCRETE = _load_kinds()
    return _KIND.get(normalise_emoji(e), "symbol")


def is_object_emoji(e: str) -> bool:
    return emoji_kind(e) == "object"


def is_concrete_emoji(e: str) -> bool:
    """Names an actual thing: food, drink, a place, transport, an animal.

    Narrower than is_object_emoji, which follows Unicode and so counts ✨ and
    🔥 as objects. This is the set worth measuring content behaviour against.
    """
    global _KIND, _CONCRETE
    if _KIND is None:
        _KIND, _CONCRETE = _load_kinds()
    key = normalise_emoji(e)
    return key in (_CONCRETE or set()) and key not in _DECORATIVE
