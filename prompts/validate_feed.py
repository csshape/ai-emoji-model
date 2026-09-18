"""Check an LLM-generated test feed against the spec before we trust it.

Usage:  uv run python prompts/validate_feed.py feeds/*.json
"""
import collections
import json
import re
import sys

sys.path.insert(0, ".")
from emoji_model.emoji_utils import split_emoji

VOCAB = set(json.load(open("data/emoji_vocab.json"))["emojis"])

# Anything that is not a face, gesture, heart or person counts as an object
# emoji -- those are the ones the model currently cannot produce.
PEOPLE_RANGES = (
    (0x1F600, 0x1F64F),  # faces and gestures
    (0x1F440, 0x1F450),  # eyes, hands
    (0x1F466, 0x1F487),  # people
    (0x1F90C, 0x1F92F),  # hand signs, more faces
    (0x1F9D0, 0x1F9DF),  # person roles
    (0x270A, 0x270D),    # fist, writing hand
)
HEARTS = set("❤\U0001F493\U0001F494\U0001F495\U0001F496\U0001F497"
             "\U0001F498\U0001F499\U0001F49A\U0001F49B\U0001F49C\U0001F49D"
             "\U0001F49E\U0001F5A4")
VARIATION = re.compile("[︎️\U0001F3FB-\U0001F3FF]")


def is_object(emoji: str) -> bool:
    if emoji in HEARTS:
        return False
    cp = ord(emoji[0])
    return not any(lo <= cp <= hi for lo, hi in PEOPLE_RANGES)


def check(path: str) -> bool:
    data = json.load(open(path, encoding="utf-8"))
    msgs = data["messages"]
    counts = collections.Counter()
    per_msg = collections.Counter()
    phases = collections.Counter()
    with_emoji = trailing_dot = at_end = 0

    for m in msgs:
        raw = VARIATION.sub("", m["text"]).rstrip()
        text, emoji = split_emoji(m["text"])
        phases[m.get("metadata", {}).get("phase", "?")] += 1
        if not emoji:
            continue
        with_emoji += 1
        counts.update(emoji)
        per_msg[len(emoji)] += 1
        if raw.endswith(tuple(emoji)):
            at_end += 1
        if re.search(r"[\U0001F300-\U0001FAFF☀-➿]\s*\.$", raw):
            trailing_dot += 1

    total = sum(counts.values())
    texts = [m["text"] for m in msgs]
    uniq = 100 * len(set(texts)) / len(texts)
    oov = {e: v for e, v in counts.items() if e not in VOCAB}
    objects = sum(v for e, v in counts.items() if is_object(e))
    top = counts.most_common()
    top3 = 100 * sum(v for _, v in top[:3]) / max(total, 1)
    rare = sum(1 for _, v in top if v <= 2)

    n = len(msgs)
    rate = 100 * with_emoji / n
    # Scale the coverage targets to the file size so any length validates.
    want_uniq_emoji = max(20, round(60 * n / 250))
    want_objects = max(12, round(37 * n / 250))

    checks = [
        ("beskeder",       n,                    n >= 100),
        ("unikke tekster", f"{uniq:.0f}%",       uniq >= 99),
        ("emoji-andel",    f"{rate:.0f}%",       30 <= rate <= 40),
        ("unikke emoji",   len(counts),          len(counts) >= want_uniq_emoji),
        ("uden for vokab", len(oov),             not oov),
        ("ting-forekomst", objects,              objects >= want_objects),
        ("top-3 andel",    f"{top3:.0f}%",       25 <= top3 <= 45),
        ("sjældne emoji",  rare,                 rare >= max(8, round(25 * n / 250))),
        ("emoji til sidst", f"{100*at_end/max(with_emoji,1):.0f}%",
                                                 at_end / max(with_emoji, 1) >= 0.6),
        ("punktum efter",  trailing_dot,         trailing_dot == 0),
        ("phase-kategorier", len(phases),        len(phases) >= 7),
    ]

    print(path)
    for name, value, ok in checks:
        print(f"  {'OK ' if ok else 'FEJL'}  {name:<17} {value}")
    if oov:
        print(f"        ^ ukendte: {' '.join(f'{e}x{v}' for e, v in oov.items())}")
    print(f"        top 10: {' '.join(f'{e}{v}' for e, v in top[:10])}")

    passed = all(ok for _, _, ok in checks)
    print("  =>", "GODKENDT" if passed else "AFVIST — bed ChatGPT rette de FEJL-linjer")
    print()
    return passed


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    print(f"{sum(results)}/{len(results)} filer godkendt")
    sys.exit(0 if all(results) else 1)
