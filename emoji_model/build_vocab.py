"""Choose the 512 emoji the model may answer with.

Frequency in mined text alone gives a Twitter-shaped label space: plenty of
✨🔥🚀💯 and 28 flags, no 🚲 or 🔑. The LLM relabelling is the direction the
training data is moving, so its proposals get a vote too -- scaled up to what
the full run would produce, since only part of it has been done.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Flags are 1.5% of observed use, never a sensible reply to a chat message, and
# rare enough that prior correction inflates them into nonsense answers.
def is_flag(e: str) -> bool:
    return 0x1F1E6 <= ord(e[0]) <= 0x1F1FF or ord(e[0]) == 0x1F3F4


def counts(path: Path) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                c.update(json.loads(line)["labels"])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, default=Path("data/mined.jsonl"))
    ap.add_argument("--relabeled", type=Path, default=Path("data/relabeled.jsonl"))
    ap.add_argument("--target", type=int, default=102726,
                    help="examples the full relabel run will cover")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--llm-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("data/emoji_vocab_v2.json"))
    args = ap.parse_args()

    human = counts(args.mined)
    llm = counts(args.relabeled) if args.relabeled.exists() else collections.Counter()
    done = sum(1 for _ in args.relabeled.open(encoding="utf-8")) if args.relabeled.exists() else 0
    scale = (args.target / done) if done else 0.0

    score: collections.Counter = collections.Counter()
    for e, n in human.items():
        if not is_flag(e):
            score[e] += n
    for e, n in llm.items():
        if not is_flag(e):
            score[e] += n * scale * args.llm_weight

    kept = [e for e, _ in score.most_common(args.size)]
    old = json.loads(Path("data/emoji_vocab.json").read_text(encoding="utf-8"))["emojis"]
    added = [e for e in kept if e not in set(old)]
    dropped = [e for e in old if e not in set(kept)]

    args.out.write_text(json.dumps({"emojis": kept}, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print(f"relabelled so far: {done:,} of {args.target:,}  (LLM votes scaled x{scale:.1f})")
    print(f"vocabulary: {len(kept)} emoji -> {args.out}")
    print(f"\nadded {len(added)}:")
    print("   " + "".join(added))
    print(f"\ndropped {len(dropped)}:")
    print("   " + "".join(dropped))

    tot = sum(human.values())
    covered = sum(human[e] for e in kept)
    old_cov = sum(human[e] for e in old)
    print(f"\nhuman-label coverage: {100*old_cov/tot:.1f}% -> {100*covered/tot:.1f}%")


if __name__ == "__main__":
    main()
