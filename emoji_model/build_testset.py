"""Turn generated chat data into the test jsonl the evaluators read.

Accepts either shape:
  * the customer's feed schema  {"messages": [{"text": ..., "metadata": {...}}]}
  * lean generation output      {"text": "...", "phase": "..."}, one per line

Emoji stay inline in `text` in both -- they are split out here, not by the
generator, so the writing stays natural.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .emoji_utils import clean_text, split_emoji


def rows_from(path: Path):
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or raw.lstrip().startswith("{\"text\""):
        for line in raw.splitlines():
            if line.strip():
                d = json.loads(line)
                yield d["text"], d.get("phase", "?")
    else:
        for msg in json.loads(raw)["messages"]:
            yield msg["text"], msg.get("metadata", {}).get("phase", "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--lang", default="da")
    args = ap.parse_args()

    seen: set[str] = set()
    kept = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for path in args.inputs:
            for text, phase in rows_from(path):
                body, emoji = split_emoji(text)
                body = clean_text(body)
                if not emoji or not body.strip() or body in seen:
                    continue
                seen.add(body)
                fh.write(json.dumps({
                    "text": body,
                    "labels": emoji,
                    "lang": args.lang,
                    "n_words": len(body.split()),
                    "phase": phase,
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{kept} eksempler -> {args.out}")


if __name__ == "__main__":
    main()
