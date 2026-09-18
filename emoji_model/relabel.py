"""Replace generic smiley labels with mood + content emoji from a local LLM.

22% of the training set (120,501 examples) carries nothing but 😂😅😊 — those
are what teach the model to answer 😂 to everything. Adding better data on top
did not help: 13k LLM labels moved 😂's share from 6.76% to 6.55%, and boosting
them to 20% of the set made every independent metric worse. Replacing the weak
labels is the remaining option.

The prompt asks for mood first, then content: "Fedt!! jeg lever af at køre
lastbil i kbh" -> 🤩🚚💰. That keeps the reaction quality the chat feeds expect
while adding the concrete emoji the model currently lacks.

Resumable: rerun to continue where it stopped.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from .emoji_utils import dedup_preserve_order, split_emoji

GENERIC = {"😂", "😅", "😊", "🤣", "😉", "😄", "😁", "🙂", "☺", "😃", "😆", "🙃"}

SYSTEM = (
    "Du vælger emoji-svar til en dansk chatbesked. "
    "Giv 2-3 emoji der tilsammen fanger BÅDE stemningen (glad, træt, irriteret, "
    "spændt, ked af det) OG det konkrete som beskeden handler om (mad, drikke, "
    "sted, aktivitet, ting). Sæt stemnings-emoji først. "
    "Svar KUN med emoji — aldrig ord, aldrig forklaring."
)


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached -- abort rather than write fallback labels.

    Falling back to the original labels on a connection failure silently fills
    the output with exactly the ironic labels this script exists to replace,
    marked "src": "relabel" and indistinguishable from real work.
    """


def relabel(text: str, model: str) -> list[str]:
    try:
        r = requests.post("http://localhost:1234/v1/chat/completions", timeout=180, json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Besked: {text}"}],
            "temperature": 0.3, "max_tokens": 200,
            "reasoning_effort": "low", "stream": False,
        })
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
    except Exception as exc:
        raise LLMUnavailable(f"LLM call failed: {exc}") from exc
    out = msg.get("content") or msg.get("reasoning") or ""
    # An answer with no emoji in it is a real answer -- the caller may keep the
    # original label for that one row.
    return dedup_preserve_order(split_emoji(out)[1])[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="data/mined.jsonl")
    ap.add_argument("--out", default="data/relabeled.jsonl")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--lang", default="da", help="'da', 'en' or 'all'")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.mined).open(encoding="utf-8") if l.strip()]
    todo = [r for r in rows
            if all(e in GENERIC for e in r["labels"])
            and (args.lang == "all" or r["lang"] == args.lang)]

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["text"])
        print(f"resuming: {len(done):,} already relabelled")

    todo = [r for r in todo if r["text"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo):,} examples to relabel (lang={args.lang})")

    written = kept = 0
    aborted = None
    with out_path.open("a", encoding="utf-8") as w, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        bar = tqdm(total=len(todo), unit="msg")
        try:
            for row, labels in zip(todo, pool.map(lambda r: relabel(r["text"], args.model), todo)):
                bar.update(1)
                if not labels:
                    # The LLM answered but named no emoji -- keep the original
                    # rather than losing the example entirely.
                    labels = row["labels"]
                    kept += 1
                w.write(json.dumps({"text": row["text"], "labels": labels,
                                    "lang": row["lang"], "src": "relabel"},
                                   ensure_ascii=False) + "\n")
                written += 1
                if written % 500 == 0:
                    w.flush()
        except LLMUnavailable as exc:
            aborted = exc
        bar.close()

    print(f"relabelled {written:,} ({kept:,} kept original) -> {out_path}")
    if aborted is not None:
        print(f"\nABORTED: {aborted}\n"
              f"Everything written so far is good; rerun to resume from "
              f"{written:,}. Is LM Studio running with the model loaded?")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
