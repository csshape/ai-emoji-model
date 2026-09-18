"""Label Danish messages with emoji using a local LLM (LM Studio or Ollama).

Measured against the user's chat feeds, llama3.1:8b agreed with the gold labels
22.5% of the time where our trained model managed 12.9% -- and its mistakes are
often better than the reference ("en god rulle" -> 🍕 where the human wrote 😀).
The point is not raw agreement but the kind of emoji: LLMs pick content, humans
on Reddit default to 😊.

Resumable: rerunning appends only messages not already annotated.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from .emoji_utils import dedup_preserve_order, split_emoji

SYSTEM = ("Du er en emoji-vælger for en dansk chat-app. "
          "Du svarer UDELUKKENDE med 1-3 emoji-tegn der passer som reaktion på beskeden. "
          "Vælg emoji der afspejler INDHOLDET (mad, sted, aktivitet, følelse), "
          "ikke bare en generisk smiley. Aldrig ord, aldrig forklaring, kun emoji.")

BACKENDS = {
    # LM Studio speaks the OpenAI chat API on 1234.
    "lmstudio": ("http://localhost:1234/v1/chat/completions", "openai"),
    "ollama": ("http://localhost:11434/api/generate", "ollama"),
}


def annotate(text: str, url: str, style: str, model: str, timeout: int = 120) -> list[str]:
    try:
        if style == "openai":
            r = requests.post(url, timeout=timeout, json={
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"Besked: {text}"}],
                # Reasoning models (gpt-oss) spend tokens thinking before they
                # answer: with max_tokens=16 the whole budget went to reasoning
                # and content came back empty. Give room, and ask it to think
                # briefly -- picking an emoji does not need deliberation.
                "temperature": 0.3, "max_tokens": 200, "stream": False,
                "reasoning_effort": "low",
            })
            msg = r.json()["choices"][0]["message"]
            out = msg.get("content") or msg.get("reasoning") or ""
        else:
            r = requests.post(url, timeout=timeout, json={
                "model": model, "system": SYSTEM,
                "prompt": f"Besked: {text}\nEmoji-svar:",
                "stream": False,
                "options": {"num_predict": 16, "temperature": 0.3},
            })
            out = r.json().get("response", "")
    except Exception:
        return []
    # Models sometimes emit reasoning before the answer; the emoji are what count.
    return dedup_preserve_order(split_emoji(out)[1])[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/unlabeled_da.jsonl")
    ap.add_argument("--out", default="data/llm_labeled_da.jsonl")
    ap.add_argument("--backend", choices=list(BACKENDS), default="lmstudio")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    url, style = BACKENDS[args.backend]
    rows = [json.loads(l) for l in Path(args.input).open(encoding="utf-8") if l.strip()]

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["text"])
        print(f"resuming: {len(done):,} already annotated")

    todo = [r for r in rows if r["text"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo):,} messages to annotate via {args.backend} ({args.model})")

    written = empty = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as w, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        bar = tqdm(total=len(todo), unit="msg")
        for row, labels in zip(todo, pool.map(
                lambda r: annotate(r["text"], url, style, args.model), todo)):
            bar.update(1)
            if not labels:
                empty += 1
                continue
            w.write(json.dumps({"text": row["text"], "labels": labels,
                                "lang": row.get("lang", "da"), "src": "llm"},
                               ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                w.flush()
        bar.close()

    dt = time.time() - t0
    rate = len(todo) / dt if dt else 0
    print(f"wrote {written:,} labelled ({empty:,} empty) in {dt/60:.1f} min "
          f"= {rate:.1f} msg/s")


if __name__ == "__main__":
    main()
