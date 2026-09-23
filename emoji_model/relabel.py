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
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from .emoji_utils import dedup_preserve_order, split_emoji

GENERIC = {"😂", "😅", "😊", "🤣", "😉", "😄", "😁", "🙂", "☺", "😃", "😆", "🙃"}

VOCAB_RULE = (
    "\n\nBrug KUN emoji fra denne liste — alt udenfor er ubrugeligt:\n{vocab}"
)

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


# Round-robined across whatever LM Studio instances are reachable, so a second
# machine on the network doubles throughput. Filled in by main().
HOSTS: list[str] = ["http://localhost:1234"]
_next_host = itertools.count()


def pick_host() -> str:
    return HOSTS[next(_next_host) % len(HOSTS)]


def probe(host: str, timeout: int = 8) -> bool:
    """Is this instance up and serving the model?"""
    try:
        r = requests.get(f"{host}/v1/models", timeout=timeout)
        return r.status_code == 200 and "gpt-oss" in r.text
    except Exception:
        return False


def relabel(text: str, model: str, system: str = SYSTEM,
            attempts: int = 4) -> list[str]:
    """Ask the LLM for labels. Retries transient failures, aborts on real ones.

    LM Studio answers /v1/models before the weights are in memory and returns
    500 until they are, so a first call can fail for a minute or two while a
    20B model loads. Giving up there would be as wrong as never giving up.
    """
    for attempt in range(attempts):
        try:
            return _one_call(text, model, system, pick_host())
        except LLMUnavailable:
            if attempt == attempts - 1:
                raise
            time.sleep(15 * (attempt + 1))
    return []


def _one_call(text: str, model: str, system: str, host: str) -> list[str]:
    try:
        r = requests.post(f"{host}/v1/chat/completions", timeout=180, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
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
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="take only every Nth example, offset I -- so two "
                         "machines can work the same file without overlap "
                         "(0/2 here, 1/2 there) and the outputs just concatenate")
    ap.add_argument("--host", action="append", default=None, metavar="URL",
                    help="LM Studio instance, repeatable; unreachable ones are "
                         "dropped at startup (default http://localhost:1234)")
    ap.add_argument("--vocab", type=Path, default=Path("data/emoji_vocab_v2.json"),
                    help="restrict answers to this emoji vocabulary")
    ap.add_argument("--mined", default="data/mined.jsonl")
    ap.add_argument("--out", default="data/relabeled.jsonl")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--lang", default="da", help="'da', 'en' or 'all'")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    global HOSTS
    wanted = args.host or ["http://localhost:1234"]
    wanted = [h if h.startswith("http") else f"http://{h}" for h in wanted]
    HOSTS = [h for h in wanted if probe(h)]
    for h in wanted:
        print(f"  {h}  {'ok' if h in HOSTS else 'unreachable — skipped'}")
    if not HOSTS:
        raise SystemExit("no LM Studio instance reachable")
    print(f"using {len(HOSTS)} instance(s)\n")

    system = SYSTEM
    if args.vocab and args.vocab.exists():
        emojis = json.loads(args.vocab.read_text(encoding="utf-8"))["emojis"]
        system += VOCAB_RULE.format(vocab=" ".join(emojis))
        print(f"restricting answers to {len(emojis)} emoji from {args.vocab}")

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
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        before = len(todo)
        todo = todo[i::n]
        print(f"shard {i}/{n}: {len(todo):,} of {before:,} examples")
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo):,} examples to relabel (lang={args.lang})")

    written = kept = 0
    aborted = None
    with out_path.open("a", encoding="utf-8") as w, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        bar = tqdm(total=len(todo), unit="msg")
        try:
            for row, labels in zip(todo, pool.map(lambda r: relabel(r["text"], args.model, system), todo)):
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
