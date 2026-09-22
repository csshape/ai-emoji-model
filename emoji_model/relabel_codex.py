"""Relabel mined messages with Codex instead of a local LLM.

Same job as relabel.py -- replace ironic human labels with ones that carry both
mood and content -- but batched, so one model call covers many messages.

Codex writes its answers to a file rather than stdout: parsing a CLI's console
output is where this kind of pipeline usually breaks.

Resumable: already-done texts are read back from the output file and skipped.
A batch that comes back unusable is skipped, never silently filled with the
original labels -- that is the failure mode that quietly poisons the corpus.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .emoji_utils import dedup_preserve_order, split_emoji

GENERIC = {"😂", "😅", "😊", "🤣", "😉", "😄", "😁", "🙂", "☺", "😃", "😆", "🙃"}

INSTRUCTIONS = """\
Du vælger emoji-svar til danske chatbeskeder.

For hver besked: giv 2-3 emoji, der tilsammen fanger BÅDE stemningen (glad,
træt, irriteret, vred, ked af det, stolt) OG det konkrete beskeden handler om
(mad, drikke, sted, transport, arbejde, ting). Sæt stemnings-emoji først.

Vigtigt: vælg efter hvad beskeden HANDLER om, ikke hvad en dansker ville have
skrevet. Danskere sætter 😂 som punktum efter alt muligt; det er netop det, vi
prøver at komme væk fra. Er beskeden vred, så skriv en vred emoji.

Brug KUN emoji fra denne liste. Alt uden for listen er ubrugeligt:
{vocab}

## Input
{infile} indeholder én JSON-linje pr. besked: {{"id": 1, "text": "..."}}

## Output
Skriv til {outfile} én JSON-linje pr. besked, samme id:
{{"id": 1, "emoji": "😠🚲"}}

Præcis {n} linjer, ét id hver, samme rækkefølge. Ingen markdown, ingen
forklaring, ingen tekst til stdout ud over en kort bekræftelse.
"""


class CodexUnavailable(RuntimeError):
    """Codex refused the call -- out of credits, not signed in, rate limited.

    Worth aborting on: every remaining batch will fail the same way, in about
    three seconds each, so grinding on just burns through the queue and reports
    them all as "skipped" with no reason attached.
    """


def run_batch(batch: list[dict], vocab: str, model: str) -> dict[int, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        infile = Path(tmp) / "in.jsonl"
        outfile = Path(tmp) / "out.jsonl"
        infile.write_text(
            "\n".join(json.dumps({"id": i, "text": r["text"]}, ensure_ascii=False)
                      for i, r in enumerate(batch)),
            encoding="utf-8")
        prompt = INSTRUCTIONS.format(vocab=vocab, infile=infile, outfile=outfile,
                                     n=len(batch))
        try:
            proc = subprocess.run(
                ["codex", "exec", "--sandbox", "workspace-write",
                 "--skip-git-repo-check", "-"],
                input=prompt, text=True, capture_output=True, timeout=900, check=False)
        except subprocess.TimeoutExpired:
            return {}
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            raise CodexUnavailable(reason)
        if not outfile.exists():
            return {}
        out: dict[int, list[str]] = {}
        for line in outfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
                emoji = dedup_preserve_order(split_emoji(d["emoji"])[1])[:3]
                if emoji:
                    out[int(d["id"])] = emoji
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, default=Path("data/mined.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/relabeled.jsonl"))
    ap.add_argument("--vocab", type=Path, default=Path("data/emoji_vocab_v2.json"))
    ap.add_argument("--lang", default="da")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many examples (default: all pending)")
    ap.add_argument("--model", default="codex")
    args = ap.parse_args()

    vocab = " ".join(json.loads(args.vocab.read_text(encoding="utf-8"))["emojis"])

    done: set[str] = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["text"])

    todo = []
    with args.mined.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["lang"] != args.lang or r["text"] in done:
                continue
            # The generic-only rows are where the ironic labels live, so they
            # are worth the model call first.
            if all(e in GENERIC for e in r["labels"]):
                todo.append(r)
    if args.limit:
        todo = todo[:args.limit]

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(done):,} already relabelled · {len(todo):,} pending "
          f"in {len(batches)} batches of {args.batch}")

    written = failed = 0
    aborted = None
    bar = tqdm(total=len(todo), unit="msg")
    with args.out.open("a", encoding="utf-8") as w, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            for batch, result in zip(batches, pool.map(
                    lambda b: run_batch(b, vocab, args.model), batches)):
                if not result:
                    failed += len(batch)
                    bar.update(len(batch))
                    continue
                for i, row in enumerate(batch):
                    labels = result.get(i)
                    if not labels:
                        failed += 1
                        continue
                    w.write(json.dumps({"text": row["text"], "labels": labels,
                                        "lang": row["lang"], "src": "codex"},
                                       ensure_ascii=False) + "\n")
                    written += 1
                w.flush()
                bar.update(len(batch))
        except CodexUnavailable as exc:
            aborted = exc
    bar.close()
    print(f"\nrelabelled {written:,} · skipped {failed:,} -> {args.out}")
    if aborted is not None:
        print(f"\nABORTED: {aborted}\n"
              f"Everything written so far is good; rerun to resume from "
              f"{written:,}.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
