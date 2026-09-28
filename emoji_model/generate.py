"""Write Danish training sentences for every emoji in the vocabulary.

124 of the 512 emoji have fewer than 50 Danish examples in the mined corpus, so
the model has almost nothing to learn them from. This asks an LLM -- LM Studio,
or Codex with --backend codex -- for sentences where each emoji would naturally
be used. Both append to the same raw file, so Codex can go first and the local
model fill whatever it leaves when its quota runs out.

One long list per emoji came back 19% duplicated and 63% built around the same
keyword ("cykel" for 🚲), which the keyword dictionary already covers. So each
emoji is asked from several angles, 20 sentences at a time -- one angle forbids
naming the thing at all -- and the filter caps keyword sentences at half.

Two stages, both resumable: raw answers go to --raw as they arrive, and the
filter rebuilds --out from them at the end, so it can be tuned without
generating again.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from . import relabel
from .emoji_utils import split_emoji
from .labels import fetch_annotations
from .relabel import LINE_RE, LLMUnavailable, auth_header, pick_host, probe

ANGLES = {
    "chat": "chatbeskeder til en ven eller i en gruppechat",
    "title": "overskrifter eller titler på opslag",
    "implicit": "beskeder der IKKE nævner tingen eller følelsen direkte -- det er "
                "situationen der gør at man vælger emojien",
    "everyday": "hverdagssituationer derhjemme, på arbejde, i skolen eller i fritiden",
    "reply": "korte svar eller reaktioner på noget en anden lige har skrevet",
    "plans": "spørgsmål eller planlægning med andre",
}
# Mood angles ("glad", "irriteret") were tried first and pulled against the
# emoji: 😠 got "Tillykke med din nye kæreste". The angle only varies the form.
PROMPT = (
    "Skriv {n} forskellige korte danske tekster, som man naturligt ville sætte "
    "{emoji} ({desc}) på. Denne gang: {angle}.\n"
    "Hver eneste tekst skal være en, hvor {emoji} er det oplagte emoji-valg -- "
    "passer vinklen dårligt, så gå på kompromis med vinklen, ikke med emojien.\n"
    "Varier længde (2-20 ord), emne og ordvalg. Skriv hverdagsdansk som rigtige "
    "mennesker skriver, gerne uformelt. Ingen gentagelser.\n"
    "Én tekst pr. linje, nummereret. Skriv IKKE selve emojien eller andre emoji."
)
WORD = re.compile(r"[a-zæøå]+")


def describe_da(emoji: str, da: dict, en: dict) -> str:
    entry = da.get(emoji) or da.get(emoji + "️") or en.get(emoji) or {}
    words = [entry.get("name") or ""] + (entry.get("keywords") or "").split("|")[:4]
    return ", ".join(dict.fromkeys(w.strip() for w in words if w.strip()))


def keywords_for(emoji: str, da: dict) -> set[str]:
    """CLDR words for the emoji; a sentence using one is 'about the keyword'."""
    entry = da.get(emoji) or da.get(emoji + "️") or {}
    text = " ".join([entry.get("name") or "", (entry.get("keywords") or "")])
    return {w for w in WORD.findall(text.lower()) if len(w) >= 4}


def ask(emoji: str, desc: str, angle: str, n: int, model: str,
        attempts: int = 3) -> list[str]:
    prompt = PROMPT.format(n=n, emoji=emoji, desc=desc, angle=ANGLES[angle])
    for attempt in range(attempts):
        try:
            host = pick_host()
            r = requests.post(f"{host}/v1/chat/completions", timeout=600,
                              headers=auth_header(host), json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9, "max_tokens": 120 * n + 400,
                "reasoning_effort": "low", "stream": False,
            })
            r.raise_for_status()
            out = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            if attempt == attempts - 1:
                raise LLMUnavailable(f"LLM call failed: {exc}") from exc
            time.sleep(15 * (attempt + 1))
            continue
        lines = []
        for line in out.splitlines():
            m = LINE_RE.match(line)
            if m:
                lines.append(m.group(2))
        return lines
    return []


CODEX_TASK = """\
Du skriver danske træningsdata til en emoji-model.

{infile} har én JSON-linje pr. emoji: {{"emoji": "🚲", "desc": "...", "angles": [...]}}.
For hver emoji og hver vinkel i dens "angles": skriv {n} forskellige korte danske
tekster, som man naturligt ville sætte emojien på.

Vinkler:
{angles}

Regler:
- Hver eneste tekst skal være en, hvor emojien er det oplagte emoji-valg. Passer
  vinklen dårligt, så gå på kompromis med vinklen, ikke med emojien.
- Varier længde (2-20 ord), emne og ordvalg. Hverdagsdansk som rigtige mennesker
  skriver, gerne uformelt. Ingen gentagelser, heller ikke på tværs af vinkler.
- Skriv IKKE emojien eller andre emoji i teksterne.
- Skriv teksterne selv. Brug ikke scripts, skabeloner eller kombinatorik til at
  generere dem -- det giver ensformige data, som er ubrugelige.

Skriv til {outfile} én JSON-linje pr. (emoji, vinkel):
{{"emoji": "🚲", "angle": "chat", "lines": ["...", "..."]}}
Ingen markdown, ingen forklaring på stdout ud over en kort bekræftelse.
"""


class CodexUnavailable(RuntimeError):
    """Codex refused the call -- out of quota, not signed in, rate limited."""


def ask_codex(group: list[tuple[str, str, list[str]]], n: int,
              model: str | None) -> list[dict]:
    """One `codex exec` for several emoji: the agent's own prompt costs the same
    per call however much it writes, so a call carries ~1,000 sentences."""
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        infile.write_text("\n".join(
            json.dumps({"emoji": e, "desc": d, "angles": a}, ensure_ascii=False)
            for e, d, a in group), encoding="utf-8")
        prompt = CODEX_TASK.format(
            infile=infile, outfile=outfile, n=n,
            angles="\n".join(f"- {k}: {v}" for k, v in ANGLES.items()))
        # The user's config runs Codex at xhigh; for writing sentences that
        # spends quota on reasoning nobody reads.
        cmd = ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check",
               "-c", 'model_reasoning_effort="low"', "-"]
        if model:
            cmd[2:2] = ["-m", model]
        try:
            proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                                  cwd=tmp, timeout=1800, check=False)
        except subprocess.TimeoutExpired:
            return []
        if proc.returncode != 0:
            # The last stderr line is Codex's token count, not the reason.
            lines = (proc.stderr or "").strip().splitlines()
            errors = [l for l in lines if l.startswith("ERROR")]
            raise CodexUnavailable((errors or lines or [f"exit {proc.returncode}"])[-1])
        if not outfile.exists():
            return []
        wanted = {(e, a) for e, _, angles in group for a in angles}
        recs = []
        for line in outfile.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                key = (d["emoji"], d["angle"])
                lines = [str(t) for t in d["lines"]]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if key in wanted:
                wanted.discard(key)
                lines = [(m.group(2) if (m := LINE_RE.match(t)) else t) for t in lines]
                recs.append({"emoji": key[0], "angle": key[1], "lines": lines})
        return recs


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def build(raw: Path, out: Path, per_emoji: int, exclude: Path | None,
          da: dict) -> None:
    """Rebuild the training file from every raw answer so far."""
    held_out = set()
    if exclude and exclude.exists():
        held_out = {norm(json.loads(l)["text"])
                    for l in exclude.open(encoding="utf-8") if l.strip()}
    by_emoji: dict[str, list[tuple[str, str]]] = {}
    owners: dict[str, set[str]] = {}
    for line in raw.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        for text in rec["lines"]:
            # The LLM appends the emoji despite being told not to; a model
            # trained on that would learn to copy it rather than to choose it.
            text = " ".join(split_emoji(text)[0].split()).strip(" \"'“”-–")
            key = norm(text)
            # gpt-oss occasionally leaks a control token such as <|endoftext|>.
            if "<|" in text or not 2 <= len(key.split()) <= 25 or key in held_out:
                continue
            owners.setdefault(key, set()).add(rec["emoji"])
            by_emoji.setdefault(rec["emoji"], []).append((key, text))

    rows, stats = [], {"kept": 0, "ambiguous": 0, "dupes": 0, "short": 0}
    for emoji, cands in by_emoji.items():
        kw = keywords_for(emoji, da)
        seen, plain, keyed = set(), [], []
        for key, text in cands:
            if key in seen:
                stats["dupes"] += 1
                continue
            seen.add(key)
            # Written for two different emoji: it teaches neither.
            if len(owners[key]) > 1:
                stats["ambiguous"] += 1
                continue
            has_kw = any(w.startswith(k) for w in key.split() for k in kw)
            (keyed if has_kw else plain).append(text)
        n_keyed = min(len(keyed), per_emoji // 2)
        keep = keyed[:n_keyed] + plain[:per_emoji - n_keyed]
        if len(keep) < per_emoji:
            stats["short"] += 1
        stats["kept"] += len(keep)
        rows += [{"text": t, "labels": [emoji], "lang": "da", "src": "synthetic"}
                 for t in keep]
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out}: {stats['kept']:,} rows for {len(by_emoji)} emoji "
          f"({stats['dupes']:,} duplicates, {stats['ambiguous']:,} shared between "
          f"emoji dropped; {stats['short']} emoji below {per_emoji})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=Path("data/emoji_vocab_v2.json"))
    ap.add_argument("--raw", type=Path, default=Path("data/synthetic_raw.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_da.jsonl"))
    ap.add_argument("--exclude", type=Path, default=Path("data/test_da_content.jsonl"),
                    help="held-out rows no synthetic sentence may copy")
    ap.add_argument("--per-emoji", type=int, default=100,
                    help="rows kept per emoji after filtering")
    ap.add_argument("--per-call", type=int, default=20)
    ap.add_argument("--emoji", nargs="*", help="only these emoji (for a trial run)")
    ap.add_argument("--host", action="append", default=None, metavar="URL")
    ap.add_argument("--backend", choices=["lmstudio", "codex"], default="lmstudio")
    ap.add_argument("--model", default=None,
                    help="default openai/gpt-oss-20b for lmstudio, Codex's own for codex")
    ap.add_argument("--codex-emoji", type=int, default=8,
                    help="emoji per codex exec call")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--build-only", action="store_true",
                    help="rebuild --out from --raw without calling the LLM")
    args = ap.parse_args()

    da, en = fetch_annotations("da"), fetch_annotations("en")
    emojis = json.loads(args.vocab.read_text(encoding="utf-8"))["emojis"]
    if args.emoji:
        emojis = [e for e in emojis if e in set(args.emoji)]

    if not args.build_only:
        done = set()
        if args.raw.exists():
            for line in args.raw.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["emoji"], rec["angle"]))
        # Six angles of 20 is 120 candidates for 100 kept rows; the filter
        # needs the slack for duplicates and the keyword cap.
        pending = {e: [a for a in ANGLES if (e, a) not in done] for e in emojis}
        pending = {e: a for e, a in pending.items() if a}
        print(f"{len(emojis)} emoji x {len(ANGLES)} angles: {len(done):,} calls done, "
              f"{sum(map(len, pending.values())):,} to go ({args.backend})")

        if args.backend == "codex":
            items = list(pending.items())
            tasks = [[(e, describe_da(e, da, en), a) for e, a in items[i:i + args.codex_emoji]]
                     for i in range(0, len(items), args.codex_emoji)]
            work = lambda g: ask_codex(g, args.per_call, args.model)
            unit, failure = "call", CodexUnavailable
        else:
            wanted = args.host or ["http://localhost:1234"]
            relabel.HOSTS = [h for h in (h if h.startswith("http") else f"http://{h}"
                                         for h in wanted) if probe(h)]
            if not relabel.HOSTS:
                raise SystemExit("no LM Studio instance reachable")
            model = args.model or "openai/gpt-oss-20b"
            tasks = [(e, a) for e, angles in pending.items() for a in angles]
            work = lambda t: [{"emoji": t[0], "angle": t[1], "lines": ask(
                t[0], describe_da(t[0], da, en), t[1], args.per_call, model)}]
            unit, failure = "call", LLMUnavailable

        aborted = None
        pool = ThreadPoolExecutor(max_workers=args.workers)
        with args.raw.open("a", encoding="utf-8") as w:
            try:
                for recs in tqdm(pool.map(work, tasks), total=len(tasks), unit=unit):
                    for rec in recs:
                        rec["backend"] = args.backend
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.flush()
            except failure as exc:
                aborted = exc
            finally:
                # map() queues every task up front; without cancelling, a dead
                # backend keeps the process alive until the whole queue has failed.
                pool.shutdown(wait=True, cancel_futures=True)
        if aborted is not None:
            print(f"\nABORTED: {aborted}\nRaw answers so far are kept; rerun to resume.")

    build(args.raw, args.out, args.per_emoji, args.exclude, da)


if __name__ == "__main__":
    main()
