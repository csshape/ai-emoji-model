# ai-emoji-model

A 1.6M-parameter transformer that reads a Danish or English chat message and
answers with 1–5 emoji. Small enough to run on a low-end phone — small enough,
it turns out, to run in a browser tab with no ML runtime at all.

**[Try it](https://csshape.github.io/ai-emoji-model/report/test.html)** — both model
versions run locally in the page, ~6.5 MB of weights each.

## What it is

Multi-label classification over a fixed 512-emoji vocabulary, not generation.
Training labels come from people: mine messages where the author used emoji,
strip the emoji out, and use them as the target.

```
                     d_model 128 · 4 heads · 4 layers · d_ff 256 · max_len 64
text ──► SentencePiece (8k, shared da/en) ──► embed + pos
     ──► 4 × pre-norm encoder block ──► masked mean pool
     ──► linear ──► 512 sigmoids ──► prior correction ──► top 1–5
```

The encoder is written with plain ops rather than `nn.TransformerEncoder`, so
the graph exports cleanly to ONNX and Core ML.

## What we learned

The interesting part of this project is not the architecture. It is that the
labels people write are not the labels you would expect.

**Emoji is punctuation, not description.** Across 50,000 Danish chat messages,
89% of emoji sit at the very end and 76% of emoji-bearing messages contain no
`.` `!` or `?` at all. The emoji *is* the full stop.

**So the model learns to punctuate.** Ask the first version `jeg er sur`
("I am angry") and it answers 😂 — because among 1,895 Danish messages
containing "sur", 😂 is the single most common label. It is not being funny;
it is ending the sentence.

**The same concept is labelled differently per language.** Danish messages
about cycling (328 of them) are labelled 😂×52 😅×40 😊×17 — not one 🚲.
English messages about cycling (237) are labelled 🚲×54 as the top choice.
Danes punctuate; English speakers describe.

**Three failure modes that look identical from outside:**

| Symptom | Real cause | Fix |
|---|---|---|
| `Jeg cykler en tur` → 😳 | 🚲 not in the vocabulary at all | rebuild the label space |
| Danish object words → 😂 | texts exist, labels are generic | relabel with an LLM |
| `beer` → 👇 | 700 texts, 🍻 is the top label, model ranks it 127th of 512 | a training problem, still open |

**Measure against a constant.** A "model" that always answers 😂😅✨😊❤
scores recall@5 0.222 on the diagnostic set. The real model scores 0.274. On
two categories it scores *below* the constant.

**Synthetic training data teaches you to agree with the generator.** Adding
689 LLM-written training examples moved the LLM-built test set from 0.274 to
0.421 (+54%) and moved real human-labelled text by −0.007, with every
confidence interval crossing zero. Measuring on one test set would have
reported a breakthrough.

## Layout

```
emoji_model/         training, evaluation, export
  build_data.py      mine messages that carry emoji
  build_vocab.py     choose the 512 emoji, blending human and LLM votes
  prepare.py         tokenizer + tensors  (--boost adds train-only examples)
  train.py           BCE, AdamW, cosine schedule  (--class-balance for rare labels)
  infer.py           CLI: reply to text with emoji
  eval_short.py      held-out short messages, banded by length
  eval_phase.py      diagnostic set, broken down by category
  relabel.py         rewrite labels with a local LLM via LM Studio
  export_web.py      flatten a checkpoint into files a browser can run
report/
  test.html          side-by-side test bench, runs in the browser
  model/emoji.js     the forward pass in plain JavaScript
  model/check.mjs    verifies the JS against PyTorch
prompts/             prompts and a validator for generating test data
data/                vocabularies and the synthetic sets only (see below)
```

## Running it

```bash
uv sync
uv run python -m emoji_model.build_data          # mine a corpus (see note)
uv run python -m emoji_model.prepare --mined data/mined.jsonl --outdir data
uv run python -m emoji_model.train --data data --out checkpoints
uv run python -m emoji_model.infer "jeg er sur"
```

The browser bench needs an http server, not `file://`:

```bash
uv run python -m emoji_model.export_web --model base:checkpoints/best.pt:data
cd report && python3 -m http.server 8000     # → http://127.0.0.1:8000/test.html
```

`report/model/check.mjs` compares the JavaScript implementation against PyTorch;
it currently matches on tokens, argmax and output emoji, within 2e-4 on
probabilities.

## About the data

**The mined corpus is not in this repository, on purpose.** It is scraped from
Twitter and Reddit: X's terms permit redistributing tweet IDs but not tweet
text, Reddit content belongs to the people who wrote it, and these are real
people's messages. `build_data.py` rebuilds it.

What *is* here is the material we generated ourselves — the LLM-written
diagnostic and training sets under `data/testdata_codex/` and
`data/traindata_codex/` — plus the emoji vocabularies and the exported weights.

## Status

Honest summary: the model does sentiment and fails at objects, places and
work. It beats a constant guess by 0.052 overall, and loses to it on two of
seven categories. The vocabulary rebuild and the LLM relabelling are the two
open threads.

MIT licensed.
