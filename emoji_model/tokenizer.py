"""SentencePiece unigram tokenizer, trained on our own mined text.

A shared Danish/English subword vocabulary keeps the embedding table -- the
single biggest chunk of the model -- small, while still covering æøå natively.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

PAD_ID, UNK_ID, BOS_ID = 0, 1, 2
CONTROL_TOKENS = ["<url>", "<user>", "<sub>"]


def build_word_list(corpus: str | Path, dictionary: str | Path,
                    min_count: int = 30, max_words: int = 2000) -> list[str]:
    """Words worth protecting from being split into fragments.

    A word qualifies if it is a real dictionary word, common in our own corpus,
    and currently tokenised into 3+ pieces. Forcing these to stay whole is what
    a dictionary can contribute: "water" should not become ▁ + wa + ter.
    """
    import collections
    import re

    known = {w.strip().lower() for w in open(dictionary, encoding="utf-8", errors="ignore")}
    freq: collections.Counter = collections.Counter()
    with open(corpus, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i > 400_000:
                break
            freq.update(re.findall(r"[a-zA-ZæøåÆØÅ']+", line.lower()))
    return [w for w, c in freq.most_common()
            if c >= min_count and w in known and len(w) > 2][:max_words]


def train_tokenizer(texts_file: str | Path, model_prefix: str | Path,
                    vocab_size: int = 8000,
                    protected_words: list[str] | None = None) -> str:
    spm.SentencePieceTrainer.train(
        input=str(texts_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9998,      # keeps æøå and common accents
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=-1,
        user_defined_symbols=CONTROL_TOKENS + list(protected_words or []),
        normalization_rule_name="nmt_nfkc_cf",   # lowercases too
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )
    return f"{model_prefix}.model"


class Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [BOS_ID] + self.sp.encode(text)
        ids = ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))
