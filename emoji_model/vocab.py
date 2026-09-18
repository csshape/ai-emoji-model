"""The label space: which emoji the model is allowed to answer with."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import emoji as emoji_lib


class EmojiVocab:
    def __init__(self, emojis: list[str]) -> None:
        self.emojis = emojis
        self.index = {e: i for i, e in enumerate(emojis)}

    def __len__(self) -> int:
        return len(self.emojis)

    def encode(self, labels: list[str]) -> list[int]:
        return [self.index[e] for e in labels if e in self.index]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.emojis[i] for i in ids]

    def name(self, e: str) -> str:
        return emoji_lib.demojize(e).strip(":").replace("_", " ")

    @classmethod
    def build(cls, label_lists, size: int = 512, min_count: int = 20) -> "EmojiVocab":
        counts = Counter(e for labels in label_lists for e in labels)
        kept = [e for e, n in counts.most_common(size) if n >= min_count]
        return cls(kept)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"emojis": self.emojis}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EmojiVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["emojis"])
