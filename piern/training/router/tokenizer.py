from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(slots=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def from_counter(cls, counter: Counter[str]) -> "CharTokenizer":
        itos = [PAD_TOKEN, UNK_TOKEN]
        for token, _ in counter.most_common():
            if token in {PAD_TOKEN, UNK_TOKEN}:
                continue
            itos.append(token)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def load(cls, path: Path | str) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]
