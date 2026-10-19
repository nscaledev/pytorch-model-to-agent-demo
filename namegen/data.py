"""Dataset utilities: load names, build a character vocabulary, make training batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"

# Token 0 is a special boundary token used both as "start of name" and "end of name".
BOUNDARY = "."


@dataclass
class Vocab:
    chars: list[str]

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.chars.index(c) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


class NamesDataset:
    """All names packed into a single (N, block_size + 1) tensor of token ids.

    Each row is: BOUNDARY, c1, c2, ..., ck, BOUNDARY, BOUNDARY, ... (padded).
    Padding positions are masked out of the loss with target -1.
    """

    def __init__(self, path: Path = DATA_PATH, split_seed: int = 42):
        words = [w.strip().lower() for w in path.read_text().splitlines()]
        words = [w for w in words if w]
        chars = sorted(set("".join(words)))
        self.vocab = Vocab([BOUNDARY] + chars)
        self.block_size = max(len(w) for w in words) + 1  # +1 for the leading boundary

        g = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(words), generator=g).tolist()
        n_val = int(0.1 * len(words))
        self.train_words = [words[i] for i in perm[n_val:]]
        self.val_words = [words[i] for i in perm[:n_val]]

        self.train = self._pack(self.train_words)
        self.val = self._pack(self.val_words)

    def _pack(self, words: list[str]) -> torch.Tensor:
        out = torch.zeros(len(words), self.block_size + 1, dtype=torch.long)
        for i, w in enumerate(words):
            ids = [0] + self.vocab.encode(w) + [0]
            out[i, : len(ids)] = torch.tensor(ids)
        return out

    def get_batch(self, split: str, batch_size: int, generator: torch.Generator | None = None):
        """Return (x, y) where y is x shifted left by one. Padding targets are -1."""
        rows = self.train if split == "train" else self.val
        idx = torch.randint(0, rows.size(0), (batch_size,), generator=generator)
        seq = rows[idx]
        x = seq[:, :-1].clone()
        y = seq[:, 1:].clone()
        # Only the first boundary after the name is a real target; later pads are ignored.
        # Find, per row, the position of the first end-boundary in y and mask after it.
        is_end = (y == 0).int()
        first_end = is_end.cumsum(dim=1)
        y[first_end > 1] = -1
        return x, y
