from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class DeterministicSampler:
    def __init__(self, values: Sequence[object], *, seed: int, epoch: int = 0) -> None:
        self.values = values
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[object]:
        indices = list(range(len(self.values)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        return (self.values[index] for index in indices)

    def __len__(self) -> int:
        return len(self.values)
