from collections import Counter


class PretokenCounter:
    counts: dict[tuple[bytes, bytes], int]
    buckets: dict[int, set[tuple[bytes, bytes]]]
    max_count: int

    def __init__(self, counter: Counter[tuple[bytes, bytes]]):
        self.counts = dict(counter.items())
        self.buckets = {}
        for pair, count in self.counts.items():
            if count not in self.buckets:
                self.buckets[count] = set()
            self.buckets[count].add(pair)
        self.max_count = max(self.counts.values()) if self.counts.values() else 0

    @classmethod
    def from_counter(cls, counter: Counter[tuple[bytes, bytes]]) -> "PretokenCounter":
        return PretokenCounter(counter)

    def add(self, pair: tuple[bytes, bytes], delta: int) -> None:
        pass

    def highest(self) -> tuple[bytes, bytes]:
        pass
