from collections import Counter
from typing import Optional


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
        if pair not in self.counts:
            self.counts[pair] = 0
        old_count: int = self.counts[pair]
        new_count: int = old_count + delta
        if new_count == 0:
            del self.counts[pair]
        else:
            self.counts[pair] += delta
            if new_count not in self.buckets:
                self.buckets[new_count] = set()
            self.buckets[new_count].add(pair)
        if old_count != 0:
            self.buckets[old_count].remove(pair)
            if not self.buckets[old_count]:
                del self.buckets[old_count]
            if old_count == self.max_count:
                i: int = self.max_count
                while not i in self.buckets and i > 0:
                    i -= 1
                self.max_count = i
        if new_count > self.max_count:
            self.max_count = new_count

    def highest(self) -> Optional[tuple[bytes, bytes]]:
        if not self.buckets:
            return None
        highest_pairs: set[tuple[bytes, bytes]] = self.buckets[self.max_count]
        return max(highest_pairs)
