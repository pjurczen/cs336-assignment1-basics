from collections import Counter, defaultdict


class Pretokens:
    pretoken_vocab: Counter[tuple[bytes, ...]]
    counts: Counter[tuple[bytes, bytes]]
    pairs_to_pretokens_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]

    def __init__(self, pretoken_vocab: Counter[tuple[bytes, ...]]):
        self.pretoken_vocab = pretoken_vocab
        self.counts = self._count_adjacent_pairs(pretoken_vocab)
        self.pairs_to_pretokens_index = self._calculate_pairs_to_pretokens_index(pretoken_vocab)

    def _count_adjacent_pairs(self, pretoken_vocab: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
        counts: Counter[tuple[bytes, bytes]] = Counter({})
        for pretoken, count in pretoken_vocab.items():
            for byte_pair in zip(pretoken[:-1], pretoken[1:]):
                counts[byte_pair] += count
        return counts

    def _calculate_pairs_to_pretokens_index(self, pretoken_vocab: dict[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
        pairs_to_pretokens_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for pretoken in pretoken_vocab.keys():
            i: int = 0
            pretoken_len: int = len(pretoken)
            while i < pretoken_len - 1:
                pairs_to_pretokens_index[(pretoken[i], pretoken[i + 1])].add(pretoken)
                i += 1
        return pairs_to_pretokens_index

    def get_highest_count_pair(self) -> tuple[bytes, bytes]:
        return max(self.counts.items(), key=lambda x: (x[1], x[0]))[0]
