from collections import Counter, defaultdict


class BpeState:
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

    def merge(self, pair: tuple[bytes, bytes]) -> None:
        # iterate over pretokens that contain this pair only
        for pretoken in self.pairs_to_pretokens_index[pair].copy():
            count: int = self.pretoken_vocab[pretoken]
            new_pretoken, count_deltas = self._merge_pretoken(pair, pretoken)
            for merged_pair, delta in count_deltas.items():
                self.counts[merged_pair] += count * delta
            for old_pair in self._count_byte_pairs(pretoken).keys():
                if pretoken in self.pairs_to_pretokens_index[old_pair]:
                    self.pairs_to_pretokens_index[old_pair].remove(pretoken)
            for new_pair in self._count_byte_pairs(new_pretoken).keys():
                self.pairs_to_pretokens_index[new_pair].add(new_pretoken)
            del self.pretoken_vocab[pretoken]
            self.pretoken_vocab[new_pretoken] += count

    @classmethod
    def _count_byte_pairs(cls, pretoken: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
        counts: Counter[tuple[bytes, bytes]] = Counter({})
        for byte_pair in zip(pretoken[:-1], pretoken[1:]):
            counts[byte_pair] += 1
        return counts

    @classmethod
    def _merge_pretoken(cls, pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]) -> tuple[
        tuple[bytes, ...], Counter[tuple[bytes, bytes]]]:
        new_pretoken: tuple[bytes, ...] = ()
        i: int = 0
        pretoken_len: int = len(pretoken)
        while i < pretoken_len:
            if i < pretoken_len - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                new_token = pretoken[i] + pretoken[i + 1]
                new_pretoken += (new_token,)
                i += 2
            else:
                new_pretoken += (pretoken[i],)
                i += 1
        old_counts: Counter[tuple[bytes, bytes]] = cls._count_byte_pairs(pretoken)
        new_counts: Counter[tuple[bytes, bytes]] = cls._count_byte_pairs(new_pretoken)
        new_counts.subtract(old_counts)
        count_deltas: Counter[tuple[bytes, bytes]] = Counter(
            {pair: count for pair, count in new_counts.items() if count != 0})
        return new_pretoken, count_deltas
