import time
from collections import Counter

from cs336_basics.pretokenization import pretokenize
from cs336_basics.pretokens import Pretokens

BYTES_COUNT = 256  # this is really an overkill since in UTF-8 192, 193 and >=245 bytes are not used, but we learn those if at inference time invalid input was somehow passed

def train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:  # (vocab, merges)
    merges: list[tuple[bytes, bytes]] = []
    indices: dict[bytes, int] = {special_tokens[x].encode('utf-8'): x for x in range(len(special_tokens))}
    vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
    special_tokens_count: int = len(special_tokens)
    for x in range(special_tokens_count, BYTES_COUNT + special_tokens_count):
        vocab[x] = bytes([x - special_tokens_count])
        indices[bytes([x - special_tokens_count])] = x
    initial_vocab_size = len(vocab)
    num_merges: int = vocab_size - initial_vocab_size
    split_special_token: bytes = special_tokens[0].encode('utf-8')
    pretoken_vocab: Counter[tuple[bytes, ...]] = pretokenize(input_path, split_special_token=split_special_token)

    t0 = time.perf_counter()
    pretokens: Pretokens = Pretokens(pretoken_vocab)
    for i in range(num_merges):
        pair: tuple[bytes, bytes] = pretokens.get_highest_count_pair()
        merges.append(pair)
        new_index: int = special_tokens_count + BYTES_COUNT + i
        merged_pair: bytes = pair[0] + pair[1]
        vocab[new_index] = merged_pair
        indices[merged_pair] = new_index
        _merge(pair, pretokens)
    t1 = time.perf_counter()
    print(f"training: {t1 - t0:.2f}s")
    return vocab, merges


def _merge(pair: tuple[bytes, bytes], pretokens: Pretokens) -> None:
    # iterate over pretokens that contain this pair only
    for pretoken in pretokens.pairs_to_pretokens_index[pair].copy():
        count: int = pretokens.pretoken_vocab[pretoken]
        new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
        for merged_pair, delta in count_deltas.items():
            pretokens.counts[merged_pair] += count * delta
        for old_pair in _count_byte_pairs(pretoken).keys():
            if pretoken in pretokens.pairs_to_pretokens_index[old_pair]:
                pretokens.pairs_to_pretokens_index[old_pair].remove(pretoken)
        for new_pair in _count_byte_pairs(new_pretoken).keys():
            pretokens.pairs_to_pretokens_index[new_pair].add(new_pretoken)
        del pretokens.pretoken_vocab[pretoken]
        pretokens.pretoken_vocab[new_pretoken] += count


def _count_byte_pairs(pretoken: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter({})
    for byte_pair in zip(pretoken[:-1], pretoken[1:]):
        counts[byte_pair] += 1
    return counts


def _merge_pretoken(pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]) -> tuple[
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
    old_counts: Counter[tuple[bytes, bytes]] = _count_byte_pairs(pretoken)
    new_counts: Counter[tuple[bytes, bytes]] = _count_byte_pairs(new_pretoken)
    new_counts.subtract(old_counts)
    count_deltas: Counter[tuple[bytes, bytes]] = Counter(
        {pair: count for pair, count in new_counts.items() if count != 0})
    return new_pretoken, count_deltas


if __name__ == "__main__":
    train("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # _merge(Counter({(b'l', b'o', b'w', b'e', b'r'): 1}), (b'l', b'o'))
