import time
from collections import Counter

from cs336_basics.pretokenization import pretokenize


def train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:  # (vocab, merges)
    merges: list[tuple[bytes, bytes]] = []
    indices: dict[bytes, int] = {special_tokens[x].encode('utf-8'): x for x in range(len(special_tokens))}
    vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
    special_tokens_count: int = len(special_tokens)
    for x in range(special_tokens_count, 256 + special_tokens_count):
        vocab[x] = bytes([x - special_tokens_count])
        indices[bytes([x - special_tokens_count])] = x
    initial_vocab_size = len(vocab)
    num_merges: int = vocab_size - initial_vocab_size
    split_special_token: bytes = special_tokens[0].encode('utf-8')
    pretoken_vocab: Counter[tuple[bytes, ...]] = pretokenize(input_path, split_special_token=split_special_token)
    t0 = time.perf_counter()
    for i in range(num_merges):
        counts: Counter[tuple[bytes, bytes]] = _count_adjacent_pairs(pretoken_vocab)
        pair: tuple[bytes, bytes] = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(pair)
        new_index: int = special_tokens_count + 256 + i
        merged_pair: bytes = vocab[indices[pair[0]]] + vocab[indices[pair[1]]]
        vocab[new_index] = merged_pair
        indices[merged_pair] = new_index
        pretoken_vocab = _merge(pretoken_vocab, pair)
    t1 = time.perf_counter()
    print(f"training: {t1 - t0:.2f}s")
    return vocab, merges


def _merge(pretoken_vocab: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes, ...]]:
    result_pretoken_vocab: Counter[tuple[bytes, ...]] = Counter({})
    merges_count = 0
    for pretoken, count in pretoken_vocab.items():
        new_pretoken_list: list[bytes] = []
        i: int = 0
        pretoken_len: int = len(pretoken)
        while i < pretoken_len:
            if i < pretoken_len - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                new_pretoken_list.append(pretoken[i] + pretoken[i + 1])
                i += 2
                merges_count += 1
            else:
                new_pretoken_list.append(pretoken[i])
                i += 1
        result_pretoken_vocab[tuple(new_pretoken_list)] += count
    print(f"merges count: {merges_count}")
    print(f"pretoken vocab size: {len(result_pretoken_vocab)}")
    return result_pretoken_vocab


def _count_adjacent_pairs(pretoken_vocab: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter({})
    for pretoken, count in pretoken_vocab.items():
        for byte_pair in zip(pretoken[:-1], pretoken[1:]):
            counts[byte_pair] += count
    return counts


def _count_byte_pairs(pretoken: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter({})
    for byte_pair in zip(pretoken[:-1], pretoken[1:]):
        counts[byte_pair] += 1
    return counts


def _merge_pretoken(pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]) -> tuple[tuple[bytes, ...], Counter[tuple[bytes, bytes]]]:
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
    train("../data/TinyStoriesV2-GPT4-valid.txt", 500, ["<|endoftext|>"])
    # _merge(Counter({(b'l', b'o', b'w', b'e', b'r'): 1}), (b'l', b'o'))
