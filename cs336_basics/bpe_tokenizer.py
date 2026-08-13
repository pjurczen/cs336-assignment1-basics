import time
from collections import Counter

from cs336_basics.bpe_state import BpeState
from cs336_basics.pretokenization import pretokenize

BYTES_COUNT = 256  # this is really an overkill since in UTF-8 192, 193 and >=245 bytes are not used, but we learn those if at inference time invalid input was somehow passed

def train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:  # (vocab, merges)
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
    special_tokens_count: int = len(special_tokens)
    for x in range(special_tokens_count, BYTES_COUNT + special_tokens_count):
        vocab[x] = bytes([x - special_tokens_count])
    initial_vocab_size = len(vocab)
    num_merges: int = vocab_size - initial_vocab_size
    split_special_token: bytes = special_tokens[0].encode('utf-8')
    pretoken_vocab: Counter[tuple[bytes, ...]] = pretokenize(input_path, split_special_token=split_special_token)

    t0 = time.perf_counter()
    bpe_state: BpeState = BpeState(pretoken_vocab)
    for i in range(num_merges):
        pair: tuple[bytes, bytes] = bpe_state.get_highest_count_pair()
        merges.append(pair)
        new_index: int = special_tokens_count + BYTES_COUNT + i
        merged_pair: bytes = pair[0] + pair[1]
        vocab[new_index] = merged_pair
        bpe_state.merge(pair)
    t1 = time.perf_counter()
    print(f"training: {t1 - t0:.2f}s")
    return vocab, merges



if __name__ == "__main__":
    train("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # _merge(Counter({(b'l', b'o', b'w', b'e', b'r'): 1}), (b'l', b'o'))
