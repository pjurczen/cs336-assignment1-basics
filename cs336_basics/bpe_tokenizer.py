import cProfile
import json
import pstats
import resource
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import lru_cache

from cs336_basics.bpe_state import BpeState
from cs336_basics.peek_memory_sampler import PeekMemorySampler
from cs336_basics.pretokenization import pretokenize, split_text_to_chunks, pretokenize_text, _build_tuple_bytes, \
    merge_pretoken, get_byte_pairs
from tests.common import gpt2_bytes_to_unicode

BYTES_COUNT = 256  # this is really an overkill since in UTF-8 192, 193 and >=245 bytes are not used, but we learn those if at inference time invalid input was somehow passed


class BpeTokenizer:
    vocab: dict[int, bytes]
    id_vocab: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    merges_dict: dict[bytes, bytes]
    merges_rank: dict[tuple[bytes, bytes], int]
    special_tokens: list[bytes]

    def __init__(self):
        self.vocab = {}
        self.id_vocab = {}
        self.merges = []
        self.special_tokens = []
        self.merges_dict = {}

    @classmethod
    def from_resources(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> "BpeTokenizer":
        bpe_tokenizer: BpeTokenizer = BpeTokenizer()
        bpe_tokenizer.merges = merges
        bpe_tokenizer.vocab = vocab
        bpe_tokenizer.special_tokens = [x.encode('utf-8') for x in special_tokens] if special_tokens else []
        bpe_tokenizer.id_vocab = {id: byte for byte, id in vocab.items()}
        bpe_tokenizer.merges_dict = {x[0]: x[1] for x in merges}
        bpe_tokenizer.merges_rank = {pair: i for i, pair in enumerate(merges)}
        return bpe_tokenizer

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:  # (vocab, merges)
        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
        self.special_tokens = [x.encode('utf-8') for x in special_tokens]
        special_tokens_count: int = len(special_tokens)
        for x in range(special_tokens_count, BYTES_COUNT + special_tokens_count):
            vocab[x] = bytes([x - special_tokens_count])
        initial_vocab_size = len(vocab)
        num_merges: int = vocab_size - initial_vocab_size
        split_special_token: bytes = self.special_tokens[0]
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
        self.vocab = vocab
        self.id_vocab = {id: byte for byte, id in vocab.items()}
        self.merges = merges
        self.merges_dict = {x[0]: x[1] for x in merges}
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []
        encoded_text: bytes = text.encode('utf-8')
        chunks: list[bytes] = split_text_to_chunks(encoded_text, self.special_tokens)
        for chunk in chunks:
            if chunk in self.special_tokens:
                encoded_ids.append(self.id_vocab[chunk])
            else:
                pretokens: list[bytes] = pretokenize_text(chunk)
                for pretoken in pretokens:
                    encoded_ids += self._encode_pretoken(pretoken)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    @lru_cache(maxsize=None)
    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        tuple_bytes: tuple[bytes, ...] = _build_tuple_bytes(pretoken)
        while True:
            byte_pairs: set[tuple[bytes, bytes]] = get_byte_pairs(tuple_bytes)
            candidates: list[tuple[bytes, bytes]] = [x for x in byte_pairs if x in self.merges]
            if not candidates:
                break
            pair: tuple[bytes, bytes] = min(candidates, key=self.merges_rank.get)
            tuple_bytes = merge_pretoken(pair, tuple_bytes)
        return [self.id_vocab[x] for x in tuple_bytes]

    def decode(self, ids: list[int]) -> str:
        bytes_list: list[bytes] = list(map(self.vocab.get, ids))
        text = b"".join(bytes_list).decode('utf-8', errors='replace')
        return text

    def save(self, vocab_path: str, merges_path: str) -> None:
        encoded_vocab: dict[str, int] = {
            bytes_to_gpt2_unicode(bytes_item): vocab_index
            for bytes_item, vocab_index in self.id_vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(encoded_vocab, f, ensure_ascii=False, indent=4)
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{bytes_to_gpt2_unicode(merge[0])} {bytes_to_gpt2_unicode(merge[1])}\n")

    def max_length_token(self) -> tuple[int, bytes]:
        return max(self.vocab.items(), key=lambda x: len(x[1]))


def bytes_to_gpt2_unicode(bytes_token: bytes) -> str:
    return ''.join([gpt2_bytes_to_unicode()[token] for token in bytes_token])


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        with PeekMemorySampler() as mem:
            bpe_tokenizer = BpeTokenizer()
            bpe_tokenizer.train("data/owt_train.txt", 32000, ["<|endoftext|>"])

            print(f"peak: {mem.peak_mib():.0f} MiB")

        print(f"Max worker memory used {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}")

        bpe_tokenizer.save("data/train-bpe-vocab-owt_train.json", "data/train-bpe-merges-owt_train.txt")
        print(bpe_tokenizer.max_length_token())
    pstats.Stats(pr).sort_stats("cumulative").print_stats(40)
