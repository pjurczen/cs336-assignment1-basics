import cProfile
import json
import pstats
import resource
import time
from collections import Counter

from cs336_basics.bpe_state import BpeState
from cs336_basics.peek_memory_sampler import PeekMemorySampler
from cs336_basics.pretokenization import pretokenize
from tests.common import gpt2_bytes_to_unicode

BYTES_COUNT = 256  # this is really an overkill since in UTF-8 192, 193 and >=245 bytes are not used, but we learn those if at inference time invalid input was somehow passed


class BpeTokenizer:
    vocab: dict[int, bytes]
    id_vocab: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    merges_dict: dict[bytes, bytes]
    special_tokens: list[str]

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
        bpe_tokenizer.special_tokens = special_tokens
        bpe_tokenizer.id_vocab = {id: byte for byte, id in vocab.items()}
        bpe_tokenizer.merges_dict = {x[0]: x[1] for x in merges}
        return bpe_tokenizer

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:  # (vocab, merges)
        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {x: special_tokens[x].encode('utf-8') for x in range(len(special_tokens))}
        self.special_tokens = special_tokens.copy()
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
        self.vocab = vocab
        self.id_vocab = {id: byte for byte, id in vocab.items()}
        self.merges = merges
        self.merges_dict = {x[0]: x[1] for x in merges}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = list(map(int, text.encode('utf-8')))
        encoded_ids: list[int] = [self.id_vocab[bytes([id])] for id in ids]
        i: int = 0
        while i < len(encoded_ids):
            if i < len(encoded_ids) - 1 and (self.vocab[encoded_ids[i]] + self.vocab[encoded_ids[i + 1]]) in self.merges_dict:
                encoded_ids[i] = self.merges_dict[self.vocab[encoded_ids[i]] + self.vocab[encoded_ids[i + 1]]]
                encoded_ids[i + 1].pop()
            else:
                i += 1
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        bytes_list: list[bytes] = list(map(self.vocab.get, ids))
        text = b"".join(bytes_list).decode('utf-8')
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
