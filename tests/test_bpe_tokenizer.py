from collections import Counter

import pytest
import tiktoken

from cs336_basics.bpe_state import BpeState
from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.pretokenization import merge_pretoken
from tests.common import FIXTURES_PATH
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

testdata = [
    (Counter({(b'l', b'o', b'w'): 5}), (b'l', b'o'), Counter({(b'lo', b'w'): 5})),
    (Counter({(b'l', b'o', b'w'): 12}), (b'o', b'w'), Counter({(b'l', b'ow'): 12})),
    (Counter({(b'l', b'o', b'w'): 12}), (b'x', b'y'), Counter({(b'l', b'o', b'w'): 12})),
    (Counter({(b'a', b'a', b'a'): 6}), (b'a', b'a'), Counter({(b'aa', b'a'): 6})),
    (Counter({(b'a', b'a', b'a', b'a'): 6}), (b'a', b'a'), Counter({(b'aa', b'aa'): 6})),
    (Counter({(b'a',): 6}), (b'a', b'a'), Counter({(b'a',): 6})),
    (Counter(), (b'a', b'a'), Counter()),
]


@pytest.mark.parametrize("pretoken_vocab,pair,expected_output", testdata)
def test_merge(pretoken_vocab: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes],
               expected_output: Counter[tuple[bytes, ...]]):
    bpe_state: BpeState = BpeState(pretoken_vocab)
    bpe_state.merge(pair)
    pairs_to_pretoken_index = bpe_state._calculate_pairs_to_pretokens_index(bpe_state.pretoken_vocab)
    counts = bpe_state._count_adjacent_pairs(bpe_state.pretoken_vocab)
    assert expected_output == bpe_state.pretoken_vocab
    assert pairs_to_pretoken_index == bpe_state.pairs_to_pretokens_index
    assert counts == bpe_state.pretoken_counter.counts


def test_train_low_lower():
    # given
    bpe_tokenizer = BpeTokenizer()
    expected_vocab: dict[int, bytes] = {0: "<|endoftext|>".encode('utf-8')}
    for x in range(1, 256 + 1):
        expected_vocab[x] = bytes([x - 1])
    for i, v in enumerate(['st', 'est', 'ow', 'low', 'west', 'ne']):
        expected_vocab[1 + 256 + i] = v.encode('utf-8')
    expected_merges = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    # when
    bpe_tokenizer.train("data/lowlower.txt", 1 + 256 + 6, ["<|endoftext|>"])
    actual_vocab = bpe_tokenizer.vocab
    actual_merges = bpe_tokenizer.merges
    # then
    assert actual_merges == expected_merges
    assert actual_vocab == expected_vocab


def test_merge_pretoken_in_the_middle():
    pair: tuple[bytes, bytes] = (b'a', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'aa', b'l')
    # deltas
    assert len(count_deltas) == 5
    # removed
    assert count_deltas[(b'b', b'a')] == -1
    assert count_deltas[(b'a', b'a')] == -1
    assert count_deltas[(b'a', b'l')] == -1
    # added
    assert count_deltas[(b'b', b'aa')] == 1
    assert count_deltas[(b'aa', b'l')] == 1


def test_merge_pretoken_at_the_start():
    pair: tuple[bytes, bytes] = (b'b', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'ba', b'a', b'l')
    # deltas
    assert len(count_deltas) == 3
    # removed
    assert count_deltas[(b'b', b'a')] == -1
    assert count_deltas[(b'a', b'a')] == -1
    # added
    assert count_deltas[(b'ba', b'a')] == 1


def test_merge_pretoken_at_the_end():
    pair: tuple[bytes, bytes] = (b'a', b'l')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'a', b'al')
    # deltas
    assert len(count_deltas) == 3
    # removed
    assert count_deltas[(b'a', b'l')] == -1
    assert count_deltas[(b'a', b'a')] == -1
    # added
    assert count_deltas[(b'a', b'al')] == 1


def test_merge_pretoken_overlapping():
    pair: tuple[bytes, bytes] = (b'a', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'aa', b'a', b'l')
    # deltas
    assert len(count_deltas) == 4
    # removed
    assert count_deltas[(b'a', b'a')] == -2
    assert count_deltas[(b'b', b'a')] == -1
    # added
    assert count_deltas[(b'b', b'aa')] == 1
    assert count_deltas[(b'aa', b'a')] == 1


def test_merge_pretoken_overlapping_double():
    pair: tuple[bytes, bytes] = (b'a', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'aa', b'aa', b'l')
    # deltas
    assert len(count_deltas) == 6
    # removed
    assert count_deltas[(b'a', b'a')] == -3
    assert count_deltas[(b'b', b'a')] == -1
    assert count_deltas[(b'a', b'l')] == -1
    # added
    assert count_deltas[(b'b', b'aa')] == 1
    assert count_deltas[(b'aa', b'aa')] == 1
    assert count_deltas[(b'aa', b'l')] == 1


def test_merge_pretoken_length_2():
    pair: tuple[bytes, bytes] = (b'b', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'ba',)
    # deltas
    assert len(count_deltas) == 1
    # removed
    assert count_deltas[(b'b', b'a')] == -1


def test_merge_pretoken_absent():
    pair: tuple[bytes, bytes] = (b'c', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'a', b'a', b'l')
    # deltas
    assert len(count_deltas) == 0


def test_merge_pretoken_2_occurences():
    pair: tuple[bytes, bytes] = (b'a', b'l')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'l', b'c', b'a', b'l')
    new_pretoken = merge_pretoken(pair, pretoken)
    old_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(pretoken)
    new_counts: dict[tuple[bytes, bytes], int] = BpeState._count_byte_pairs(new_pretoken)
    count_deltas: dict[tuple[bytes, bytes], int] = BpeState.get_count_deltas(old_counts, new_counts)
    assert new_pretoken == (b'b', b'al', b'c', b'al')
    # deltas
    assert len(count_deltas) == 7
    # removed
    assert count_deltas[(b'a', b'l')] == -2
    assert count_deltas[(b'b', b'a')] == -1
    assert count_deltas[(b'l', b'c')] == -1
    assert count_deltas[(b'c', b'a')] == -1
    # added
    assert count_deltas[(b'b', b'al')] == 1
    assert count_deltas[(b'al', b'c')] == 1
    assert count_deltas[(b'c', b'al')] == 1


def test_encode_iterable_single_characters_stream():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    text: str = "This is a cat<|endoftext|>That cat likes to eat"
    reference_ids: list[int] = reference_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    all_ids: list[int] = []
    for _id in tokenizer.encode_iterable(text.__iter__()):
        all_ids.append(_id)

    assert reference_ids == all_ids


def test_encode_iterable_chunked_stream():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    text: list[str] = ["This is a cat\n", "<|endoftext|>\n", "That cat likes to eat\n", "<|endoftext|>\n"]
    reference_ids: list[int] = reference_tokenizer.encode(''.join(text), allowed_special={"<|endoftext|>"})
    all_ids: list[int] = []
    for _id in tokenizer.encode_iterable(text.__iter__()):
        all_ids.append(_id)

    assert reference_ids == all_ids


def test_encode_iterable_tinystories_short():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample_short.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample_short.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    assert reference_ids == all_ids
