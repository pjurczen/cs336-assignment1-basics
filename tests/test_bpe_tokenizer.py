from collections import Counter

import pytest

from cs336_basics.bpe_tokenizer import _merge, train, _merge_pretoken

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
    actual_output = _merge(pretoken_vocab, pair)
    assert expected_output == actual_output


def test_train_low_lower():
    # given
    expected_vocab: dict[int, bytes] = {0: "<|endoftext|>".encode('utf-8')}
    for x in range(1, 256 + 1):
        expected_vocab[x] = bytes([x - 1])
    for i, v in enumerate(['st', 'est', 'ow', 'low', 'west', 'ne']):
        expected_vocab[1 + 256 + i] = v.encode('utf-8')
    expected_merges = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    # when
    actual_vocab, actual_merges = train("data/lowlower.txt", 1 + 256 + 6, ["<|endoftext|>"])
    # then
    assert actual_merges == expected_merges
    assert actual_vocab == expected_vocab


def test_merge_pretoken_in_the_middle():
    pair: tuple[bytes, bytes] = (b'a', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
    assert new_pretoken == (b'ba',)
    # deltas
    assert len(count_deltas) == 1
    # removed
    assert count_deltas[(b'b', b'a')] == -1


def test_merge_pretoken_absent():
    pair: tuple[bytes, bytes] = (b'c', b'a')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'a', b'l')
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
    assert new_pretoken == (b'b', b'a', b'a', b'l')
    # deltas
    assert len(count_deltas) == 0


def test_merge_pretoken_2_occurences():
    pair: tuple[bytes, bytes] = (b'a', b'l')
    pretoken: tuple[bytes, ...] = (b'b', b'a', b'l', b'c', b'a', b'l')
    new_pretoken, count_deltas = _merge_pretoken(pair, pretoken)
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
