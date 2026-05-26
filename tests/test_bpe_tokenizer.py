from collections import Counter

import pytest

from bpe_tokenizer import _merge, train

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
    actual_vocab, actual_merges = train("../data/lowlower.txt", 1 + 256 + 6, ["<|endoftext|>"])
    # then
    assert actual_merges == expected_merges
    assert actual_vocab == expected_vocab
