from collections import Counter
from typing import Optional

from cs336_basics.pretoken_counter import PretokenCounter


def test_from_counter_empty():
    pretoken_counter = PretokenCounter.from_counter(Counter({}))
    assert pretoken_counter is not None
    assert pretoken_counter.counts == {}
    assert pretoken_counter.buckets == {}
    assert pretoken_counter.max_count == 0


def test_from_counter_one_entry():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2}))
    assert pretoken_counter is not None
    assert pretoken_counter.counts == {(b'a', b'b'): 2}
    assert pretoken_counter.buckets == {2: {(b'a', b'b')}}
    assert pretoken_counter.max_count == 2


def test_from_counter_shared_count():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'c', b'da'): 5, (b'dd', b'a'): 5}))
    assert pretoken_counter is not None
    assert pretoken_counter.counts == {(b'a', b'b'): 2, (b'c', b'da'): 5, (b'dd', b'a'): 5}
    assert pretoken_counter.buckets == {2: {(b'a', b'b')}, 5: {(b'c', b'da'), (b'dd', b'a')}}
    assert pretoken_counter.max_count == 5


def test_highest_empty():
    pretoken_counter = PretokenCounter.from_counter(Counter({}))
    pair: Optional[tuple[bytes, bytes]] = pretoken_counter.highest()
    assert pair is None


def test_highest_one_entry():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2}))
    pair: Optional[tuple[bytes, bytes]] = pretoken_counter.highest()
    assert pair == (b'a', b'b')


def test_highest_two_entries_same_count():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'ba', b'a'): 2}))
    pair: Optional[tuple[bytes, bytes]] = pretoken_counter.highest()
    assert pair == (b'ba', b'a')


def test_highest_multiple_entries():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'a', b'a'): 6}))
    pair: Optional[tuple[bytes, bytes]] = pretoken_counter.highest()
    assert pair == (b'a', b'a')


def test_add_new_pair():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'ba', b'a'): 2}))
    pretoken_counter.add((b'c', b'a'), 3)
    assert pretoken_counter.max_count == 3
    assert pretoken_counter.highest() == (b'c', b'a')
    assert pretoken_counter.buckets == {3: {(b'c', b'a')}, 2: {(b'a', b'b'), (b'ba', b'a')}}
    assert pretoken_counter.counts == {(b'c', b'a'): 3, (b'a', b'b'): 2, (b'ba', b'a'): 2}


def test_add_change_highest_pair():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'ba', b'a'): 2}))
    pretoken_counter.add((b'a', b'b'), 1)
    assert pretoken_counter.max_count == 3
    assert pretoken_counter.highest() == (b'a', b'b')
    assert pretoken_counter.buckets == {3: {(b'a', b'b')}, 2: {(b'ba', b'a')}}
    assert pretoken_counter.counts == {(b'a', b'b'): 3, (b'ba', b'a'): 2}


def test_add_change_highest_pair_same_count_higher_lexicography():
    pretoken_counter = PretokenCounter.from_counter(Counter({(b'a', b'b'): 2, (b'ba', b'a'): 3, (b'b', b'a'): 4}))
    pretoken_counter.add((b'ba', b'a'), 1)
    assert pretoken_counter.max_count == 4
    assert pretoken_counter.highest() == (b'ba', b'a')
    assert pretoken_counter.buckets == {4: {(b'b', b'a'), (b'ba', b'a')}, 2: {(b'a', b'b')}}
    assert pretoken_counter.counts == {(b'b', b'a'): 4, (b'ba', b'a'): 4, (b'a', b'b'): 2}
