from collections import Counter

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
