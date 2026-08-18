from cs336_basics.pretokenization import split_text_to_chunks, pretokenize_text


def test_split_text_to_chunks_single_word():
    result = split_text_to_chunks(b'the', [])
    assert result == [b'the']


def test_split_text_to_chunks_with_single_special_token():
    result = split_text_to_chunks(b'the<|endoftext|>world', [b'<|endoftext|>'])
    assert result == [b'the', b'<|endoftext|>', b'world']


def test_split_text_to_chunks_with_double_special_token():
    result = split_text_to_chunks(b'the<|endoftext|><|endoftext|>world', [b'<|endoftext|>'])
    assert result == [b'the', b'<|endoftext|>', b'<|endoftext|>', b'world']


def test_pretokenize_text_single_character():
    result = pretokenize_text(b's')
    assert result == [b's']


def test_pretokenize_text_two_words():
    result = pretokenize_text(b'the cat')
    assert result == [b'the', b' cat']


def test_split_text_to_chunks_new_line_characters():
    result = split_text_to_chunks(b'the cat\n<|endoftext|>\nlikes to eat\n<|endoftext|>\n', [b'<|endoftext|>'])
    assert result == [b'the cat\n', b'<|endoftext|>', b'\nlikes to eat\n', b'<|endoftext|>', b'\n']
