# CS336 Assignment 1 — Written Answers

## 2.1 Problem (unicode1): Understanding Unicode

**(a)** chr(0) is code point U+0000 - a non-printing control character

**(b)** Printed representation is invisible while __repr__() prints visible '\x00'

**(c)** chr(0) alone in python interpreter prints its __repr__() which is '\x00', in text it's invisible but present (affects text length by 1).

## 2.2 Problem (unicode2): Unicode Encodings

**(a)** UTF-8 encodes ASCII in a single byte while UTF-16 and UTF-32 pad it with
null bytes carrying no information, so a byte-level tokenizer would spend merges
undoing the encoding rather than learning sub-word structure. UTF-8 also guarantees
that no ASCII byte value can occur inside a multi-byte character, which makes
searching raw bytes for a separator — and therefore splitting a corpus at arbitrary
offsets for parallel pretokenization — provably correct, whereas UTF-16 admits
constructible false positives and requires knowing the stream's endianness.

**(b)**

```python
decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
```

The function decodes each byte independently, but UTF-8 is variable-width, so the
individual bytes of a multi-byte character such as `こ` (`E3 81 93`) are not
decodable on their own.

**(c)**

```python
bytes([0x37, 0xF4]).decode("utf-8")
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf4 in position 1: unexpected end of data
```

`0xF4` is a valid four-byte leading byte whose `11110` prefix announces three
continuation bytes, but the sequence ends immediately after it, so no character is
completed.
