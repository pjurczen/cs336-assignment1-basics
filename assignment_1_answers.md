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

## 2.5 Problem (train_bpe_tinystories): BPE Training on TinyStories

**(a)** Training took 34.8 s (29.0 s pretokenization across 10 worker processes, 5.2 s for
the merge loop) and peaked at ~1,540 MiB resident across the parent and all workers
combined, well under the 30 GB budget because the 2.2 GB corpus is memory-mapped and
streamed rather than loaded — memory tracks the ~60,000-entry pretoken frequency table and
its indices, not the size of the data. The longest token is `b' accomplishment'` (15 bytes),
which makes sense: it is a genuine English word frequent enough in TinyStories' narrow,
repetitive vocabulary to earn a slot, and its leading space is an artifact of the GPT-2
pretokenization regex attaching a space to the word that follows it.

**(b)** In the naive implementation, selecting the most frequent pair dominated at 59.4 s
of a 66 s merge loop, because it rescanned the whole pair-count table on every one of the
9,743 merges; replacing that linear scan with count-bucketed lookup reduced the merge loop
to 5.2 s and moved the bottleneck to pretokenization, which is now 83% of the run and is
itself dominated by the GPT-2 regex.

## 2.6 Problem (train_bpe_expts_owt): BPE Training on OpenWebText

**(a)** Training on the 11.9 GB OpenWebText corpus with a vocabulary of 32,000 took
61.2 minutes (4.8 min pretokenization, 56.2 min merges) and peaked at ~3.6 GB resident.
The longest token is 64 bytes — the two-character sequence
`ÃÂ` repeated sixteen times — which is mojibake rather than language: `Ã` (0xC3) and `Â`
(0xC2) are the commonest UTF-8 leading bytes, so they saturate any text that has been
decoded as Latin-1 and re-encoded. It makes sense mechanically (that exact 64-byte
sequence occurs 4,679 times in the training file, e.g. in `can't` where the apostrophe was
mangled) but is linguistically worthless, and it is diagnostic of a data-quality problem in
the corpus. Repeated patterns win the longest-token slot in general because merging a token
with itself doubles its length, so repetitions grow exponentially while real words grow one
byte at a time.

**(b)** The TinyStories tokenizer reflects a small, clean, synthetic corpus — its longest
token is the ordinary English word ` accomplishment`, few distinct pretokens exist so the
merge loop is cheap, and pretokenization is 83% of training time with memory peaking in the
worker processes. The OpenWebText tokenizer reflects scraped web text: far greater pretoken
diversity makes each merge touch roughly ten times as many pretokens, so merging instead
accounts for 92% of the time and memory peaks in the parent's merge-loop state (3.6 GB
versus 1.5 GB), and the vocabulary spends slots on formatting artifacts and encoding
corruption — long hyphen runs and mojibake — rather than on words.

## 2.7 Problem (tokenizer_experiments): Experiments with tokenizers

**(a)** Sampling 10 documents at uniform random byte offsets from each validation set, the
TinyStories tokenizer (10K vocabulary) achieves ~4.0 bytes/token and the OpenWebText
tokenizer (32K vocabulary) ~4.5 bytes/token. The gap reflects the corpora more than the
tokenizers: TinyStories uses deliberately short, simple words, so each token carries fewer
bytes, whereas OpenWebText's longer words and repeated boilerplate pack more bytes into each
token.

**(b)** Encoding the OpenWebText sample with the TinyStories tokenizer drops the compression
ratio from 4.65 to 3.19 bytes/token, a 31% degradation. Vocabulary size accounts for only a
small part of that (going from 50K to 200K buys reference tokenizers ~6% on this corpus), so
the bulk is domain mismatch: the TinyStories merges were learned on short, repetitive
children's prose, and OpenWebText's technical vocabulary, proper nouns, URLs and formatting
runs have no corresponding merges, falling back to near-byte-level tokens.

**(c)** Encoding a random 10 MB slice of OpenWebText with the 32K tokenizer runs at
~3.4 MB/s single-threaded, which is consistent with pretokenization alone measuring ~4.2 MB/s
per process on the same corpus. At that rate the 825 GB Pile would take roughly 2.8 days on
one core, or about 7 hours parallelized across the 10 worker processes the pretokenizer
already uses.

**(d)** uint16 is appropriate choice because it allows us range of [0, 2^16) = [0, 65536) => 0 - 65535
enough than our vocab size of 32k and saves us a lot of space by only taking 2 bytes compared
to python int object which would take 28 bytes for every number in our 0-32k range. uint8 would be only
256 numbers which is too little, uint32 would be just bigger range (and therefore bigger file size) with
no need.
