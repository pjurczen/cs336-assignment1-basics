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

## 3.5 Problem (transformer_accounting): Transformer LM resource accounting

**(a)** 1,640,452,800 parameters. This would require 6.56GB memory to load the model assuming float32 for every parameter.

**(b)** For our GPT-2 XL-shaped model the FLOPs can be calculated as 8 bnd²l + 4bn²dl + 6 bnd × d_ff × l + 2bnd × vocab_size
which for one sequence and context_length tokens means b=1 and n=context_length therefore:
8 × 1 × 1024 × 1600² × 48 + 4 × 1 × 1024² × 1600 × 48 + 6 × 1 × 1024 × 1600 × 4288 × 48 + 2 × 1 × 1024 × 1600 × 50257 =
=
= 3'516'769'894'400 FLOPs ≈ 3.5 × 10¹² FLOPs

**(c)**
For the given parameters and input sequence length single most expensive single component is the output projection,
but as model part that would be feed-forward layers in transformers blocks. Within a single transformer layer
the most expensive part is the feed-forward: 16d vs. 8d + 4n ratio, with small input sequence feed-forward is more expensive,
from n=2d its equal and above dominates compute.

self-attention: [FLOPs] 8 bnd² + 4bn²d = 8 × 1024 × 1600² + 4 × 1024² × 1600 = 27'682'406'400
- for all layers: 27'682'406'400 × 48 = 1'328'755'507'200 ≈ 38% of FLOPs
- qkv projections: 6bnd² = 6 × 1024 × 1600² = 15'728'640'000 ≈ 57% of attention FLOPs
- scaled dot product attention: 4bn²d = 4 × 1024² × 1600 = 6'710'886'400 ≈ 24% of attention FLOPs
- attention output projection: 2bnd² = 2 × 1024 × 1600² = 5'242'880'000 ≈ 19% of attention FLOPs

ffn in transformer block: [FLOPs] 16 bnd² or 6 bnd d_ff = 6 × 1024 × 1600 × 4288 = 42'152'755'200
- for all layers: 42'152'755'200 × 48 = 2'023'332'249'600 ≈ 57% of FLOPs

all layers together: (27'682'406'400 + 42'152'755'200) × 48 = 3'352'087'756'800

output projection: [FLOPs] 2bnd × vocab_size = 2 × 1024 × 1600 × 50257 = 164'682'137'600 ≈ 5% of FLOPs

**(d)**

**GPT-2 small** (12 layers, 768 d_model, 12 heads, d_ff 2048)

self-attention: [FLOPs] 8 bnd² + 4bn²d = 8.05E+09
- for all layers: 9.66E+10 ≈ 33.13% of FLOPs
- qkv projections: 6bnd² = 3.62E+09 ≈ 45.00% of attention FLOPs
- scaled dot product attention: 4bn²d = 3.22E+09 ≈ 40.00% of attention FLOPs
- attention output projection: 2bnd² = 1.21E+09 ≈ 15.00% of attention FLOPs

ffn in transformer block: [FLOPs] 16 bnd² or 6 bnd d_ff = 9.66E+09
- for all layers: 1.16E+11 ≈ 39.76% of FLOPs

all layers together: (8.05E+09 + 9.66E+09) × 12 = 2.13E+11

output projection: [FLOPs] 2bnd × vocab_size = 7.90E+10 ≈ 27.10% of FLOPs

total forward pass: 2.92E+11 FLOPs

**GPT-2 medium** (24 layers, 1024 d_model, 16 heads, d_ff 2752)

self-attention: [FLOPs] 8 bnd² + 4bn²d = 1.29E+10
- for all layers: 3.09E+11 ≈ 37.25% of FLOPs
- qkv projections: 6bnd² = 6.44E+09 ≈ 50.00% of attention FLOPs
- scaled dot product attention: 4bn²d = 4.29E+09 ≈ 33.33% of attention FLOPs
- attention output projection: 2bnd² = 2.15E+09 ≈ 16.67% of attention FLOPs

ffn in transformer block: [FLOPs] 16 bnd² or 6 bnd d_ff = 1.73E+10
- for all layers: 4.16E+11 ≈ 50.05% of FLOPs

all layers together: (1.29E+10 + 1.73E+10) × 24 = 7.25E+11

output projection: [FLOPs] 2bnd × vocab_size = 1.05E+11 ≈ 12.70% of FLOPs

total forward pass: 8.30E+11 FLOPs

**GPT-2 large** (36 layers, 1280 d_model, 20 heads, d_ff 3392)

self-attention: [FLOPs] 8 bnd² + 4bn²d = 1.88E+10
- for all layers: 6.76E+11 ≈ 38.25% of FLOPs
- qkv projections: 6bnd² = 1.01E+10 ≈ 53.57% of attention FLOPs
- scaled dot product attention: 4bn²d = 5.37E+09 ≈ 28.57% of attention FLOPs
- attention output projection: 2bnd² = 3.36E+09 ≈ 17.86% of attention FLOPs

ffn in transformer block: [FLOPs] 16 bnd² or 6 bnd d_ff = 2.67E+10
- for all layers: 9.60E+11 ≈ 54.30% of FLOPs

all layers together: (1.88E+10 + 2.67E+10) × 36 = 1.64E+12

output projection: [FLOPs] 2bnd × vocab_size = 1.32E+11 ≈ 7.45% of FLOPs

total forward pass: 1.77E+12 FLOPs

Conclusion: Output projection part of FLOPs decreases from 27% to 5% with increasing model size.
Feed-forward transformer layers increase from 40% to 57% with increasing model size.
For GPT-2 small the proportion of output projection to feed-forward transformer layers is roughly 68:100, for GPT-2 XL it's 8:100.
The attention part of total FLOPs increases from 33% to 38% with increasing model size. 

**(e)**

When increasing context length to 16,384 the total FLOPs increase to 1.34E+14 roughly 40x compared to 1024 context length.
The contribution of model component changes as following:
- attention: 38% -> 74% increase
- ffn transformer layers: 58% -> 24% decrease
- output projection: 5% -> 2% decrease

That's expected and caused by quadratic term of input sequence in scaled dot product attention matmul.

## 4.2 Problem (learning_rate_tuning): Tuning the learning rate

For lr 1e1 the loss was decaying much faster than with lr=1, with lr=1e2 it was first decaying very quickly but then increasing and varying around small range.
With lr=1e3 the training loss was diverging after 2 iterations already and increasing to very big loss over further iterations.

## 4.3 Problem (adamw_accounting): Resource accounting for training with AdamW

**(a)**

Activations:

Transformer block: 2bnd + 5bnd + 2bhn² + bn 35/3d = 56/3 bnd + 2bhn²

- 2 RMSNorms: 2 × batch_size × context_length × d_model = 2bnd
- Multi-head self-attention sublayer: 3bnd + bhn² + bhn² + bnd + bnd = 5bnd + 2bhn²
    - 𝑄𝐾𝑉 projections: 3 × batch_size × context_length × d_model
    - 𝑄𝐾⊤ matrix multiply: batch_size × num_heads × context_length²
    - softmax: batch_size × num_heads × context_length²
    - weighted sum of values: batch_size × context_length × d_model
    - output projection: batch_size × context_length × d_model
- Position-wise feed-forward (SwiGLU): bn 8/3d + bnd + bn 8/3d + bn 8/3d + bn 8/3d = bn 35/3d
    - 𝑊_1: batch_size × context_length × 8/3 × d_model
    - 𝑊_2: batch_size × context_length × d_model
    - SiLU on the gate branch: batch_size × context_length × 8/3 × d_model
    - element-wise product: batch_size × context_length × 8/3 × d_model
    - 𝑊_3: batch_size × context_length × 8/3 × d_model

Final RMSNorm: batch_size × context_length × d_model = bnd

Output embedding: batch_size × context_length × vocab_size = bn vocab_size

Cross-entropy on logits: batch_size × context_length × vocab_size = bn vocab_size

Total activations: num_layers × (56/3 bnd + 2bhn²) + bnd + 2bn vocab_size
lets call it expression A

Parameters:

Token embeddings: vocab_size × d_model

Transformer block: 4 × d_model² + 2 × d_model + 3 × d_model × d_ff

- 2 RMSNorms: 2 × d_model
- Multi-head self-attention sublayer:
    - 𝑄𝐾𝑉 projections: 3 × d_model²
    - output projection: d_model²
- Position-wise feed-forward (SwiGLU): 3 × d_model × d_ff

Final RMSNorm: d_model

Output embedding: d_model × vocab_size

Total: 2 × d_model × vocab_size + num_layers × (4 × d_model² + 2 × d_model + 3 × d_model × d_ff) + d_model
for d_ff = 8/3 d_model:
2 × d_model × vocab_size + num_layers × (12 × d_model² + 2 × d_model) + d_model
lets call it as expression P

Gradients: exact same as parameters count so P

Optimizer state:

2 parameters for each parameter, so 2P

Total: A + 4P = num_layers × (56/3 bnd + 2bhn²) + bnd + 2bn vocab_size + 4 × (2 × d × vocab_size + num_layers × (12d² + 2d) + d) =
num_layers × (56/3 bnd + 2bhn²) + bnd + 2bn vocab_size + 8 d × vocab_size + 4 × num_layers × (12d² + 2d) + 4d

**(b)**

For GPT-2 XL-shaped model:
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25

Total:
1'468'006'400 × batch_size + 2'516'582'400 × batch_size + 1'638'400 × batch_size + 102'926'336 × batch_size + 643'289'600 + 4 × 1'474'713'600 + 6400 =
3'984'588'800 × batch_size + 1'638'400 × batch_size + 102'926'336 × batch_size + 6'542'150'400 =
4'089'153'536 × batch_size + 6'542'150'400

a=4'089'153'536
b=6'542'150'400

80 GiB = 85'899'345'920 bytes

(4'089'153'536 × batch_size + 6'542'150'400) × 4 bytes = 16'356'614'144 × batch_size + 26'168'601'600 bytes

solving for batch_size:
16'356'614'144 × batch_size + 26'168'601'600 = 85'899'345'920
16'356'614'144 × batch_size = 59'730'744'320
batch_size = 3.6517792615

rounding down answer is batch_size=3