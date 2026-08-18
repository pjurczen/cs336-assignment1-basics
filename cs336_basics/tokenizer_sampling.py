import mmap
import os
import pathlib
from random import Random

from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

TINY_STORIES_VOCAB_PATH = pathlib.Path(__file__).resolve().parent / "../data/train-bpe-vocab-TinyStoriesV2-GPT4-train.json"
TINY_STORIES_MERGES_PATH = pathlib.Path(__file__).resolve().parent / "../data/train-bpe-merges-TinyStoriesV2-GPT4-train.txt"
TINY_STORIES_FILE_PATH = pathlib.Path(__file__).resolve().parent / "../data/TinyStoriesV2-GPT4-valid.txt"

OPEN_WEB_TEXT_VOCAB_PATH = pathlib.Path(__file__).resolve().parent / "../data/train-bpe-vocab-owt_train.json"
OPEN_WEB_TEXT_MERGES_PATH = pathlib.Path(__file__).resolve().parent / "../data/train-bpe-merges-owt_train.txt"
OPEN_WEB_TEXT_FILE_PATH = pathlib.Path(__file__).resolve().parent / "../data/owt_valid.txt"

SPECIAL_TOKEN: str = "<|endoftext|>"

SAMPLE_SIZE: int = 10


def compression_ratio_sampling(vocab_path: os.PathLike, merges_path: os.PathLike, text_file_path: os.PathLike, special_token: str) -> None:
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, [special_token])
    samples_compression_ratio: list[float] = []
    SEP: bytes = SPECIAL_TOKEN.encode('utf-8')
    with open(text_file_path, "r") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size: int = file.tell()
        file.seek(0)
        random = Random()
        sample_indices: list[int] = random.sample(range(file_size), SAMPLE_SIZE)
        sample_chunks_boundaries: list[tuple[int, int]] = []
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(0, len(sample_indices)):
                sample_idx: int = sample_indices[i]
                s: int = mm.rfind(SEP, 0, min(len(mm), sample_idx + len(SEP) - 1))
                start: int = 0 if s < 0 else s + len(SEP)
                e: int = mm.find(SEP, start)
                end: int = len(mm) if e < 0 else e
                sample_chunks_boundaries.append((start, end))

        for chunk_boundaries in sample_chunks_boundaries:
            file.seek(chunk_boundaries[0])
            chunk: str = file.read(chunk_boundaries[1] - chunk_boundaries[0])
            ids: list[int] = list(chunk.encode('utf-8'))
            encoded_ids: list[int] = tokenizer.encode(chunk)
            samples_compression_ratio.append(len(ids) / len(encoded_ids))

    print(samples_compression_ratio)
    print(sum(samples_compression_ratio) / len(samples_compression_ratio))


if __name__ == "__main__":
    compression_ratio_sampling(TINY_STORIES_VOCAB_PATH, TINY_STORIES_MERGES_PATH, TINY_STORIES_FILE_PATH, SPECIAL_TOKEN)
    compression_ratio_sampling(OPEN_WEB_TEXT_VOCAB_PATH, OPEN_WEB_TEXT_MERGES_PATH, OPEN_WEB_TEXT_FILE_PATH, SPECIAL_TOKEN)
    compression_ratio_sampling(TINY_STORIES_VOCAB_PATH, TINY_STORIES_MERGES_PATH, OPEN_WEB_TEXT_FILE_PATH, SPECIAL_TOKEN)
