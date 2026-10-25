# build_index.py
# Step 2 of the project: build the two search indexes once, so searching later is fast.
#   1. BM25 keyword index  - finds chunks that share the query's exact words.
#   2. Embedding vectors   - one list of 384 numbers per chunk that captures its meaning,
#                            so "car crash" can find a chunk that says "vehicle collision".
# Both are saved inside data/. search.py loads them.
# Run:  python build_index.py   (about 10-25 minutes on a laptop CPU, mostly for the embeddings)

import os                     # file sizes and folders
import time                   # measure how long each step takes
import numpy as np            # arrays of numbers
import pandas as pd           # read the chunk table
import bm25s                  # fast BM25 keyword search
import config                 # project settings
from text_utils import tokenize   # same keyword rules as search time


# Returns the total size of a file or folder in megabytes (for the report).
def size_mb(path):
    if os.path.isfile(path):                              # a single file
        return os.path.getsize(path) / 1e6                # bytes -> MB
    total = 0                                             # a folder: add up every file inside
    for name in os.listdir(path):                         # each file in the folder
        total += os.path.getsize(os.path.join(path, name))    # add its size
    return total / 1e6                                    # bytes -> MB


# Builds the BM25 keyword index from all chunks and saves it to disk.
def build_bm25(chunks):
    start = time.time()                                   # start timer
    tokens = [tokenize(t) for t in chunks["text"]]        # each chunk -> list of keywords
    retriever = bm25s.BM25()                              # empty BM25 index (default k1=1.5, b=0.75)
    retriever.index(tokens, show_progress=False)          # count words and pre-compute scores
    retriever.save(config.BM25_DIR)                       # save to data/bm25_index/
    vocab = len({w for doc in tokens for w in doc})       # number of different keywords
    print("----- BM25 index -----")                       # header
    print(f"Chunks indexed            : {len(tokens)}")   # should equal number of chunks
    print(f"Unique keywords           : {vocab}")         # vocabulary size
    print(f"Avg keywords per chunk    : {np.mean([len(t) for t in tokens]):.1f}")   # after stopword removal
    print(f"Build time                : {time.time() - start:.1f} s")   # how long it took
    print(f"Size on disk              : {size_mb(config.BM25_DIR):.1f} MB")   # storage cost


# Checks how many chunks are longer than the embedding model can read.
# The model reads at most 256 tokens; anything after that is ignored when making the vector.
def check_truncation(model, chunks, sample_size=1000):
    sample = chunks["text"].sample(min(sample_size, len(chunks)), random_state=config.RANDOM_SEED)  # random chunks
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in sample]   # token count of each chunk
    limit = model.max_seq_length                          # model's maximum input length
    over = sum(1 for n in lengths if n > limit)           # how many are too long
    print(f"Model max tokens          : {limit}")         # usually 256
    print(f"Avg tokens per chunk      : {np.mean(lengths):.0f}")   # typical chunk length in tokens
    print(f"Chunks over the limit     : {over / len(lengths) * 100:.1f}% (end of those chunks is cut)")   # truncation rate


# Turns every chunk into a 384-number meaning vector and saves them all in one file.
def build_embeddings(chunks):
    from sentence_transformers import SentenceTransformer   # imported here because it loads slowly
    print("----- Embeddings -----")                       # header
    model = SentenceTransformer(config.EMBED_MODEL)       # downloads the model the first time (~90 MB)
    check_truncation(model, chunks)                       # report how much text the model cannot see
    start = time.time()                                   # start timer
    vectors = model.encode(list(chunks["text"]),          # all chunk texts
                           batch_size=config.EMBED_BATCH_SIZE,   # chunks per batch
                           show_progress_bar=True,        # progress bar in the terminal
                           normalize_embeddings=True,     # length 1, so dot product = cosine similarity
                           convert_to_numpy=True)         # return a numpy array
    np.save(config.EMBEDDINGS_FILE, vectors.astype(np.float16))   # float16 halves file size; precision loss is tiny
    print(f"Vectors shape             : {vectors.shape}") # (number of chunks, 384)
    print(f"Encode time               : {time.time() - start:.1f} s")   # how long it took
    print(f"Size on disk              : {size_mb(config.EMBEDDINGS_FILE):.1f} MB")   # storage cost


def main():
    chunks = pd.read_parquet(config.CHUNKS_FILE)          # chunks made by data_prep.py
    print(f"Loaded {len(chunks)} chunks from {config.CHUNKS_FILE}")   # confirm input
    build_bm25(chunks)                                    # keyword index (fast)
    build_embeddings(chunks)                              # meaning vectors (slow part)
    print("Done. Indexes saved in", config.DATA_DIR)      # finished


if __name__ == "__main__":
    main()