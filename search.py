# search.py
# Step 3 of the project: take a user query and return the most relevant documents.
# Four search methods are available, so the evaluation can compare them:
#   "bm25"          - keyword match only (traditional information retrieval)
#   "dense"         - meaning match only (embeddings)
#   "hybrid"        - both lists combined with Reciprocal Rank Fusion (RRF)
#   "hybrid_rerank" - hybrid, then a cross-encoder re-scores the top chunks (default, most accurate)
# Search works on chunks; the final answer is per document: each document is ranked by
# its best chunk, and that chunk is shown as the excerpt.
# Run:  python search.py   (loads everything, runs a few demo queries, then lets you type your own)

import time                   # measure search speed
import numpy as np            # arrays of numbers
import pandas as pd           # chunk and article tables
import bm25s                  # BM25 keyword search
import faiss                  # fast vector search
import config                 # project settings
from text_utils import tokenize   # same keyword rules as the index

METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]   # all search methods, simplest first


# Loads data, indexes and models once. Everything is kept in one dictionary called "engine",
# which is passed to every other function. Loading takes some seconds; searching is then fast.
def load_engine():
    from sentence_transformers import SentenceTransformer, CrossEncoder   # imported here because it loads slowly
    chunks = pd.read_parquet(config.CHUNKS_FILE)          # all chunks (row number = position in indexes)
    articles = pd.read_parquet(config.ARTICLES_FILE).set_index("doc_id")   # full articles, looked up by doc_id
    bm25 = bm25s.BM25.load(config.BM25_DIR)               # keyword index from build_index.py
    vectors = np.load(config.EMBEDDINGS_FILE).astype(np.float32)   # meaning vectors (faiss needs float32)
    index = faiss.IndexFlatIP(vectors.shape[1])           # exact search by inner product (= cosine, vectors are normalized)
    index.add(vectors)                                    # put all chunk vectors into the index
    return {"chunks": chunks,                             # chunk table
            "articles": articles,                         # article table
            "bm25": bm25,                                 # keyword index
            "index": index,                               # vector index
            "embedder": SentenceTransformer(config.EMBED_MODEL),   # turns a query into a vector
            "reranker": CrossEncoder(config.RERANK_MODEL)}         # re-scores (query, chunk) pairs


# Keyword search: returns chunk row numbers, best first. Chunks sharing no keyword are dropped.
def bm25_search(engine, query, k):
    tokens = tokenize(query)                              # query -> keywords, same rules as the index
    if not tokens:                                        # query was only stopwords, e.g. "what is the"
        return []                                         # nothing to match
    k = min(k, len(engine["chunks"]))                     # cannot ask for more chunks than exist
    rows, scores = engine["bm25"].retrieve([tokens], k=k, show_progress=False)   # top-k chunks
    return [int(r) for r, s in zip(rows[0], scores[0]) if s > 0]   # keep only real keyword matches


# Meaning search: returns chunk row numbers, best first.
def dense_search(engine, query, k):
    vector = engine["embedder"].encode([query], normalize_embeddings=True)   # query -> 384 numbers
    scores, rows = engine["index"].search(vector.astype(np.float32), k)      # k most similar chunk vectors
    return [int(r) for r in rows[0] if r >= 0]            # faiss uses -1 for "no result"


# Reciprocal Rank Fusion: combines several ranked lists into one.
# Each chunk gets 1 / (RRF_K + rank) from every list it appears in; the scores are added.
# Example with RRF_K = 60: rank 1 in BM25 and rank 3 in dense -> 1/61 + 1/63 = 0.0323.
# It uses ranks only, so BM25 scores and cosine scores never need to be put on the same scale.
def rrf_fuse(ranked_lists):
    scores = {}                                           # chunk row -> fused score
    for ranked in ranked_lists:                           # each method's list
        for rank, row in enumerate(ranked, start=1):      # rank starts at 1
            scores[row] = scores.get(row, 0) + 1 / (config.RRF_K + rank)   # add this list's share
    return sorted(scores, key=scores.get, reverse=True)   # chunk rows, highest fused score first


# Cross-encoder reranking: the model reads the query and each chunk together, which is
# more accurate than comparing two separate vectors, but slower, so only the top chunks are re-scored.
def rerank(engine, query, rows):
    top = rows[:config.RERANK_TOP]                        # chunks worth the extra cost
    if not top:                                           # nothing to rerank
        return rows                                       # return as is
    pairs = [(query, engine["chunks"]["text"].iat[r]) for r in top]   # (query, chunk text) pairs
    scores = engine["reranker"].predict(pairs)            # one relevance score per pair
    order = np.argsort(-np.asarray(scores))               # positions sorted by score, highest first
    return [top[i] for i in order] + rows[config.RERANK_TOP:]   # reranked top, then the rest unchanged


# Runs the chosen method and returns chunk row numbers, best first.
def rank_chunks(engine, query, method):
    if method == "bm25":                                  # keywords only
        return bm25_search(engine, query, config.CANDIDATES)
    if method == "dense":                                 # meaning only
        return dense_search(engine, query, config.CANDIDATES)
    fused = rrf_fuse([bm25_search(engine, query, config.CANDIDATES),     # combine both lists
                      dense_search(engine, query, config.CANDIDATES)])
    if method == "hybrid":                                # combined, no reranking
        return fused
    if method == "hybrid_rerank":                         # combined + reranked
        return rerank(engine, query, fused)
    raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}")   # typo protection


# Turns a ranked chunk list into a ranked document list.
# A document's position is the position of its best chunk; that chunk becomes the excerpt.
def chunks_to_docs(engine, rows, top_n):
    results, seen = [], set()                             # output list, doc_ids already added
    for row in rows:                                      # chunks from best to worst
        chunk = engine["chunks"].iloc[row]                # that chunk's row in the table
        if chunk["doc_id"] in seen:                       # this document is already in the list
            continue                                      # skip its weaker chunks
        seen.add(chunk["doc_id"])                         # remember it
        results.append({"rank": len(results) + 1,         # 1, 2, 3, ...
                        "doc_id": chunk["doc_id"],        # which article
                        "chunk_id": chunk["chunk_id"],    # which part of it matched best
                        "excerpt": chunk["text"]})        # text shown to the user
        if len(results) == top_n:                         # have enough documents
            break                                         # stop
    return results                                        # list of result dicts


# Main entry point: query in, top documents out, plus how long the search took.
def search(engine, query, top_n=config.TOP_N, method="hybrid_rerank"):
    start = time.time()                                   # start timer
    rows = rank_chunks(engine, query, method)             # ranked chunks
    results = chunks_to_docs(engine, rows, top_n)         # ranked documents
    return results, time.time() - start                   # results and seconds taken


# Returns the full cleaned text of one article (used by the summarizer).
def get_article(engine, doc_id):
    return engine["articles"].at[doc_id, "text"]          # look up by doc_id


# Prints results in a short, readable form.
def show(results, seconds, method):
    print(f"\n[{method}] {len(results)} documents in {seconds * 1000:.0f} ms")   # header with timing
    for r in results:                                     # each document
        print(f"  {r['rank']}. {r['doc_id']}  {r['excerpt'][:110]}...")   # rank, id, start of excerpt


def main():
    print("Loading search engine...")                     # loading takes a while
    engine = load_engine()                                # load everything once
    demo = ["plane crash investigation", "football transfer record fee"]   # example queries
    for query in demo:                                    # each demo query
        print(f"\n===== {query} =====")                   # query header
        for method in METHODS:                            # compare all four methods
            results, seconds = search(engine, query, method=method)   # run search
            show(results, seconds, method)                # print it
    while True:                                           # let the user try their own queries
        query = input("\nType a query (or press Enter to quit): ").strip()   # read input
        if not query:                                     # empty input
            break                                         # quit
        results, seconds = search(engine, query)          # default method: hybrid_rerank
        show(results, seconds, "hybrid_rerank")           # print it


if __name__ == "__main__":
    main()