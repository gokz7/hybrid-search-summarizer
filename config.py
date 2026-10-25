# config.py
# All project settings live here, so every other file reads the same values.
# If you want to change a size, a path or a model, change it only in this file.

import os  # used to build file paths

# ---------- Folders and files ----------
DATA_DIR = "data"                                             # folder for all generated data
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.parquet")    # cleaned full articles
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.parquet")        # articles split into chunks
BM25_DIR = os.path.join(DATA_DIR, "bm25_index")               # saved BM25 keyword index (a folder)
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")    # one meaning-vector per chunk
RESULTS_DIR = "results"                                       # folder for evaluation outputs

# ---------- Data preparation ----------
DATASET_NAME = "abisee/cnn_dailymail"   # Hugging Face dataset id
DATASET_CONFIG = "3.0.0"                # version with human-written highlights
DATASET_SPLIT = "test"                  # 11,490 articles, small download (no training happens here)
NUM_ARTICLES = 10000                    # how many articles to keep in the corpus
MIN_ARTICLE_WORDS = 100                 # articles shorter than this are dropped
CHUNK_WORDS = 200                       # words per chunk
CHUNK_OVERLAP = 50                      # words shared between neighbouring chunks
MIN_NEW_WORDS = 50                      # a last chunk adding fewer new words than this is merged
RANDOM_SEED = 42                        # fixed seed so results can be repeated

# ---------- Search ----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # small, fast embedding model (384 numbers per text)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # reads query + chunk together to re-score them
EMBED_BATCH_SIZE = 64                   # chunks encoded at once (lower it if you run out of memory)
CANDIDATES = 100                        # chunks taken from BM25 and from embeddings before combining
RRF_K = 60                              # constant in Reciprocal Rank Fusion: score = 1 / (RRF_K + rank)
RERANK_TOP = 30                         # how many combined chunks the reranker re-scores
TOP_N = 5                               # documents returned by default

# ---------- Summarization (Gemini) ----------
GEMINI_MODEL = "gemini-3.5-flash-lite"  # main model: fast, reliably available, cheap (names from list_models.py)
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"]   # backups if the main model is busy
TEMPERATURE = 0.2                       # low = more factual, less creative wording
SUMMARY_LENGTHS = {"short": 50, "medium": 120, "long": 250}   # target words for each length option
DOC_WORDS_FOR_LLM = 400                 # words of each article sent to the LLM (news puts key facts first)
LLM_CACHE_FILE = os.path.join(DATA_DIR, "llm_cache.json")     # saved LLM answers, so repeat calls cost nothing
MAX_RETRIES = 4                         # attempts per LLM call before giving up

# ---------- Search evaluation ----------
TEST_SIZE = 200                         # articles in the test set
QUERY_BATCH = 10                        # articles per LLM call when writing test queries (200 / 10 = 20 calls)
QUERY_SOURCE_WORDS = 250                # words of each article shown to the LLM when writing its queries
EVAL_TOP_K = 10                         # documents retrieved per test query (for Hit@10)
TEST_QUERIES_FILE = os.path.join(RESULTS_DIR, "test_queries.csv")          # generated queries (fixed test set)
SEARCH_EVAL_FILE = os.path.join(RESULTS_DIR, "search_eval.csv")            # metric table
SEARCH_DETAILS_FILE = os.path.join(RESULTS_DIR, "search_eval_details.csv") # rank of the right document per query

# ---------- Summary evaluation ----------
ROUGE_SAMPLE = 100                      # single articles scored with ROUGE against their human highlights
HUMAN_EVAL_SIZE = 20                    # query summaries for the fact-check and human ratings
ROUGE_FILE = os.path.join(RESULTS_DIR, "rouge_scores.csv")                 # ROUGE per article
ROUGE_SUMMARY_FILE = os.path.join(RESULTS_DIR, "rouge_summary.csv")        # average ROUGE per system
HUMAN_EVAL_FILE = os.path.join(RESULTS_DIR, "human_eval.csv")              # query summaries + ratings to fill in