# Hybrid Search Summarizer

Ask a question about the news. The app finds the most relevant articles out of 9,986 and writes a
short summary of them, with every fact linked to the article it came from.

![The Search Gazette](docs/screenshot.png)

The full write-up (method, results, challenges) is in [REPORT.md](REPORT.md).

---

## How it works

1. **Data:** 10,000 CNN/DailyMail news articles (2007-2015) are cleaned (bylines, datelines and
   duplicates removed) and split into overlapping 200-word chunks.
2. **Search:** two searches run on every query and their results are combined:
   - **BM25** finds chunks that share the query's exact words.
   - **Embeddings** (all-MiniLM-L6-v2) find chunks with the same meaning in different words.
   - **Reciprocal Rank Fusion** merges the two ranked lists, and a **cross-encoder reranker**
     re-orders the top 30.
3. **Summary:** Gemini summarizes the top documents in the length you choose
   (short ~50, medium ~120, long ~250 words) and cites each fact as [1], [2], ...
4. **App:** a newspaper-style Streamlit page with query suggestions, pagination,
   summary length control and a fast/accurate search switch.

---

## Results

**Search accuracy:** 398 test queries (2 per article for 199 articles). Hit@1 = the correct article is the first result.

| Method | Hit@1 (keywords) | Hit@1 (questions) | Hit@10 (questions) | Time per query |
|---|---|---|---|---|
| BM25 only | 78.9% | 47.7% | 80.9% | ~1 ms |
| Embeddings only | 72.9% | 48.7% | 83.4% | ~15 ms |
| Hybrid | 76.4% | 64.3% | 85.9% | ~16 ms |
| **Hybrid + reranker** | **85.4%** | **72.9%** | **88.9%** | ~1.1 s |

Combining both searches adds 16 points on natural questions; the reranker adds about 9 more.

**Summary quality:**

| Check | Result |
|---|---|
| ROUGE-1 / 2 / L vs human summaries (100 articles) | 41.0 / 14.7 / 35.5 (Lead-3 baseline: 40.5 / 17.8 / 36.7) |
| Length (target 120 words) | 125 words on average, 85% within ±20% |
| Citations | 100% of summaries cite sources, 0 invalid |
| Automatic fact-check (caught 15/15 planted errors) | 4.95 / 5 |
| Human rating, 20 summaries (1-5) | relevance 4.80, coherence 4.70, faithfulness 4.55 |

---

## Setup

You need Python 3.10-3.12 and a free Gemini API key from https://aistudio.google.com/apikey.
No GPU needed.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a file called `.env` in the project folder with your key:

```
GEMINI_API_KEY=your_key_here
```

---

## Run

Run these once, in order. Each step saves its output for the next one.

```bash
python data_prep.py         # 1. download and clean the articles        (~1 min)
python build_index.py       # 2. build the search indexes               (10-25 min)
streamlit run app.py        # 3. open the app at http://localhost:8501
```

To reproduce the evaluation:

```bash
python evaluate_search.py   # search accuracy for all 4 methods         (~15 min)
python evaluate_summary.py  # ROUGE, citations, fact-check              (~5-10 min)
streamlit run rate_app.py   # rate the 20 summaries yourself (human evaluation)
python evaluate_summary.py  # run again to print the human averages
```

All results are saved in the `results/` folder.

---

## Files

| File | What it does |
|---|---|
| `config.py` | All settings in one place: paths, model names, sizes |
| `data_prep.py` | Downloads, cleans and chunks the articles |
| `text_utils.py` | Turns text into keywords for BM25 |
| `build_index.py` | Builds the BM25 index and the embeddings |
| `search.py` | The 4 search methods: bm25, dense, hybrid, hybrid_rerank |
| `summarize.py` | Gemini summaries with caching, retries and backup models |
| `evaluate_search.py` | Creates test queries and measures search accuracy |
| `evaluate_summary.py` | ROUGE, citation check, fact-check and human-rating averages |
| `app.py` | The newspaper-style web app |
| `rate_app.py` | Helper page for rating summaries by hand |
| `list_models.py` | Lists the Gemini models your API key can use |

---

## Good to know

- **Gemini model names change often.** If you get a "model not found" error, run
  `python list_models.py` and update `GEMINI_MODEL` in `config.py`.
- **Free-tier limits:** when Gemini is busy, the code switches to a backup model straight away
  and only waits if every model is busy. Every answer is saved in `data/llm_cache.json`,
  so re-running costs no API calls and gives the same results.
- **The news is old (2007-2015).** CNN/DailyMail is a fixed benchmark; it was chosen because it
  comes with human-written summaries, which ROUGE needs.