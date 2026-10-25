# Document Search and Summarization Using LLMs: Report

**Project:** Hybrid Search Summarizer, a system that searches 9,986 news articles with a user query,
returns the most relevant documents, and summarizes them with an LLM in a length the user chooses.

**Summary of results**

- Hybrid search with reranking puts the correct article first for **85.4%** of keyword queries
  and **72.9%** of natural-language questions (top 10: 99.5% and 88.9%).
- Combining BM25 with embeddings improves question search by **+16 points** over either method
  alone; the difference is statistically significant (p < 0.0001).
- Gemini summaries match the Lead-3 baseline on ROUGE-1 (41.0 vs 40.5) while being 37% shorter.
- Human ratings of 20 query summaries: relevance **4.80**, coherence **4.70**, faithfulness
  **4.55** out of 5. Every summary cites its sources, with 0 invalid citations.

---

## 1. Data preparation

### 1.1 Corpus choice

| Option considered | Decision |
|---|---|
| **CNN/DailyMail 3.0.0** (news, human-written highlights) | **Chosen** |
| BillSum (US bills with summaries) | Strong alternative; legal text, longer documents |
| XL-Sum Tamil, recent Tamil Nadu news | Rejected: no reference summaries at scale (ROUGE impossible), Tamil breaks standard ROUGE and BM25 tokenization |
| Wikipedia and similar | Rejected: no reference summaries |

The deciding requirement was **ROUGE**, which needs a human-written reference summary for each
document. CNN/DailyMail provides one for every article, and its published baseline scores make it
possible to verify the evaluation setup (see 4.2).

I used **10,000 random articles from the test split** (seed 42). Nothing in this project is
trained, so the split does not matter; the test split is the smallest download. The articles
date from 2007 to 2015.

### 1.2 Cleaning steps

| Step | Why | Effect |
|---|---|---|
| Remove CNN datelines, e.g. `LONDON, England (CNN) --` | Boilerplate words that match many queries in BM25 | |
| Remove Daily Mail headers, e.g. `By . Name . PUBLISHED: ... UPDATED: ...` | Bylines and timestamps are noise for search and waste LLM tokens | |
| Remove `Last updated at ...` and `Scroll down for video` | Same | |
| **Total of the three steps above** | | **2,921 articles changed, 13,912 words removed** |
| Normalize whitespace | Makes the patterns and the word counts reliable | |
| Drop articles under 100 words | Too short to summarize meaningfully | 12 removed |
| Drop exact duplicates (MD5 hash of cleaned text) | Duplicates would make search evaluation ambiguous | 2 removed |
| Highlights: join lines into sentences, fix spaces before full stops | Clean reference summaries for ROUGE | |

**Result:** 9,986 articles, 681.7 words on average.

Every rule was tested on sample texts in the real formats before running. The first full run
exposed two problems that were then fixed:
- a "Scroll down for video" phrase that slipped through
- a misleading report comparing averages over different article sets

**Known limitation:** Daily Mail photo captions ("Baby-faced: This little boy...") are mixed into
the article text without any reliable marker, so removing them would also delete real sentences.
They were left in.

### 1.3 Chunking

Articles are split into **200-word chunks with a 50-word overlap**. A final chunk that would add
fewer than 50 new words is merged into the previous one.

**Result:** 43,794 chunks, 4.39 per article, 194 words on average.

Why chunk at all:
1. **Long articles bury the relevant part.** A 1,300-word article about a trial might answer the
   query in one paragraph; scoring the whole article dilutes that signal.
2. **The embedding model reads at most 256 tokens.** A 200-word chunk (~250 tokens) fits almost
   entirely, while a full article would be cut after its first ~190 words.
3. **The overlap covers the cut.** A sentence that crosses a chunk boundary, or a tail the
   embedding model truncates, also appears at the start of the next chunk.

Search runs on chunks; results are returned per document (section 2.5).

---

## 2. Document search

Four methods were built so that each component's contribution could be measured (section 4.1).

```
query ─┬─ BM25 (keywords) ── top 100 chunks ─┐
       │                                     ├─ Reciprocal Rank Fusion ─ cross-encoder rerank (top 30) ─ chunks → documents
       └─ embeddings (meaning) ─ top 100 ────┘
```

### 2.1 BM25 (traditional retrieval)

- **Tokenization:** text is lowercased, only letters and digits are kept, and ~130 English
  stopwords are removed. The same function is used for indexing and searching.
- **Library:** `bm25s` (k1 = 1.5, b = 0.75). It was chosen over `rank_bm25` after measuring both
  on 43,794 chunks:

| Library | Query time | Build time |
|---|---|---|
| rank_bm25 | 93 ms | 6 s |
| **bm25s** | **4 ms** | 11 s |

  `bm25s` pre-computes scores into a sparse matrix, so query time grows far more slowly with
  corpus size.
- Chunks that share no keyword with the query are dropped.

### 2.2 Embeddings (semantic retrieval)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.
- **Storage:** vectors are normalized, so the dot product equals cosine similarity, and stored
  as float16 (33.6 MB instead of 67 MB; the precision loss is negligible for ranking).
- **Search:** FAISS `IndexFlatIP`, an exact search, which is fast enough at this size (section 6).

**On "LLM embeddings":** I used a local transformer embedding model rather than a hosted LLM
embedding API, for three reasons:
1. No API cost or rate limit for 43,794 chunks
2. No network dependency at query time
3. Results are repeatable

The model name lives in `config.py`, so switching to an API embedding model only requires a
different encoder in `build_index.py` and `search.py`.

### 2.3 Hybrid: Reciprocal Rank Fusion

Each method returns its top 100 chunks. Every chunk receives `1 / (60 + rank)` from each list it
appears in, and the scores are added.

Example: rank 1 in BM25 and rank 3 in embeddings gives 1/61 + 1/63 = 0.0323.

RRF uses **ranks only**, so BM25 scores (unbounded) and cosine similarities (−1 to 1) never need to
be put on a common scale. That is the main reason for choosing it over a weighted score sum, which
needs a tuned weight.

### 2.4 Reranking

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the query and each chunk
*together* and re-scores the top 30 fused chunks.

This is more accurate than comparing two separate vectors, but it needs one model pass per chunk,
so it is applied only to the top 30.

### 2.5 From chunks to documents

The ranked chunk list is converted into a document list. Each document is ranked by its
**best-scoring chunk**, and that chunk becomes the excerpt shown to the user. The top N documents
are returned (N = 5 by default, 18 in the app for pagination).

---

## 3. Summarization

### 3.1 Model

**`gemini-3.5-flash-lite`**, with `gemini-3.8-flash` and `gemini-3.7-flash` as backups.

The first choice was `gemini-3.8-flash`, but it returned 503 "high demand" on almost every call
during development. Flash-lite:
- answered in 2–5 seconds
- was consistently available
- matches the brief's advice to use resource-efficient LLMs

Using one model for all evaluation runs also keeps the results comparable. The cache records which
model wrote each answer: all 100 ROUGE summaries and all 20 query summaries came from flash-lite.

### 3.2 What the model receives

For each of the top N documents, the model receives:
- **the article's first 400 words**, since news puts the key facts first, and
- **the matching chunk**, if it lies later in the article.

That is about 2,000 words for 5 documents, which keeps cost and latency low without hiding the
part of the article that matched the query.

### 3.3 Prompt rules

1. Use only facts stated in the articles, with no outside knowledge.
2. Cite each fact with its source number, e.g. [1] or [2][3].
3. Focus on the articles that answer the query; mention unrelated ones in at most one short sentence.
4. Keep different events separate instead of merging them into one story.
5. Write like a news report, without referring to "the articles".
6. About *N* words, where *N* is 50, 120 or 250 (short / medium / long) or any number.

Rules 3 and 5 were added after reviewing early outputs (section 7).

### 3.4 Reliability

- **Cache:** every LLM answer is stored by prompt hash in `data/llm_cache.json`. Repeat runs are
  free and give identical results.
- **Fallback:** on a busy (503) or rate-limit (429) error, the next model is tried immediately.
  The model that answers is remembered for the rest of the run, and the code only waits
  (10 s, 20 s, 30 s) when every model is busy. Errors that cannot be fixed by retrying (wrong
  key, unknown model) stop at once.

---

## 4. Evaluation

### 4.1 Search accuracy

**Test set**
- 200 random articles; Gemini wrote **two queries per article**, 10 articles per call
  (20 calls in total):
  - a **keyword query** ("jenson button bahrain grand prix practice trouble")
  - a **natural question** ("Why did the British Formula 1 driver struggle during the practice
    sessions in Bahrain?")
- The prompt forbade copying phrases of 3 or more words, so BM25 cannot win simply by matching
  copied text.
- Two batches returned unreadable JSON at first. The code was changed to keep the readable entries
  and re-ask for missing articles. **199 articles (398 queries)** were kept; one article never
  received valid queries after 3 attempts.

**Metrics:** each query has exactly one correct article.
- **Hit@k:** share of queries whose article is in the top k results.
- **MRR:** mean of 1/rank, with 0 when the article is not found.

| Method | Query type | Hit@1 | Hit@5 | Hit@10 | MRR | Median time |
|---|---|---|---|---|---|---|
| BM25 | keyword | 78.9% | 96.0% | 99.0% | 0.865 | <1 ms |
| BM25 | question | 47.7% | 75.9% | 80.9% | 0.601 | <1 ms |
| Embeddings | keyword | 72.9% | 93.0% | 97.0% | 0.814 | 14 ms |
| Embeddings | question | 48.7% | 73.9% | 83.4% | 0.600 | 14 ms |
| Hybrid | keyword | 76.4% | 96.5% | 99.0% | 0.852 | 16 ms |
| Hybrid | question | 64.3% | 80.4% | 85.9% | 0.717 | 16 ms |
| **Hybrid + rerank** | keyword | **85.4%** | **97.5%** | **99.5%** | **0.905** | 1,084 ms |
| **Hybrid + rerank** | question | **72.9%** | **88.4%** | **88.9%** | **0.797** | 1,084 ms |

**Which differences are real?** Every method answered the same queries, so the pairs were compared
with a paired sign test (McNemar) on Hit@1. "Only A" counts the queries that A got right and B got
wrong.

| Comparison | Query type | Only A | Only B | p-value | Real difference? |
|---|---|---|---|---|---|
| BM25 vs embeddings | question | 28 | 30 | 0.90 | No |
| BM25 vs hybrid | question | 9 | 42 | <0.0001 | **Yes** |
| Embeddings vs hybrid | question | 7 | 38 | <0.0001 | **Yes** |
| Hybrid vs hybrid + rerank | question | 10 | 27 | 0.008 | **Yes** |
| BM25 vs embeddings | keyword | 22 | 10 | 0.05 | Borderline |
| BM25 vs hybrid | keyword | 14 | 9 | 0.40 | No |
| Hybrid vs hybrid + rerank | keyword | 3 | 21 | 0.0003 | **Yes** |

**Findings**

1. **BM25 and embeddings fail on different questions, which is why fusion works.** On questions
   they are tied (47.7% vs 48.7%) but disagree often: 28 questions only BM25 answers, 30 only
   embeddings answers. Hybrid answers 128, more than the 125 answered by either method alone,
   because an article ranked 2nd by both methods moves up to 1st when the lists are combined.
2. **On keyword queries, hybrid adds nothing measurable over BM25** (p = 0.40). Keywords
   already contain the names BM25 matches exactly.
3. **The reranker gives a real gain on both query types** (+9.0 points on keywords, +8.5 on
   questions), at about 67 times the search time.
4. **My hypothesis was wrong.** I expected embeddings to clearly beat BM25 on natural questions.
   They tied. The generated questions remove the names ("the British Formula 1 driver" instead of
   Jenson Button), which hurts both methods equally.
5. **The app therefore defaults to hybrid + rerank** (the brief ranks accuracy highest) and offers
   a "Fast" mode using plain hybrid.

**Failure analysis.** With hybrid + rerank, 22 of 199 questions did not return the correct article
in the top 10. Reading them shows three patterns:

| Pattern | Example | Why it fails |
|---|---|---|
| The question has no distinguishing names | "Which football team secured an important victory thanks to a spectacular long-range strike?" | Matches dozens of football reports equally well |
| The corpus has another article on the same event | "Who has been selected to officiate the upcoming mega-fight...?" | Search returned a *different* article about the same referee appointment; the summary answered correctly (human rating 5/5/5) |
| A similar but different story wins | "...teenager who verbally abused a transit worker on video" | Returned a different train-abuse video story |

The second pattern means **Hit@k slightly understates real quality**. For 2 of the 3 human-rated
queries where the exact source was missed, the summary still answered correctly from another article
on the same event.

### 4.2 Summary quality: ROUGE

**Setup**
- 100 random articles (different from the search test set).
- Gemini writes a **short (~50-word)** summary of each, which matches the length of the human
  highlights (50 words on average).
- Both Gemini and the **Lead-3 baseline** (the article's first three sentences) are compared with
  the human highlights.
- ROUGE uses `rouge-score` with stemming, F1, and ROUGE-Lsum computed per sentence.

| System | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Words |
|---|---|---|---|---|
| Lead-3 (this project) | 40.5 | 17.8 | 36.7 | 79 |
| Lead-3 (published for this dataset) | 40.3 | 17.7 | 36.6 | |
| **Gemini 3.5 flash-lite** | **41.0** | **14.7** | **35.5** | **50** |

**The setup is verified:** our Lead-3 reproduces the published Lead-3 scores almost exactly. That
confirms the cleaning, sentence splitting and ROUGE configuration are all correct.

**Comparing Gemini with Lead-3** (per-article differences, bootstrap 95% confidence intervals and
Wilcoxon tests):

| Metric | Gemini minus Lead-3 | 95% CI | Gemini better on | Significant? |
|---|---|---|---|---|
| ROUGE-1 | +0.5 | −1.4 to +2.5 | 53 / 100 | No |
| ROUGE-2 | −3.1 | −5.0 to −1.3 | 36 / 100 | **Yes** (p = 0.001) |
| ROUGE-Lsum | −1.2 | −3.2 to +0.8 | 46 / 100 | No |

**Interpretation**
- Gemini is **comparable to Lead-3 on ROUGE-1 and ROUGE-L while using 37% fewer words**. Lead-3's
  extra length inflates its recall.
- **Lead-3 is higher on ROUGE-2 (word pairs)** because it copies the article's exact phrasing,
  while the LLM rewrites in its own words.
- This is ROUGE's known blind spot: it measures word overlap, not meaning or correctness. That
  blind spot is why the next checks exist.

### 4.3 Summary quality: query summaries

20 test questions were each searched (hybrid + rerank, top 5) and summarized at medium length.

| Automatic check | Result |
|---|---|
| Average length (target 120 words) | 125 words; 85% within ±20% |
| Summaries containing citations | 100% |
| Citations pointing to a non-existent source | 0 |
| LLM fact-check score (1-5) | 4.95; 1 summary flagged |

**Validating the fact-checker.** A fact-check is only useful if it can catch errors. The first
version gave every summary 5/5, which could mean "no errors" or "doesn't notice errors".

To find out, **one number in each summary was deliberately changed** (e.g. "150 people" →
"303 people") and the fact-checker was run again. It caught **15 of 15** planted errors.

So its scores are credible for factual slips such as wrong numbers. That is the only kind of error
tested, and the claim is limited to it.

### 4.4 Summary quality: human evaluation

I rated all 20 query summaries on a 1-5 scale, using a written rubric and the source articles.
Each rating was based on checking 2-3 key facts against the cited source. The rating page hid the
LLM fact-check score so it could not influence the ratings.

| | Relevance | Coherence | Faithfulness |
|---|---|---|---|
| **All 20 summaries** | **4.80** | **4.70** | **4.55** |
| Excluding the one search failure (below) | 5.00 | 4.89 | 4.74 |
| Score distribution, faithfulness | | | 14 × 5, 5 × 4, 1 × 1 |

- **One summary was rated 1/1/1.** It is a search failure: the correct article was not
  retrieved, and the summary stated that the results did not answer the question. That is the
  honest behaviour, but it did not help the user.
- **Human vs LLM fact-check:** they agreed within 1 point on 95% of summaries. Where they
  differed, the human score was usually lower: on 5 summaries I gave 4 where the LLM gave 5.
  The LLM checker is **slightly lenient**, likely because a model is grading text from its own
  model family. It catches clear factual errors reliably, but human review still finds subtler
  issues.
- **The fact-checker's one flag** (the Miami Open summary) was not confirmed by my review, so
  the automatic check also produces occasional false alarms.

---

## 5. Interface

"The Search Gazette" is a Streamlit app styled as a newspaper front page:

- **Query auto-suggestion:** while typing, the box suggests 2+ word names that appear in at
  least 3 articles ("Manchester United", "White House"), mined from the articles' opening
  paragraphs. Free text is also accepted.
- **Summary length:** short / medium / long, plus the number of documents to summarize (1-10).
- **Pagination:** 18 results, 6 per page, with Previous/Next. Results are kept between clicks,
  so changing page does not search again.
- **Fast / Accurate switch:** plain hybrid (~16 ms) or hybrid + rerank (~1.1 s).
- **Traceability:** the query becomes the headline. Citation numbers in the summary are red marks
  that match the numbered "clippings" below, each with a headline (the article's first sentence),
  the matching excerpt and the full article on demand.
- **Robustness:** if Gemini is unavailable, the app says so and still shows the search results.

A second page, `rate_app.py`, was built for the human evaluation (section 4.4).

---

## 6. Efficiency and scalability

**Measured on a laptop CPU, 43,794 chunks:**

| Step | Median | 95th percentile |
|---|---|---|
| BM25 | <1 ms | 6 ms |
| Embedding search | 14 ms | 16 ms |
| Hybrid | 16 ms | 18 ms |
| Hybrid + rerank | 1,084 ms | 1,310 ms |
| Summary (Gemini flash-lite, 5 documents) | 2-7 s | |
| Summary (cached) | <0.1 s | |

**Storage:**

| Item | Size |
|---|---|
| Embeddings (43,794 × 384 × 2 bytes) | 33.6 MB |
| Indexes | built once (step 2 in the README) |
| Search engine | loaded once and kept in memory by the app |

**Scaling to much larger corpora:**

| Part | At ~10k articles (now) | At 1M+ articles |
|---|---|---|
| BM25 | `bm25s` sparse matrix in memory | Same library scales to millions; beyond that, Elasticsearch / OpenSearch |
| Embeddings | Exact search (FAISS flat) | Approximate search (FAISS IVF or HNSW); the vectors alone are ~3.4 GB in float16 for 4.4M chunks, so product quantization or a vector database |
| Index building | CPU, 10-25 min | Batch encoding on a GPU; incremental indexing of new articles |
| Reranker | 30 chunks per query, ~1 s | Same fixed cost per query, independent of corpus size; a GPU brings it to ~50 ms |
| LLM | One call per query, cached | Fixed number of documents per summary, so cost does not grow with the corpus; cache shared across users |

The key property is that **query cost grows slowly with corpus size**. Only the first-stage
retrieval (BM25, vector search) touches the whole corpus, and both have standard sub-linear
solutions. The reranker and LLM always see a fixed number of items.

---

## 7. Challenges and solutions

| Challenge | Solution |
|---|---|
| Dataset boilerplate (bylines, timestamps, "Scroll down for video") | Regex cleaning, tested on real formats; the run report showed 2,921 articles cleaned and caught a missed pattern |
| Hugging Face was unreachable from the development sandbox | All logic was tested with stand-in models and synthetic data before running on the real data |
| Gemini model unavailable (`gemini-2.5-flash` retired for new users) | Added `list_models.py`; the model is set in `config.py` |
| Gemini overload (503) and rate limits (429) | Instant switch to backup models, remembering the working one; wait only when all are busy. The first version waited ~130 s per call; the fix brought this to near zero |
| Wrong API key retried for 60 s | Errors that retrying cannot fix (400/401/403/404) now stop immediately |
| LLM returned broken JSON for 2 of 20 query batches | Parse entry by entry, then re-ask only for the missing articles with a changed prompt so the cache does not return the same broken answer. Recovered 19 of 20 articles |
| BM25 could win by matching copied phrases | Test queries must not copy 3+ word phrases; questions also remove names |
| Summaries opened with "Based on the provided articles..." and described every unrelated search result | Prompt rules: write like a news report; focus on relevant articles, unrelated ones in one sentence at most. Length accuracy improved from 65% to 85% within ±20% of the target |
| LLM fact-check gave every summary 5/5 | Validated it with planted errors (15/15 caught) and compared it with human ratings |
| Streamlit's file watcher crashed on an unused `transformers` module (torchvision) | Disabled the watcher in `.streamlit/config.toml` |

---

## 8. Limitations and future work

**Limitations**

- **No fine-tuning.** The brief suggests it as a hint. I did not fine-tune the LLM: there is no
  local GPU, the Gemini free tier does not offer fine-tuning, and the summaries already reach
  4.7+/5 on human relevance and coherence. The more promising target would be the **embedding
  model**: fine-tuning MiniLM on query–article pairs (generated the same way as the test set,
  from articles outside it) is a ~20-minute Colab job and could narrow the gap on name-free
  questions.
- **Prompt compliance is partial.** 11 of 20 summaries still use words like "provided articles"
  despite the rule, and 18 of 20 add one sentence about unrelated results. A better fix is
  upstream: drop results whose reranker score is below a threshold before summarizing, so
  unrelated articles never reach the LLM. That also saves tokens.
- **Evaluation scale.** Human evaluation has 20 summaries and one rater (the author). Test queries
  are LLM-generated, and the questions are harder than typical user queries because the names are
  removed.
- **Single-article references.** ROUGE can only be computed for single-article summaries, because
  the dataset has no references for multi-document, query-focused summaries. Those were evaluated
  by the automatic checks and the human ratings instead.
- **Old data.** The news is from 2007-2015. The LLM may have seen these public articles in
  training, which could slightly flatter the ROUGE scores.
- **Photo captions** from Daily Mail articles remain in the text (section 1.2).

**Future work**

- Reranker score threshold before summarization
- Fine-tuned embedding model
- A larger human evaluation with a second rater and agreement statistics
- An English Tamil Nadu news corpus as a generalization test