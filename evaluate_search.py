# evaluate_search.py
# Step 5 of the project: measure how accurate each search method is.
#   1. Pick 200 random articles as the test set.
#   2. Ask Gemini to write two search queries for each article:
#        - a keyword query  (how people type into a search box)  e.g. "germanwings copilot crash alps"
#        - a natural question                                     e.g. "why did the plane crash in the french alps?"
#      The queries must use the model's own words, not copied sentences, so BM25 cannot win
#      just by matching copied phrases.
#   3. Run every query through all 4 search methods and check where the source article appears.
#   4. Report Hit@1, Hit@5, Hit@10, MRR and average search time for every method and query type.
# Metrics (each query has exactly one correct article):
#   Hit@k = share of queries whose correct article is in the top k results.
#   MRR   = average of 1/rank of the correct article (rank 1 -> 1.0, rank 2 -> 0.5, not found -> 0).
# Run:  python evaluate_search.py

import os                     # folders
import re                     # find {...} entries in a broken answer
import json                   # read the LLM's JSON answer
import pandas as pd           # tables
import config                 # project settings
from summarize import call_llm    # cached Gemini call with fallback models
from search import load_engine, search, METHODS   # the search engine


# Picks the test articles at random (fixed seed, so the same articles every time).
def pick_test_articles(articles):
    return articles.sample(n=config.TEST_SIZE, random_state=config.RANDOM_SEED)   # 200 random rows


# Builds the prompt that asks for two queries per article, for one batch of articles.
# On a retry, an extra line asks for strictly valid JSON (this also makes the prompt different,
# so the cache does not return the same broken answer again).
def build_query_prompt(batch, retry=False):
    parts = []                                            # one text block per article
    for doc_id, text in zip(batch["doc_id"], batch["text"]):   # each article in the batch
        opening = " ".join(text.split()[:config.QUERY_SOURCE_WORDS])   # first 250 words (main facts)
        parts.append(f"ID: {doc_id}\n{opening}")          # label it with its id
    articles_text = "\n\n---\n\n".join(parts)             # separate articles clearly
    return ("You are creating a test set for a news search engine.\n"
            "For EACH article below, write two search queries that a real user might type to find it:\n"
            "1. keyword_query: 3 to 6 keywords, lowercase, no full sentence.\n"
            "2. question: a natural question of 8 to 15 words.\n"
            "Rules:\n"
            "- Use your own words. Do not copy phrases of 3 or more words from the article.\n"
            "- You may use important names (people, places, organisations).\n"
            "- Each query should point to this specific article, not to the topic in general.\n"
            "Return ONLY a JSON list, one object per article, in this exact format:\n"
            '[{"id": "doc_00001", "keyword_query": "...", "question": "..."}]\n'
            + ("Your previous answer was not valid JSON. Do not use double quote marks inside the "
               "query text, and make sure every article has both fields.\n" if retry else "")
            + f"\nArticles:\n\n{articles_text}")


# Pulls the article entries out of the LLM answer.
# First tries to read the whole JSON list. If one entry is broken (e.g. a stray quote mark),
# the whole list fails, so it then reads each {...} entry on its own and keeps the good ones.
def parse_json_list(answer):
    start = answer.find("[")                              # first "["
    end = answer.rfind("]")                               # last "]"
    if start != -1 and end != -1:                         # a list is present
        try:
            items = json.loads(answer[start:end + 1])     # parse the whole list
            if isinstance(items, list):                   # got a real list
                return [i for i in items if isinstance(i, dict)]   # keep only entry objects
        except json.JSONDecodeError:                      # broken somewhere
            pass                                          # fall through to entry-by-entry parsing
    items = []                                            # entries that parse on their own
    for piece in re.findall(r"\{[^{}]*\}", answer):       # every {...} block
        try:
            items.append(json.loads(piece))               # good entry
        except json.JSONDecodeError:                      # this entry is broken
            continue                                      # skip only this one
    return items


# Sends one batch of articles to Gemini and returns query rows for the articles it answered properly.
def ask_for_queries(batch, retry=False):
    batch_ids = set(batch["doc_id"])                      # only these ids are valid answers for this call
    answer, model = call_llm(build_query_prompt(batch, retry))   # one Gemini call
    rows, found = [], set()                               # output rows, ids answered
    for item in parse_json_list(answer):                  # each article's entry
        doc_id = str(item.get("id", ""))                  # which article
        keyword = str(item.get("keyword_query", "")).strip()   # keyword query
        question = str(item.get("question", "")).strip()  # natural question
        if doc_id not in batch_ids or doc_id in found or not keyword or not question:   # bad or incomplete entry
            continue                                      # skip it (it will be asked again)
        found.add(doc_id)                                 # mark as done
        rows.append({"doc_id": doc_id, "query_type": "keyword_query", "query": keyword, "model": model})
        rows.append({"doc_id": doc_id, "query_type": "question", "query": question, "model": model})
    return rows, found


# Asks Gemini for queries in batches of 10 articles and returns one row per (article, query type).
# Articles that come back missing or broken are asked again, on their own, up to 2 more times.
def generate_queries(test_articles):
    rows, still_missing = [], []                          # output rows, ids that never got queries
    for start in range(0, len(test_articles), config.QUERY_BATCH):   # 0, 10, 20, ...
        batch = test_articles.iloc[start:start + config.QUERY_BATCH]  # next 10 articles
        batch_rows, found = ask_for_queries(batch)        # first try
        rows += batch_rows                                # keep what worked
        first_try = len(found)                            # for the progress line
        missing = batch[~batch["doc_id"].isin(found)]     # articles without queries after the first try
        group_size = len(missing)                         # how many go into follow-up 1
        if group_size > 0:                                # follow-up 1: all missing articles in one call
            retry_rows, retry_found = ask_for_queries(missing, retry=True)   # new prompt, so not the cached answer
            rows += retry_rows                            # keep what worked
            found |= retry_found                          # add to the done set
        missing = batch[~batch["doc_id"].isin(found)]     # still missing?
        if group_size > 1:                                # follow-up 2 (if follow-up 1 had only 1 article, it would repeat it)
            for i in range(len(missing)):                 # one article per call
                retry_rows, retry_found = ask_for_queries(missing.iloc[i:i + 1], retry=True)   # smallest prompt
                rows += retry_rows                        # keep what worked
                found |= retry_found                      # add to the done set
        still_missing += list(set(batch["doc_id"]) - found)   # give up on these
        print(f"  Batch {start // config.QUERY_BATCH + 1}: {first_try}/{len(batch)} first try, "
              f"{len(found)}/{len(batch)} after retries")
    print(f"Queries created: {len(rows)} for {len(rows) // 2} articles "
          f"(articles missing: {len(still_missing)})")    # summary
    return pd.DataFrame(rows)                             # one row per query


# Loads the saved test queries, or creates them the first time.
# Saving them keeps the test set fixed, so every method is always compared on the same queries.
def load_or_create_queries(articles):
    if os.path.exists(config.TEST_QUERIES_FILE):          # created in an earlier run
        print(f"Using saved test queries from {config.TEST_QUERIES_FILE}")
        return pd.read_csv(config.TEST_QUERIES_FILE)      # reuse them
    print("Creating test queries with the LLM...")        # first run
    queries = generate_queries(pick_test_articles(articles))   # ask Gemini
    queries.to_csv(config.TEST_QUERIES_FILE, index=False) # save for next time
    return queries


# Runs every query through one search method and records where the correct article appeared.
def run_method(engine, queries, method):
    rows = []                                             # one row per query
    for q in queries.itertuples():                        # each test query
        results, seconds = search(engine, q.query, top_n=config.EVAL_TOP_K, method=method)   # search
        found = [r["doc_id"] for r in results]            # returned article ids, best first
        rank = found.index(q.doc_id) + 1 if q.doc_id in found else None   # 1-based rank, or None if missed
        rows.append({"method": method, "query_type": q.query_type, "doc_id": q.doc_id,
                     "query": q.query, "rank": rank, "ms": seconds * 1000,
                     "top_result": found[0] if found else None})   # what came first (for failure analysis)
    return rows


# Turns per-query ranks into the metric table (one row per method and query type).
def compute_metrics(details):
    table = []                                            # output rows
    for (method, qtype), group in details.groupby(["method", "query_type"], sort=False):   # each combination
        ranks = group["rank"]                             # ranks (NaN = not found)
        table.append({"method": method,                   # search method
                      "query_type": qtype,                # keyword or question
                      "queries": len(group),              # how many queries
                      "hit@1": (ranks <= 1).mean(),       # correct article first
                      "hit@5": (ranks <= 5).mean(),       # in the top 5
                      "hit@10": (ranks <= 10).mean(),     # in the top 10
                      "mrr": (1 / ranks).fillna(0).mean(),   # average 1/rank, 0 when not found
                      "avg_ms": group["ms"].mean()})      # average search time
    return pd.DataFrame(table)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)        # create results/ if missing
    print("Loading search engine...")
    engine = load_engine()                                # load once
    queries = load_or_create_queries(engine["articles"].reset_index())   # fixed test set
    all_rows = []                                         # per-query results for every method
    for method in METHODS:                                # bm25, dense, hybrid, hybrid_rerank
        print(f"Evaluating {method} on {len(queries)} queries...")
        all_rows += run_method(engine, queries, method)   # run all queries
    details = pd.DataFrame(all_rows)                      # one row per (method, query)
    details.to_csv(config.SEARCH_DETAILS_FILE, index=False)   # save for failure analysis
    metrics = compute_metrics(details)                    # summary table
    metrics.to_csv(config.SEARCH_EVAL_FILE, index=False)  # save for the report
    print("\n----- Search evaluation -----")
    print(metrics.round(3).to_string(index=False))        # print the table
    print(f"\nSaved {config.SEARCH_EVAL_FILE} and {config.SEARCH_DETAILS_FILE}")


if __name__ == "__main__":
    main()