# evaluate_summary.py
# Step 6 of the project: measure summary quality in three ways.
#   A. ROUGE (automatic): 100 single articles. The LLM summary and a simple baseline (Lead-3 = the
#      article's first 3 sentences) are both compared with the human-written highlights.
#      ROUGE-1 = shared words, ROUGE-2 = shared word pairs, ROUGE-Lsum = longest shared word sequence
#      per sentence. Scores are F1 (0 to 1, higher = more overlap with the human summary).
#   B. Query summaries (what users actually see): 20 test questions -> search -> medium summary.
#      Each summary is checked automatically:
#        - citations: does every [n] point to a real source (1 to 5)?
#        - LLM fact-check: a second LLM call lists claims not supported by the sources (score 1-5).
#   C. Human evaluation: the 20 summaries are saved to results/human_eval.csv with empty columns
#      relevance, coherence, faithfulness (1-5). You fill them in, then run this file again
#      to get the averages. Existing ratings are never overwritten.
#   D. Fact-checker test: one number in each summary is deliberately changed, and the fact-check
#      must flag it. This shows whether the automatic fact-check scores can be trusted.
# Run:  python evaluate_summary.py

import os                     # check files
import re                     # sentences and citations
import json                   # read the fact-check answer
import pandas as pd           # tables
from rouge_score import rouge_scorer   # standard ROUGE implementation
import config                 # project settings
from summarize import call_llm, summarize_article, summarize_results, document_for_llm, target_words
from search import load_engine, search, get_article   # search engine

SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)   # "arrested" = "arrest"


# Splits text into sentences at ". ", "! " or "? " followed by a capital letter or quote.
# It does not split after common abbreviations ("Mr. Smith", "Dr. Jones", "U.S. President").
SENTENCE_END = re.compile(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!U\.S\.)"
                          r"(?<=[.!?])\s+(?=[A-Z\"'])")   # end of sentence, but not after an abbreviation


def split_sentences(text):
    return [s for s in SENTENCE_END.split(text.strip()) if s]   # drop empty pieces


# Lead-3 baseline: the first three sentences of the article.
# News puts the key facts first, so this simple rule is a strong, well-known baseline.
def lead3(text):
    return " ".join(split_sentences(text)[:3])            # first 3 sentences joined back together


# ROUGE F1 scores between a summary and the reference highlights.
# ROUGE-Lsum expects one sentence per line, so both texts are split into lines first.
def rouge(summary, reference):
    pred = "\n".join(split_sentences(summary))            # one sentence per line
    ref = "\n".join(split_sentences(reference))           # one sentence per line
    scores = SCORER.score(ref, pred)                      # compare
    return {name: s.fmeasure for name, s in scores.items()}   # keep the F1 part of each


# Part A: ROUGE for the LLM and for Lead-3 on 100 random articles.
def run_rouge(articles):
    sample = articles.sample(n=config.ROUGE_SAMPLE, random_state=config.RANDOM_SEED + 1)   # different articles from the search test
    rows = []                                             # one row per (article, system)
    for i, a in enumerate(sample.itertuples(), start=1):  # each article
        llm_summary, model = summarize_article(a.text, "short")   # Gemini summary (~50 words)
        for system, summary in [("llm", llm_summary), ("lead3", lead3(a.text))]:   # both systems
            rows.append({"doc_id": a.doc_id, "system": system, "model": model if system == "llm" else "",
                         "words": len(summary.split()),    # summary length
                         "reference_words": len(a.highlights.split()),   # human summary length
                         **rouge(summary, a.highlights)})  # rouge1, rouge2, rougeLsum
        if i % 20 == 0:                                   # progress every 20 articles
            print(f"  ROUGE: {i}/{len(sample)} articles done")
    scores = pd.DataFrame(rows)                           # all rows
    scores.to_csv(config.ROUGE_FILE, index=False)         # save per-article scores
    summary = scores.groupby("system")[["rouge1", "rouge2", "rougeLsum", "words", "reference_words"]].mean()   # averages
    summary.to_csv(config.ROUGE_SUMMARY_FILE)             # save the table
    print("\n----- ROUGE (F1, average over", len(sample), "articles) -----")
    print(summary.round(3).to_string())                   # print the table
    print("Models used for LLM summaries:", scores.loc[scores.system == "llm", "model"].value_counts().to_dict())


# Finds citation numbers like [2] or [2][3] or [2, 3] and checks they point to real sources.
def check_citations(summary, n_sources):
    numbers = []                                          # every cited number
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", summary):   # "[2]" -> "2", "[2, 3]" -> "2, 3"
        numbers += [int(n) for n in group.split(",")]     # split "2, 3" into 2 and 3
    invalid = sum(1 for n in numbers if n < 1 or n > n_sources)   # numbers with no matching source
    return len(numbers), invalid


# Asks the LLM to fact-check a summary against its sources and return a 1-5 score and unsupported claims.
def fact_check(sources, summary):
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))   # numbered sources
    prompt = ("You are a strict fact-checker. Compare the SUMMARY with the SOURCES.\n"
              "List every claim in the summary that is not supported by the sources "
              "(wrong facts, wrong numbers, wrong source number, or added information).\n"
              "Then give a faithfulness score: 5 = everything supported, 4 = one minor issue, "
              "3 = a few issues, 2 = several clear errors, 1 = mostly unsupported.\n"
              'Return ONLY JSON: {"faithfulness": 5, "unsupported_claims": ["..."]}\n\n'
              f"SOURCES:\n{numbered}\n\nSUMMARY:\n{summary}")
    answer, model = call_llm(prompt)                      # one LLM call
    match = re.search(r"\{.*\}", answer, re.DOTALL)       # the JSON object inside the answer
    try:
        result = json.loads(match.group(0))               # parse it
        return int(result.get("faithfulness", 0)), list(result.get("unsupported_claims", []))
    except (AttributeError, ValueError, TypeError):       # no JSON or broken JSON
        return None, ["(fact-check answer could not be read)"]


# Part B: query summaries with automatic checks, saved as the human evaluation sheet.
def create_human_eval_sheet(engine):
    queries = pd.read_csv(config.TEST_QUERIES_FILE)       # test queries from evaluate_search.py
    questions = queries[queries.query_type == "question"] # natural questions read like real user requests
    sample = questions.sample(n=config.HUMAN_EVAL_SIZE, random_state=config.RANDOM_SEED)   # 20 of them
    rows = []                                             # one row per query
    for i, q in enumerate(sample.itertuples(), start=1):  # each query
        results, _ = search(engine, q.query)              # top 5 documents (hybrid_rerank)
        summary, model, seconds = summarize_results(engine, q.query, results, "medium")   # ~120 words
        sources = [document_for_llm(get_article(engine, r["doc_id"]), r["excerpt"], r["chunk_id"])
                   for r in results]                      # exactly what the summarizer saw
        citations, invalid = check_citations(summary, len(results))   # citation check
        score, claims = fact_check(sources, summary)      # LLM fact-check
        rows.append({"query": q.query,                    # the question
                     "summary": summary,                  # the summary to rate
                     "sources": ", ".join(r["doc_id"] for r in results),   # which articles were used
                     "source_was_found": q.doc_id in [r["doc_id"] for r in results],   # did search find the test article?
                     "model": model,                      # which model wrote it
                     "words": len(summary.split()),       # length (target 120)
                     "citations": citations,              # number of [n] citations
                     "invalid_citations": invalid,        # citations with no matching source
                     "llm_faithfulness": score,           # fact-check score 1-5
                     "unsupported_claims": " | ".join(claims),   # what the fact-check flagged
                     "relevance": "", "coherence": "", "faithfulness": "",   # YOU fill these in (1-5)
                     "comments": ""})                     # optional notes
        print(f"  Query summary {i}/{len(sample)} done")
    sheet = pd.DataFrame(rows)                            # all rows
    sheet.to_csv(config.HUMAN_EVAL_FILE, index=False)     # save the sheet
    print_query_summary_report(sheet)                     # automatic results
    print(f"\nNow open {config.HUMAN_EVAL_FILE}, read each summary and its sources, "
          f"fill relevance / coherence / faithfulness (1-5), save, and run this file again.")


# Prints the automatic checks for the query summaries.
def print_query_summary_report(sheet):
    target = target_words("medium")                       # 120
    print("\n----- Query summaries (automatic checks) -----")
    print(f"Summaries                       : {len(sheet)}")
    print(f"Avg words (target {target})          : {sheet['words'].mean():.1f}")
    print(f"Within +/-20% of target length  : {((sheet['words'] - target).abs() <= 0.2 * target).mean() * 100:.0f}%")
    print(f"Summaries with citations        : {(sheet['citations'] > 0).mean() * 100:.0f}%")
    print(f"Invalid citations (total)       : {int(sheet['invalid_citations'].sum())}")
    print(f"Avg LLM faithfulness (1-5)      : {pd.to_numeric(sheet['llm_faithfulness'], errors='coerce').mean():.2f}")
    print(f"Summaries flagged by fact-check : {(pd.to_numeric(sheet['llm_faithfulness'], errors='coerce') < 5).sum()}")


# Changes the first number in a summary that is not a citation, e.g. "150 people" -> "303 people".
# Returns the changed summary, or None if the summary has no such number.
def plant_error(summary):
    match = re.search(r"(?<![\[\d,])\b(\d+)\b(?![\]\d])", summary)   # a number not inside [ ]
    if not match:                                         # no number to change
        return None
    wrong = str(int(match.group(1)) * 2 + 3)              # always a different number
    return summary[:match.start(1)] + wrong + summary[match.end(1):]   # swap it in


# Part D: tests the fact-checker itself. Each summary gets one deliberately wrong number;
# a working fact-checker should flag it (score below 5). The catch rate shows whether the
# fact-check scores can be trusted.
def planted_error_test(engine, sheet):
    caught, tested = 0, 0                                 # counters
    for row in sheet.itertuples():                        # each query summary
        broken = plant_error(row.summary)                 # summary with one wrong number
        if broken is None:                                # no number in this summary
            continue                                      # skip it
        results, _ = search(engine, row.query)            # same sources as before (search is repeatable)
        sources = [document_for_llm(get_article(engine, r["doc_id"]), r["excerpt"], r["chunk_id"])
                   for r in results]                      # exactly what the summarizer saw
        score, _ = fact_check(sources, broken)            # fact-check the broken summary
        tested += 1                                       # one more test
        if score is not None and score < 5:               # the error was noticed
            caught += 1                                   # count it
    print("\n----- Fact-checker test (one planted wrong number per summary) -----")
    if tested:                                            # at least one summary had a number
        print(f"Errors caught: {caught}/{tested} ({caught / tested * 100:.0f}%)")
    else:
        print("No summaries contained numbers to change.")


# Part C: reads your ratings from the sheet and prints the averages.
def report_human_ratings():
    sheet = pd.read_csv(config.HUMAN_EVAL_FILE)           # the sheet you filled in
    print_query_summary_report(sheet)                     # automatic checks again, for reference
    print("\n----- Human evaluation (1-5) -----")
    for column in ["relevance", "coherence", "faithfulness"]:   # each rating
        values = pd.to_numeric(sheet[column], errors="coerce")   # blank -> NaN
        print(f"{column:<12}: {values.mean():.2f}  ({values.notna().sum()}/{len(sheet)} rated)")
    human = pd.to_numeric(sheet["faithfulness"], errors="coerce")   # your faithfulness scores
    llm = pd.to_numeric(sheet["llm_faithfulness"], errors="coerce") # the fact-check's scores
    both = human.notna() & llm.notna()                    # rows with both scores
    if both.sum() >= 5:                                   # enough rows to compare
        agree = ((human[both] - llm[both]).abs() <= 1).mean() * 100   # within 1 point of each other
        print(f"Human vs LLM fact-check: agree within 1 point on {agree:.0f}% of {both.sum()} summaries")


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)        # create results/ if missing
    print("Loading search engine...")
    engine = load_engine()                                # load once
    print("Part A: ROUGE on single articles...")
    run_rouge(engine["articles"].reset_index())           # cached LLM answers make reruns free
    if os.path.exists(config.HUMAN_EVAL_FILE):            # sheet already created: never overwrite your ratings
        print("\nPart B/C: using the existing human evaluation sheet")
        report_human_ratings()                            # print averages
    else:
        print("\nPart B: query summaries + automatic checks...")
        create_human_eval_sheet(engine)                   # create the sheet
    print("\nPart D: testing the fact-checker with planted errors...")
    planted_error_test(engine, pd.read_csv(config.HUMAN_EVAL_FILE))   # can the fact-check catch a wrong number?


if __name__ == "__main__":
    main()