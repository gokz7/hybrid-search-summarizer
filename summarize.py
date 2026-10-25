# summarize.py
# Step 4 of the project: summarize the documents that search returned, using Gemini.
#   - summarize_results(): one combined summary of the top documents for a query (what the user sees).
#   - summarize_article(): a summary of one article with no query (used for ROUGE evaluation,
#                          because only single articles have human-written reference summaries).
# The user picks the length: "short" (~50 words), "medium" (~120) or "long" (~250).
# Every LLM answer is saved in data/llm_cache.json, so the same request never costs a second API call.
# Run:  python summarize.py   (needs GEMINI_API_KEY in a .env file)

import os                     # read the API key from the environment
import json                   # save and load the cache file
import time                   # timing and waiting between retries
import hashlib                # turn a prompt into a short cache key
from dotenv import load_dotenv    # load the .env file
import config                 # project settings

load_dotenv()                 # copies GEMINI_API_KEY from .env into the environment
_client = None                # Gemini client, created on first use
_cache = None                 # cache dictionary, loaded on first use
_working_model = None         # last model that answered in this run; tried first next time
_dead_models = set()          # models that said "not found / not allowed" in this run; never tried again


# Creates the Gemini client once and reuses it.
def get_client():
    global _client                                        # change the variable defined above
    if _client is None:                                   # first call only
        from google import genai                          # Google's Gemini SDK
        key = os.getenv("GEMINI_API_KEY")                 # API key from .env
        if not key:                                       # missing key is the most common setup mistake
            raise RuntimeError("GEMINI_API_KEY not found. Add it to a .env file in the project folder.")
        _client = genai.Client(api_key=key)               # connect
    return _client


# Loads the cache file once (an empty dictionary if it does not exist yet).
def load_cache():
    global _cache                                         # change the variable defined above
    if _cache is None:                                    # first call only
        if os.path.exists(config.LLM_CACHE_FILE):         # cache saved by an earlier run
            with open(config.LLM_CACHE_FILE, "r", encoding="utf-8") as f:   # open it
                _cache = json.load(f)                     # key -> answer
        else:
            _cache = {}                                   # nothing cached yet
    return _cache


# Saves the cache to disk after every new answer, so nothing is lost if the program stops.
def save_cache():
    os.makedirs(config.DATA_DIR, exist_ok=True)           # make sure data/ exists
    with open(config.LLM_CACHE_FILE, "w", encoding="utf-8") as f:   # overwrite the file
        json.dump(_cache, f, ensure_ascii=False)          # write all answers


# Sends the prompt to one model, once. Returns the answer text or raises the error.
def ask_model(model, prompt):
    from google.genai import types                        # request settings
    settings = types.GenerateContentConfig(temperature=config.TEMPERATURE)   # low temperature = factual
    response = get_client().models.generate_content(model=model,     # which model
                                                    contents=prompt, # the prompt
                                                    config=settings) # settings
    return (response.text or "").strip()                  # answer (empty if the model returned nothing)


# Returns the models to try, in order: the one that worked last time first,
# then the main model, then the backups, skipping models that do not exist for this key.
def model_order():
    models = [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS   # main model, then backups
    if _working_model in models:                          # a model already answered in this run
        models.remove(_working_model)                     # take it out ...
        models.insert(0, _working_model)                  # ... and put it first
    return [m for m in models if m not in _dead_models]   # drop models that said "not found"


# Sends one prompt to Gemini and returns (answer text, model that answered).
# 1. Checks the cache first.
# 2. If a model is busy (503) or rate-limited (429), moves to the next model straight away.
# 3. Only when every model is busy does it wait (10 s, 20 s, ...) and go round again.
# The model that answers is remembered, so the rest of the run goes straight to it.
def call_llm(prompt):
    global _working_model                                 # change the variable defined at the top
    cache = load_cache()                                  # saved answers
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest() # same prompt -> same key
    if key in cache:                                      # asked before
        return cache[key]["text"], cache[key]["model"]    # free and instant
    last_error = None                                     # remember the latest failure to show it at the end
    for round_number in range(1, config.MAX_RETRIES + 1): # a few rounds over all models
        models = model_order()                            # models to try in this round
        if not models:                                    # every model was "not found"
            raise RuntimeError("No usable Gemini model. Run list_models.py and fix config.py.")
        for model in models:                              # try each model once
            try:
                text = ask_model(model, prompt)           # one request
                if model != _working_model:               # first success, or a switch
                    print(f"  Using {model}")             # tell the user which model is answering
                _working_model = model                    # try this one first next time
                cache[key] = {"text": text, "model": model}   # remember the answer and who wrote it
                save_cache()                              # write to disk
                return text, model
            except Exception as error:                    # this model failed
                code = getattr(error, "code", None)       # HTTP status code, if the error has one
                if code in (400, 401):                    # bad API key: no model can help
                    raise                                 # stop now and show the real error
                if code in (403, 404):                    # model not available to this key
                    _dead_models.add(model)               # never try it again in this run
                print(f"  {model} unavailable ({code}), trying next model")   # short note
                last_error = error                        # keep it in case everything fails
        if round_number < config.MAX_RETRIES:             # rounds left
            wait = 10 * round_number                      # 10 s, 20 s, 30 s
            print(f"  All models busy. Waiting {wait} s before trying again...")
            time.sleep(wait)                              # wait, then another round
    raise last_error                                      # every round failed: show the last error


# Converts a length choice into a target word count.
# Accepts "short" / "medium" / "long" or a plain number of words.
def target_words(length):
    if isinstance(length, int):                           # user gave an exact number
        return length
    return config.SUMMARY_LENGTHS[length]                 # named length -> words


# Prepares the text of one result for the LLM: the article's opening plus its best-matching chunk.
# News articles put the main facts first, so the opening covers most of the story.
# If the chunk that matched the query sits later in the article, it is added so the LLM sees it too.
def document_for_llm(article_text, excerpt, chunk_id):
    words = article_text.split()                          # article as words
    opening = " ".join(words[:config.DOC_WORDS_FOR_LLM])  # first 400 words
    chunk_index = int(chunk_id.split("_c")[-1])           # "doc_00012_c03" -> 3
    if len(words) > config.DOC_WORDS_FOR_LLM and chunk_index >= 2:   # matching chunk is past the opening
        return opening + " [...] " + excerpt              # opening + the relevant part
    return opening                                        # opening already covers it


# Builds the prompt for a multi-document, query-focused summary.
# Numbered sources let the LLM cite [1], [2], ... so every statement can be traced back.
def build_query_prompt(query, docs, words):
    sources = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(docs, start=1))   # numbered documents
    return (f"You are summarizing news articles returned by a search engine.\n"
            f"User query: \"{query}\"\n\n"
            f"Articles:\n{sources}\n\n"
            f"Write one coherent summary of about {words} words that captures what these articles say "
            f"about the query.\n"
            f"Rules:\n"
            f"- Use only facts stated in the articles. Do not add outside knowledge.\n"
            f"- Cite sources in square brackets after each fact, e.g. [1] or [2][3].\n"
            f"- Focus on the articles that answer the query. Leave out articles about unrelated events "
            f"(search can return some); if you mention them at all, use one short sentence.\n"
            f"- If relevant articles describe different events, keep them separate instead of merging them "
            f"into one story.\n"
            f"- Write like a news report: start directly with the facts. Never use the words "
            f"\"article\", \"articles\", \"source\", \"sources\" or \"provided\".\n"
            f"- Plain prose, no headings, no bullet points.")


# Builds the prompt for a single-article summary (no query), in the style of the CNN/DailyMail highlights.
def build_article_prompt(article_text, words):
    return (f"Summarize the following news article in about {words} words.\n"
            f"Cover the most important facts: who, what, where, when.\n"
            f"Use only facts stated in the article. Plain prose, no headings, no bullet points.\n\n"
            f"Article:\n{article_text}")


# Summarizes the documents returned by search for a query.
# Returns the summary text, the model that wrote it, and the number of seconds it took.
def summarize_results(engine, query, results, length="medium"):
    from search import get_article                        # full article lookup
    start = time.time()                                   # start timer
    docs = [document_for_llm(get_article(engine, r["doc_id"]), r["excerpt"], r["chunk_id"])   # text per result
            for r in results]
    prompt = build_query_prompt(query, docs, target_words(length))   # full prompt
    summary, model = call_llm(prompt)                     # ask Gemini
    return summary, model, time.time() - start            # summary, model and seconds


# Summarizes one full article (no query). Used by evaluate_summary.py for ROUGE.
# Returns the summary text and the model that wrote it.
def summarize_article(article_text, length="short"):
    words = article_text.split()                          # article as words
    text = " ".join(words[:config.DOC_WORDS_FOR_LLM * 2]) # up to 800 words, enough for nearly all articles
    return call_llm(build_article_prompt(text, target_words(length)))   # ask Gemini


def main():
    from search import load_engine, search                # search engine
    print("Loading search engine...")                     # loading takes a while
    engine = load_engine()                                # load once
    query = "plane crash investigation"                   # demo query
    results, seconds = search(engine, query)              # top 5 documents
    print(f"Query: {query}  ({len(results)} documents, search {seconds * 1000:.0f} ms)")   # header
    for length in ["short", "medium", "long"]:            # try every length option
        summary, model, seconds = summarize_results(engine, query, results, length)   # summarize
        print(f"\n--- {length} (target {target_words(length)} words, "
              f"got {len(summary.split())} words, {seconds:.1f} s, {model}) ---")    # length check, time, model
        print(summary)                                    # the summary itself


if __name__ == "__main__":
    main()