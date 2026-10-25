# data_prep.py
# Step 1 of the project: build a clean corpus that search and summarization can use.
#   1. Download CNN/DailyMail news articles (each comes with human-written highlights).
#   2. Clean them: remove "(CNN) --" datelines, Daily Mail bylines/timestamps, extra spaces.
#   3. Drop very short articles and exact duplicates.
#   4. Save the clean articles, then split each one into overlapping word chunks and save those.
# Run:  python data_prep.py

import os            # for creating the data folder
import re            # regular expressions for cleaning text
import hashlib       # MD5 hash to find duplicate articles
import pandas as pd  # tables and parquet files
import config        # project settings

# ---------- Cleaning patterns ----------
# CNN articles start with a dateline like "(CNN) -- " or "LONDON, England (CNN) -- ".
CNN_PREFIX = re.compile(r"^.{0,80}?\(CNN\)\s*(--\s*)?")                    # dateline must sit in the first 80 characters

# Daily Mail articles start with "By . Name . PUBLISHED: . 12:01 EST, 3 May 2013 . | . UPDATED: . 14:20 EST, 3 May 2013 ."
DM_UPDATED = re.compile(r"^.{0,400}?UPDATED:\s*\.\s*\d{1,2}:\d{2}\s*EST,\s*\d{1,2}\s+\w+\s+\d{4}\s*\.\s*")      # cut up to the UPDATED timestamp
DM_PUBLISHED = re.compile(r"^.{0,400}?PUBLISHED:\s*\.\s*\d{1,2}:\d{2}\s*EST,\s*\d{1,2}\s+\w+\s+\d{4}\s*\.\s*(\|\s*\.\s*)?")  # used when there is no UPDATED part
DM_BYLINE = re.compile(r"^By \.\s*(?:[^.]{1,60}\.\s*){1,3}?(?=[A-Z])")    # "By . Name . " with no timestamp after it
DM_LAST_UPDATED = re.compile(r"Last updated at \d{1,2}:\d{2}\s*(?:AM|PM) on \d{1,2}\w{0,2} \w+ \d{4}\s*\.?\s*")  # can appear anywhere
DM_SCROLL = re.compile(r"Scroll down for (?:the )?videos?\s*\.?\s*", re.IGNORECASE)   # Daily Mail "Scroll down for video ." note

SPACES = re.compile(r"\s+")         # any run of spaces, tabs or newlines
DOUBLE_DOTS = re.compile(r"(\s\.){2,}")   # leftover " . . " artifacts from removed lines


# Removes source-specific boilerplate from one article and fixes spacing.
# Boilerplate words (bylines, timestamps) would pollute BM25 matches and waste LLM tokens.
def clean_article(text):
    text = SPACES.sub(" ", text).strip()                  # one space everywhere first, so patterns match reliably
    text = CNN_PREFIX.sub("", text, count=1)              # remove the CNN dateline at the start
    if DM_UPDATED.search(text):                           # Daily Mail header with an UPDATED time
        text = DM_UPDATED.sub("", text, count=1)          # remove everything up to that time
    elif DM_PUBLISHED.search(text):                       # Daily Mail header with only a PUBLISHED time
        text = DM_PUBLISHED.sub("", text, count=1)        # remove everything up to that time
    else:                                                 # no timestamp found
        text = DM_BYLINE.sub("", text, count=1)           # remove a plain "By . Name ." byline
    text = DM_LAST_UPDATED.sub("", text)                  # remove "Last updated at ..." lines
    text = DM_SCROLL.sub("", text)                        # remove "Scroll down for video ." notes
    text = DOUBLE_DOTS.sub(" .", text)                    # collapse repeated " . " leftovers
    text = SPACES.sub(" ", text).strip()                  # tidy spaces again after removals
    return text                                           # cleaned article


# Highlights only need their newlines turned into normal sentence breaks.
# They are kept as the reference summaries for ROUGE later.
def clean_highlights(text):
    text = text.replace("\n", ". ")                       # each highlight line becomes its own sentence
    text = re.sub(r"\s+\.", ".", text)                    # "word ." -> "word." (dataset puts a space before full stops)
    text = re.sub(r"\.(\s*\.)+", ".", text)               # avoid ".." where a line already ended with "."
    return SPACES.sub(" ", text).strip()                  # tidy spaces


# Splits one article into overlapping word windows.
# Overlap keeps a sentence that crosses a boundary visible in at least one chunk.
def split_into_chunks(text):
    words = text.split()                                  # list of words
    size = config.CHUNK_WORDS                             # words per chunk
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP      # how far each new chunk moves forward
    spans = []                                            # list of (start, end) word positions
    start = 0                                             # first chunk starts at word 0
    while True:                                           # keep making chunks until the end
        end = min(start + size, len(words))               # chunk end, never past the last word
        spans.append((start, end))                        # remember this chunk
        if end >= len(words):                             # reached the end of the article
            break                                         # stop
        start = start + step                              # move forward by one step
    if len(spans) > 1:                                    # only check when there are 2+ chunks
        prev_start, prev_end = spans[-2]                  # the chunk before the last one
        new_words = spans[-1][1] - prev_end               # words the last chunk adds that the previous one lacks
        if new_words < config.MIN_NEW_WORDS:              # last chunk is mostly a repeat
            spans = spans[:-2] + [(prev_start, len(words))]   # extend previous chunk to the end, drop the last
    return [" ".join(words[s:e]) for s, e in spans]       # turn word positions back into text


# Downloads the dataset and keeps NUM_ARTICLES random articles.
def load_raw_articles():
    from datasets import load_dataset                     # imported here so cleaning functions work without it
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT)  # download
    ds = ds.shuffle(seed=config.RANDOM_SEED)              # random order, repeatable
    ds = ds.select(range(min(config.NUM_ARTICLES, len(ds))))  # keep the first NUM_ARTICLES
    return ds.to_pandas()[["id", "article", "highlights"]]    # convert to a pandas table


# Cleans, filters and de-duplicates the raw table, printing what happened at each step.
def clean_corpus(raw):
    df = raw.copy()                                       # do not change the original table
    df["raw_words"] = df["article"].str.split().str.len()   # word count before cleaning
    df["text"] = df["article"].apply(clean_article)       # clean every article
    df["highlights"] = df["highlights"].apply(clean_highlights)  # clean every highlight
    df["num_words"] = df["text"].str.split().str.len()    # word count after cleaning
    words_before = df["raw_words"].mean()                 # average before cleaning (all articles)
    words_after = df["num_words"].mean()                  # average after cleaning (same articles, so a fair comparison)
    words_removed = int(df["raw_words"].sum() - df["num_words"].sum())   # total boilerplate words removed
    n_changed = int((df["raw_words"] != df["num_words"]).sum())          # articles where cleaning removed something

    n_start = len(df)                                     # count before filtering
    df = df[df["num_words"] >= config.MIN_ARTICLE_WORDS]  # drop very short articles
    n_short = n_start - len(df)                           # how many were too short

    df["hash"] = df["text"].apply(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())  # fingerprint of each text
    n_before_dedup = len(df)                              # count before removing duplicates
    df = df.drop_duplicates(subset="hash")                # keep only the first copy of each text
    n_dupes = n_before_dedup - len(df)                    # how many duplicates were removed

    df = df.reset_index(drop=True)                        # clean row numbers 0..N-1
    df["doc_id"] = [f"doc_{i:05d}" for i in range(len(df))]  # simple ids like doc_00000

    print("----- Cleaning report -----")                  # header
    print(f"Articles loaded           : {n_start}")       # starting count
    print(f"Articles changed by clean : {n_changed}")     # how many had boilerplate
    print(f"Boilerplate words removed : {words_removed}") # total words cut
    print(f"Avg words before cleaning : {words_before:.1f}")   # all loaded articles, raw
    print(f"Avg words after cleaning  : {words_after:.1f}")    # same articles, cleaned
    print(f"Removed (too short)       : {n_short}")       # short ones
    print(f"Removed (duplicates)      : {n_dupes}")       # duplicates
    print(f"Articles kept             : {len(df)}")       # final count
    print(f"Avg words, kept articles  : {df['num_words'].mean():.1f}")  # final corpus average
    return df[["doc_id", "text", "highlights", "num_words"]]   # only the columns we need


# Turns every article into chunk rows: chunk_id, doc_id, chunk_index, text.
def build_chunks(articles):
    rows = []                                             # one dict per chunk
    for doc_id, text in zip(articles["doc_id"], articles["text"]):   # go through every article
        for i, chunk in enumerate(split_into_chunks(text)):   # every chunk of that article
            rows.append({"chunk_id": f"{doc_id}_c{i:02d}",    # unique chunk id, e.g. doc_00003_c01
                         "doc_id": doc_id,                    # which article it came from
                         "chunk_index": i,                    # position inside the article
                         "text": chunk})                      # chunk text
    chunks = pd.DataFrame(rows)                           # list of dicts -> table
    print("----- Chunking report -----")                  # header
    print(f"Total chunks              : {len(chunks)}")   # number of chunks
    print(f"Avg chunks per article    : {len(chunks) / len(articles):.2f}")  # chunks per article
    print(f"Avg words per chunk       : {chunks['text'].str.split().str.len().mean():.1f}")  # chunk size check
    return chunks                                         # chunk table


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)           # create data/ if missing
    raw = load_raw_articles()                             # download
    articles = clean_corpus(raw)                          # clean + filter
    articles.to_parquet(config.ARTICLES_FILE, index=False)    # save articles
    chunks = build_chunks(articles)                       # split into chunks
    chunks.to_parquet(config.CHUNKS_FILE, index=False)    # save chunks
    print(f"Saved {config.ARTICLES_FILE} and {config.CHUNKS_FILE}")   # done message
    print("----- Sample cleaned article (first 300 chars) -----")    # let the user eyeball the cleaning
    for t in articles["text"].head(3):                    # three examples
        print(t[:300], "\n")                              # print the start of each


if __name__ == "__main__":
    main()