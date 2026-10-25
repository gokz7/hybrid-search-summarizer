# rate_app.py
# Helper page for the human evaluation: shows the 20 query summaries one at a time,
# with their source articles, and saves your 1-5 ratings straight into results/human_eval.csv.
# The LLM fact-check score is hidden on purpose, so your ratings stay independent of it.
# Run:  streamlit run rate_app.py      (after evaluate_summary.py has created the sheet)
# Then: python evaluate_summary.py     (prints the averages)

import re                         # citation marks
import html                       # safe HTML
import pandas as pd               # read and save the sheet
import streamlit as st            # web page
import config                     # file paths

RATINGS = ["relevance", "coherence", "faithfulness"]     # the three scores you give
QUESTIONS = {"relevance": "Relevance: does it answer the question?",
             "coherence": "Coherence: is it clear and easy to follow?",
             "faithfulness": "Faithfulness: are the facts correct? (check 2-3 facts in the sources)"}
HELP = "5 = fully, 4 = mostly (one small issue), 3 = partly / one clear error, 2 = poorly / several errors, 1 = not at all"

st.set_page_config(page_title="Rate summaries", layout="wide")   # browser tab title


# Reads the sheet fresh each time, so saved ratings are always shown.
def load_sheet():
    return pd.read_csv(config.HUMAN_EVAL_FILE, dtype={"comments": str})   # comments stay text


# Loads the cleaned articles once (only the text is needed here, no models).
@st.cache_resource
def load_articles():
    return pd.read_parquet(config.ARTICLES_FILE).set_index("doc_id")["text"]   # doc_id -> text


# Shows [1] / [2][3] citations as bold red numbers so they are easy to match with the sources.
def with_citations(summary):
    safe = html.escape(summary)                           # protect against stray < > &
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", r"<b style='color:#B3261E'>[\1]</b>", safe)


# Saves the ratings of one row into the CSV file.
def save_row(sheet, i, values, comment):
    for name in RATINGS:                                  # each score
        sheet.loc[i, name] = values[name]                 # store it
    sheet.loc[i, "comments"] = comment                    # store the note
    sheet.to_csv(config.HUMAN_EVAL_FILE, index=False)     # write the whole sheet back


def main():
    sheet = load_sheet()                                  # current ratings
    articles = load_articles()                            # article texts
    total = len(sheet)                                    # 20
    rated = int(sheet[RATINGS].notna().all(axis=1).sum()) # rows with all three scores
    if "i" not in st.session_state:                       # first visit
        unrated = sheet.index[sheet[RATINGS].isna().any(axis=1)]   # rows still missing a score
        st.session_state.i = int(unrated[0]) if len(unrated) else 0   # start at the first unrated row
    i = st.session_state.i                                # current row number
    row = sheet.loc[i]                                    # current row

    st.title("Rate the summaries")
    st.progress(rated / total, text=f"{rated} of {total} rated")   # overall progress
    st.subheader(f"Summary {i + 1} of {total}")
    st.markdown(f"**Question:** {html.escape(row['query'])}")
    st.markdown(f"<div style='font-size:1.1rem;line-height:1.7'>{with_citations(row['summary'])}</div>",
                unsafe_allow_html=True)                   # the summary to judge

    left, right = st.columns([3, 2], gap="large")         # sources left, scores right
    with left:
        st.markdown("**Sources** (the [n] numbers in the summary match these)")
        for n, doc_id in enumerate(str(row["sources"]).split(", "), start=1):   # each source article
            text = articles.get(doc_id, "(article not found)")   # full text
            with st.expander(f"[{n}] {text[:110]}..."):   # opening line as the title
                st.write(text)                            # full article

    with right:
        values = {}                                       # the scores chosen on this page
        for name in RATINGS:                              # three questions
            current = row[name]                           # saved score, if any
            index = int(current) - 1 if pd.notna(current) else None   # preselect it
            values[name] = st.radio(QUESTIONS[name], [1, 2, 3, 4, 5], index=index,
                                    horizontal=True, help=HELP, key=f"{name}_{i}")
        old_comment = row["comments"] if pd.notna(row["comments"]) else ""   # saved note
        comment = st.text_input("Comment (optional)", value=old_comment, key=f"comment_{i}")
        complete = all(v is not None for v in values.values())   # all three chosen?

        back, forward = st.columns(2)                     # navigation buttons
        if back.button("Previous", disabled=i == 0):      # not on the first row
            if complete:                                  # keep what was chosen
                save_row(sheet, i, values, comment)
            st.session_state.i = i - 1                    # go back
            st.rerun()
        if forward.button("Save and next" if i < total - 1 else "Save", type="primary", disabled=not complete):
            save_row(sheet, i, values, comment)           # write to the CSV
            st.session_state.i = min(i + 1, total - 1)    # next row (stay on the last one)
            st.rerun()
        if not complete:                                  # explain the disabled button
            st.caption("Choose all three scores to save.")

    if rated == total:                                    # finished
        st.success("All summaries rated. Stop this page (Ctrl+C) and run: python evaluate_summary.py")


main()