# app.py
# Step 7 of the project: the web interface (Streamlit), styled as a newspaper front page.
#   - Masthead "The Search Gazette" with the real article count and today's date.
#   - Search box with auto-suggestions: while you type, it suggests names that appear often
#     in the articles (e.g. "Manchester United", "Barack Obama"). You can also type anything.
#   - The query becomes the lead headline; the summary is set in two columns with a drop cap,
#     and citations [1], [2] become small red marks that point to the clippings below.
#   - Results are "clippings": the article's first sentence as headline, the best-matching excerpt,
#     and the full article in an expander. 6 per page, with Previous / Next buttons.
#   - Sidebar: Accurate (hybrid + reranker, ~1 s) or Fast (hybrid, ~20 ms) search,
#     summary length (short / medium / long) and how many documents to summarize.
# Colours and fonts for the whole app are in .streamlit/config.toml and the CSS below.
# Run:  streamlit run app.py

import re                         # find names for suggestions, turn [n] into red marks
import html                       # make article text safe to put inside HTML
from datetime import date         # today's date for the masthead
from collections import Counter   # count how often each name appears
import streamlit as st            # web interface
import config                     # project settings
from text_utils import STOPWORDS  # common words (to drop bad suggestions like "The Prime")
from search import load_engine, search, get_article   # search engine
from summarize import summarize_results              # Gemini summaries

PAGE_SIZE = 6                     # clippings per page (3 rows of 2)
MAX_RESULTS = 18                  # results fetched per search (3 pages)
MODES = {"Accurate (~1 s)": "hybrid_rerank",          # label shown -> search method
         "Fast (~20 ms)": "hybrid"}

st.set_page_config(page_title="The Search Gazette", layout="wide")   # browser tab title, wide layout

# Newspaper styling. Fonts come from Google Fonts:
#   UnifrakturMaguntia = blackletter masthead (classic newspaper nameplate)
#   Libre Caslon       = headlines and body text (the typeface of old printed newspapers)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=Libre+Caslon+Display&family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input { font-family: 'Libre Caslon Text', Georgia, serif; }
.block-container { max-width: 1100px; padding-top: 4rem; }

/* Masthead */
.dateline { display: flex; justify-content: space-between; font-size: 0.85rem; padding-bottom: 0.3rem;
            border-bottom: 1px solid #1C1C1A; }
.nameplate { font-family: 'UnifrakturMaguntia', serif; font-size: clamp(2.6rem, 7vw, 5rem); text-align: center;
             line-height: 1.05; margin: 0.6rem 0 0.1rem 0; }
.motto { text-align: center; font-style: italic; font-size: 0.95rem; margin-bottom: 0.4rem; }
.double-rule { border-top: 3px double #1C1C1A; margin: 0.2rem 0 1.2rem 0; }

/* Lead story: the query as headline, summary in two columns with a drop cap */
.lead-headline { font-family: 'Libre Caslon Display', serif; font-size: clamp(1.9rem, 4vw, 3rem);
                 line-height: 1.1; margin: 0.8rem 0 0.3rem 0; }
.byline { font-style: italic; font-size: 0.9rem; margin-bottom: 0.9rem; }
.lead-body { column-count: 2; column-gap: 2.2rem; column-rule: 1px solid #C9C6BD; font-size: 1.05rem;
             line-height: 1.65; text-align: left; }
.lead-body::first-letter { font-family: 'Libre Caslon Display', serif; float: left; font-size: 4.2rem;
                           line-height: 0.8; padding: 0.35rem 0.5rem 0 0; }
.cite { color: #B3261E; font-weight: 700; font-size: 0.7em; vertical-align: super; margin-left: 1px; }
@media (max-width: 800px) { .lead-body { column-count: 1; } }

/* Clippings (search results) */
.section-rule { border-top: 1px solid #1C1C1A; margin: 1.6rem 0 0.4rem 0; }
.section-title { font-family: 'Libre Caslon Display', serif; font-size: 1.4rem; margin-bottom: 0.8rem; }
.clip-number { color: #B3261E; font-weight: 700; margin-right: 0.35rem; }
.clip-headline { font-family: 'Libre Caslon Display', serif; font-size: 1.25rem; line-height: 1.25;
                 margin: 0 0 0.35rem 0; }
.clip-excerpt { font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.3rem; }
.clip-meta { font-size: 0.8rem; font-style: italic; color: #55534E; }

/* Search box and buttons */
div[data-baseweb="select"] > div { background: #FBFAF7; border: 1px solid #1C1C1A; border-radius: 0; font-size: 1.1rem; }
.stButton > button { border-radius: 0; border: 1px solid #1C1C1A; background: transparent; }
[data-testid="stExpander"] details { border-radius: 0; border-color: #C9C6BD; }
button:focus-visible, input:focus-visible { outline: 2px solid #B3261E; outline-offset: 2px; }
</style>
"""


# Loads the search engine once and keeps it in memory for every user and every click.
@st.cache_resource(show_spinner="Loading search engine (first time only)...")
def get_engine():
    return load_engine()


# Builds the suggestion list once: multi-word capitalised names found in the articles' opening
# paragraphs, most frequent first. Example matches: "Manchester United", "White House", "Barack Obama".
@st.cache_resource(show_spinner="Preparing suggestions...")
def get_suggestions(_engine, how_many=600):
    counts = Counter()                                    # name -> how many articles mention it
    for text in _engine["articles"]["text"]:              # every article
        opening = " ".join(text.split()[:150])            # first 150 words hold the main names
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", opening)   # 2+ capitalised words in a row
        for name in set(names):                           # count each name once per article
            if name.split()[0].lower() not in STOPWORDS:  # skip "The Prime", "But Police", ...
                counts[name] += 1                         # one more article mentions it
    return [name for name, n in counts.most_common(how_many) if n >= 3]   # names in 3+ articles


# Runs the search only when the query or mode changes, and keeps the results between clicks
# (Streamlit re-runs this whole file on every click, so without this, every page change would search again).
def get_results(engine, query, method):
    key = (query, method)                                 # what was searched
    if st.session_state.get("last_key") != key:           # new search
        results, seconds = search(engine, query, top_n=MAX_RESULTS, method=method)   # search
        st.session_state.update(last_key=key, results=results, seconds=seconds, page=1)   # remember it, back to page 1
    return st.session_state["results"], st.session_state["seconds"]


# Uses the article's first sentence as its headline (the articles have no titles; news leads work as one).
def headline_of(text, max_chars=140):
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]   # text up to the first sentence end
    return first if len(first) <= max_chars else first[:max_chars].rsplit(" ", 1)[0] + "..."   # shorten long ones


# Turns the summary into safe HTML and replaces [1] / [2][3] / [2, 3] with small red citation marks.
# "[1][3]" becomes one mark "1,3" (two marks side by side would read as "13"),
# and the space before a mark is removed so it sits right after the word, like a footnote.
def summary_html(summary):
    safe = html.escape(summary)                           # protect against stray < > & in the text
    safe = re.sub(r"\]\s*\[", ", ", safe)                 # "[1][3]" -> "[1, 3]"
    return re.sub(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]",         # every [n] or [n, m], with the space before it
                  lambda m: f"<span class='cite'>{m.group(1).replace(' ', '')}</span>", safe)


# The masthead: dateline row, blackletter nameplate, motto, double rule.
def show_masthead(n_articles):
    today = date.today().strftime("%A, %d %B %Y")         # e.g. Sunday, 25 October 2026
    st.markdown(f"""
        <div class="dateline"><span>Vol. I, No. {n_articles:,}</span><span>{today}</span></div>
        <div class="nameplate">The Search Gazette</div>
        <div class="motto">All the news that fits your query</div>
        <div class="double-rule"></div>""", unsafe_allow_html=True)


# The lead story: the query as the headline, then the summary in two columns.
def show_lead(engine, query, results, length, n_summarize):
    st.markdown(f"<div class='lead-headline'>{html.escape(query[:1].upper() + query[1:])}</div>",
                unsafe_allow_html=True)                   # query as headline, first letter capitalised
    with st.spinner("Writing the summary..."):
        try:
            summary, model, seconds = summarize_results(engine, query, results[:n_summarize], length)
        except Exception as error:                        # e.g. Gemini quota used up
            st.error(f"The summary could not be written right now ({type(error).__name__}). "
                     f"Try again in a minute; the clippings below are still available.")
            return
    st.markdown(f"<div class='byline'>Summarized from the top {min(n_summarize, len(results))} clippings "
                f"by {model} in {seconds:.1f} s. Red numbers point to the clippings below.</div>"
                f"<div class='lead-body'>{summary_html(summary)}</div>", unsafe_allow_html=True)


# One clipping: red number, first-sentence headline, best-matching excerpt, full article on request.
def show_clipping(engine, result):
    article = get_article(engine, result["doc_id"])       # full text
    starts_mid_article = not result["chunk_id"].endswith("_c00")   # chunk 0 is the article's beginning
    excerpt = ("..." if starts_mid_article else "") + result["excerpt"][:320] + "..."   # mark cut text, like a quote
    st.markdown(f"""
        <div class="clip-headline"><span class="clip-number">{result['rank']}</span>{html.escape(headline_of(article))}</div>
        <div class="clip-excerpt">{html.escape(excerpt)}</div>
        <div class="clip-meta">{result['doc_id']}, {len(article.split())} words</div>""", unsafe_allow_html=True)
    with st.expander("Read the full article"):            # hidden until clicked
        st.write(article)


# Previous / page counter / Next under the clippings.
def show_pagination(total):
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)  # e.g. 18 results / 6 per page = 3 pages
    left, middle, right = st.columns([1, 2, 1])           # three columns
    if left.button("Previous page", disabled=st.session_state.page <= 1):   # not on page 1
        st.session_state.page -= 1                        # go back
        st.rerun()                                        # redraw with the new page
    middle.markdown(f"<div style='text-align:center;font-style:italic'>Page {st.session_state.page} of {pages}</div>",
                    unsafe_allow_html=True)               # page counter in the middle
    if right.button("Next page", disabled=st.session_state.page >= pages):  # not on the last page
        st.session_state.page += 1                        # go forward
        st.rerun()                                        # redraw with the new page


def main():
    st.markdown(CSS, unsafe_allow_html=True)              # apply the newspaper styling
    engine = get_engine()                                 # cached after the first load
    suggestions = get_suggestions(engine)                 # cached after the first load
    show_masthead(len(engine["articles"]))                # nameplate with the real article count

    with st.sidebar:                                      # settings panel on the left
        st.header("Edition settings")
        mode_label = st.radio("Search mode", list(MODES),  # accurate or fast
                              captions=["Hybrid search + reranker", "Hybrid search only"])   # what each one does
        length = st.select_slider("Summary length", options=["short", "medium", "long"], value="medium")
        st.caption(f"About {config.SUMMARY_LENGTHS[length]} words")   # show the target length
        n_summarize = st.slider("Clippings to summarize", 1, 10, 5)  # top N results go into the summary
        st.caption(f"{len(engine['articles']):,} CNN/DailyMail articles, 2007-2015. "
                   f"Hybrid search: BM25 keywords + MiniLM embeddings.")

    query = st.selectbox("Search the archive", options=suggestions, index=None,   # empty at start
                         accept_new_options=True,         # allow any text, not only suggestions
                         placeholder="Type a topic and press Enter, e.g. plane crash investigation")
    if not query:                                         # nothing searched yet
        st.markdown("<div class='byline'>Type a topic above, or start typing a name and pick a suggestion. "
                    "The answer is written from the matching articles, with every fact linked to its source.</div>",
                    unsafe_allow_html=True)
        return

    results, seconds = get_results(engine, query, MODES[mode_label])   # search (or reuse)
    if not results:                                       # nothing matched
        st.warning("No articles matched. Try other words, or a name from the suggestions.")
        return

    show_lead(engine, query, results, length, n_summarize)   # headline + summary

    st.markdown(f"<div class='section-rule'></div><div class='section-title'>Clippings</div>"
                f"<div class='clip-meta'>{len(results)} articles found in {seconds * 1000:.0f} ms</div>",
                unsafe_allow_html=True)
    start = (st.session_state.page - 1) * PAGE_SIZE       # first clipping on this page
    page_results = results[start:start + PAGE_SIZE]       # clippings on this page
    for row in range(0, len(page_results), 2):            # two clippings per row, like newspaper columns
        columns = st.columns(2, gap="large")              # left and right column
        for column, result in zip(columns, page_results[row:row + 2]):   # fill each column
            with column:
                show_clipping(engine, result)             # draw the clipping
    show_pagination(len(results))                         # page buttons


main()