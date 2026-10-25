# text_utils.py
# Turns text into a list of keywords for BM25 keyword search.
# The same function is used when building the index and when searching,
# so a word is always written the same way on both sides (e.g. "Police" and "police" match).

import re  # regular expressions

# Very common words that appear in almost every article. They carry no meaning for search,
# so removing them makes BM25 focus on the words that actually describe the topic.
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own said same
says she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves mr mrs ms one two new also year years
""".split())  # split the block into single words


# Lowercases the text, keeps only letters and digits, and removes stopwords.
# Example: "The Police arrested 2 men in Texas!" -> ["police", "arrested", "men", "texas"]
def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())        # lowercase, then pull out words and numbers
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]   # drop 1-letter tokens and stopwords