# 1. Regular Expressions

# Regex practice problems

# 2. Speeches I

# a) Reading Files

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from string import digits, punctuation
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import scipy.cluster.hierarchy as sch

FILES = list(Path("data/speeches").glob("R0*"))

corpus = []
for i in FILES:
    try:
        e = open(i, mode='r', encoding="utf-8").read()
        corpus.append(e)
    except UnicodeDecodeError:
        print(f'{i}')

# b) Vectorization

def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and punctuation."""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]

_stemmer = nltk.snowball.SnowballStemmer("english")
_stopwords = nltk.corpus.stopwords.words("english")

tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
tfidf_matrix = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.todense().T,
                        index=tfidf.get_feature_names_out())

# c) Saving the Matrix

pickle_out = open("./output/speech_matrix.pk","wb")
pickle.dump(tfidf_matrix, pickle_out)

terms = pd.DataFrame(df_tfidf.index)
terms.to_csv("./output/terms.csv")

# 3. Speeches II

# a) Reading the Matrix

pickle_in = open("./output/speech_matrix.pk","rb")
loaded_matrix = pickle.load(pickle_in)

# b) Dendrogram

array_dendro = loaded_matrix.toarray()
Z = sch.linkage(array_dendro, method="complete", metric="cosine")

# c) Saving Dendrogram

plt.figure()
sch.dendrogram(Z, color_threshold=0.85, labels=None)
plt.savefig("./output/speeches_dendrogram.pdf")

