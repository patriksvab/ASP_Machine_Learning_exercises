# Natural Language Processing, solution by Patrik Svab

# Importing

from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from string import digits, punctuation
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import scipy.cluster.hierarchy as sch
import re
from math import floor

# 1. Regular Expressions

# Regex practice problems, solved online and applied in the exercise 4a)

# 2. Speeches I

# a) Reading Files

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

tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem,
                        ngram_range=(1, 3))
tfidf_matrix = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.todense().T,
                        index=tfidf.get_feature_names_out())

# c) Saving the Matrix

pickle_out = open("./output/speech_matrix.pk", "wb")
pickle.dump(tfidf_matrix, pickle_out)
pickle_out.close()

terms = pd.DataFrame(df_tfidf.index)
terms.to_csv("./output/terms.csv")

# 3. Speeches II

# a) Reading the Matrix

pickle_in = open("./output/speech_matrix.pk", "rb")
loaded_matrix = pickle.load(pickle_in)
pickle_in.close()

# b) Dendrogram

array_dendro = loaded_matrix.toarray()
Z = sch.linkage(array_dendro, method="complete", metric="cosine")

# c) Saving Dendrogram
# Check the threshold

plt.figure()
sch.dendrogram(Z, color_threshold=0.85, no_labels=True)
plt.savefig("./output/speeches_dendrogram.pdf")

# 4. Job Ads

# a) Reading File and Splitting Columns

FNAME = open("./data/Stellenanzeigen.txt", mode="r", encoding="utf-8").read()

newspaper = re.findall(r"(.*),\s\d{1,2}\.\s\w+\s\d{4}", FNAME)
date = re.findall(r".*,\s(\d{1,2}\.\s\w+\s\d{4})", FNAME)
ads = re.findall(r".*,\s\d{1,2}\.\s\w+\s\d{4}\s+(.*\n?.*)", FNAME)

job_ads_df = pd.DataFrame({"Newspaper": newspaper, "Date": date,
                           "Job Ad": ads})
job_ads_df["Date"] = job_ads_df["Date"].str.replace("März", "3.")
job_ads_df["Date"] = job_ads_df["Date"].astype("datetime64[ns]")

# b) Words per Job Ad

job_ads_df["Words per Job Ad"] = job_ads_df["Job Ad"].\
    apply(lambda x: len(str(x).split(" ")))

years = job_ads_df["Date"].dt.year


def year_to_decade(year: int) -> str:
    decade = floor(int(year) / 10) * 10
    return f"{decade}s"


decades = years.apply(year_to_decade)

df2 = pd.DataFrame({"Decade": decades,
                    "Words per Job Ad": job_ads_df["Words per Job Ad"]})
df_for_plotting = df2.groupby(["Decade"]).agg("mean")

df_for_plotting.plot.bar()
plt.legend(["Average Words per Job Ad"])

# c) Aggregating Job Ads by Decade

df3 = pd.DataFrame({"Decade": decades, "Job Ads": job_ads_df["Job Ad"]})
df_ads_by_decade = pd.DataFrame(df3.groupby(["Decade"]).sum())

_stemmer_ger = nltk.snowball.SnowballStemmer("german")
_stopwords_ger = nltk.corpus.stopwords.words("german")
_stopwords_eng = nltk.corpus.stopwords.words("english")
_stopwords_all = _stopwords_ger + _stopwords_eng


def tokenize_and_stem_ger(text):
    """Return tokens of text deprived of numbers and punctuation."""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer_ger.stem(t) for t in nltk.word_tokenize(text.lower())]


corpus_ger = df_ads_by_decade["Job Ads"].tolist()
corpus_ger_str = []
for item in corpus_ger:
    a = str(item)
    corpus_ger_str.append(a)

corpus_ger_without_symbol = []
for item in corpus_ger_str:
    b = item.replace('§', '')
    corpus_ger_without_symbol.append(b)

count = CountVectorizer(stop_words=_stopwords_all,
                        tokenizer=tokenize_and_stem_ger)
count.fit(corpus_ger_without_symbol)
count_matrix = count.transform(corpus_ger_without_symbol)
df_count = pd.DataFrame(count_matrix.todense().T,
                        index=count.get_feature_names_out(),
                        columns=df_ads_by_decade.index)

top_words_by_decade = pd.DataFrame()

for i in range(0, len(df_count.columns)):
    col = df_count.columns[i]
    sort = df_count[col].sort_values(ascending=False).iloc[0:10]
    top_words_by_decade[col] = sort.index

print(top_words_by_decade)
