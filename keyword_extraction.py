import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
data=pd.read_csv("C:\\Users\\rouna\\Downloads\\papers.csv")
print(data.head())
data = data.iloc[:5000,:]
print(data.shape)
# Adding new stopwords
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig","figure","image","sample","using",
             "show", "result", "large",
             "also", "one", "two", "three",
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_stop_words))

# cleaning process of data
def preprocess_text(txt):
    # Lower case
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    txt = [word for word in txt if word not in stop_words]
    # Remove words less than three letters
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatize
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)
# Demo
print(preprocess_text("HELO word loving moving the to from 99999 *&^ <p>This is a <b>sample</b> text with <i>HTML tags</i>.</p>"))

docs = data['paper_text'].apply(lambda x:preprocess_text(x))
print(docs)

# Apply countvectorizer
vectorizer=CountVectorizer(max_features=6000, ngram_range=(1, 2))

# Create a vocabulary and word count vectors
word_count_vectors = vectorizer.fit_transform(docs)


# apply Tdf Transformer
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vectors)


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    # taking top items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of features,score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]  # Fix: Changed '==' to '='
    return results


# get feature names
feature_names = vectorizer.get_feature_names_out()


def get_keywords(idx, docs):
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(vectorizer.transform([docs[idx]]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords


def print_results(idx, keywords, df):
    # now print the results
    print("\n=====Title=====")
    print(df['title'][idx])
    print("\n=====Abstract=====")
    print(df['abstract'][idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k, keywords[k])


idx = 941
keywords = get_keywords(idx, docs)
print_results(idx, keywords, data)


# pickle import libraray

import pickle
pickle.dump(tfidf_transformer,open('tfidf_transformer.pkl','wb'))
pickle.dump(vectorizer,open('count_vectorizer.pkl','wb'))
pickle.dump(feature_names,open('feature_names.pkl','wb'))