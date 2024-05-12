import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def train_faiss_index(csv_filepath: str):
    netflix_df = pd.read_csv(csv_filepath)

    netflix_df = netflix_df.fillna("")

    netflix_df["combined"] = netflix_df["title"] + " " + netflix_df["description"]

    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = set(stopwords.words("english"))
    netflix_df["combined"] = netflix_df["combined"].apply(
        lambda x: " ".join(
            [
                word.lower()
                for word in word_tokenize(x)
                if word.lower() not in stop_words
            ]
        )
    )

    tfidf = TfidfVectorizer()
    tfidf_matrix = (
        tfidf.fit_transform(netflix_df["combined"]).toarray().astype("float32")
    )

    index = faiss.IndexIDMap(faiss.IndexFlatIP(tfidf_matrix.shape[1]))
    faiss.normalize_L2(tfidf_matrix)
    index.add_with_ids(tfidf_matrix, np.arange(tfidf_matrix.shape[0]))

    return index


def save_faiss_index(faiss_index):
    faiss.write_index(faiss_index, "faiss/netflix.index")


if __name__ == "__main__":
    csv_filepath = "faiss_trainer/netflix_titles.csv"

    faiss_index = train_faiss_index(csv_filepath)
    save_faiss_index(faiss_index)
