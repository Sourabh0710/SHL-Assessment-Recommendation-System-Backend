import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.df["text"] = self.df["name"].fillna("") + " " + self.df["test_type"].fillna("")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

    def recommend(self, query: str, top_k: int = 10):
        query_vec = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarity_scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "assessment_name": row["name"],
                "url": row["url"],
                "test_type": row["test_type"],
                "duration": row.get("duration"),
                "remote_testing": row.get("remote_testing"),
                "adaptive_irt": row.get("adaptive_irt"),
                "score": float(similarity_scores[idx])
            })

        return results
