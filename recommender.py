import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.df.columns = self.df.columns.str.strip().str.lower()

        required_cols = ["assesment_name", "test_type", "url"]
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df["text"] = (
            self.df["assesment_name"].fillna("") + " " +
            self.df["test_type"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

    def recommend(self, query: str, max_results: int = 10):
        if not query or not query.strip():
            return []

        query = query.strip().lower()

        try:
            query_vec = self.vectorizer.transform([query])
        except ValueError:
            return []

        if query_vec.nnz == 0:
            fallback = self.df.head(max_results)
            return fallback.apply(lambda row: {
                "assesment_name": row.get("assesment_name", ""),
                "test_type": row.get("test_type", ""),
                "duration": row.get("duration", ""),
                "remote_testing": row.get("remote_testing", ""),
                "adaptive_irt": row.get("adaptive_irt", ""),
                "url": row.get("url", "")
            }, axis=1).tolist()

        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:max_results]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "assesment_name": row.get("assesment_name", ""),
                "test_type": row.get("test_type", ""),
                "duration": row.get("duration", ""),
                "remote_testing": row.get("remote_testing", ""),
                "adaptive_irt": row.get("adaptive_irt", ""),
                "url": row.get("url", "")
            })

        return results
