import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.df.columns = [c.strip().lower() for c in self.df.columns]

        if "assessment_name" in self.df.columns:
            self.name_col = "assessment_name"
        elif "name" in self.df.columns:
            self.name_col = "name"
        else:
            raise ValueError(
                "CSV must contain either 'assessment_name' or 'name' column"
            )

        self.test_type_col = "test_type" if "test_type" in self.df.columns else None
        self.duration_col = "duration" if "duration" in self.df.columns else None
        self.remote_col = "remote_testing" if "remote_testing" in self.df.columns else None
        self.adaptive_col = "adaptive_irt" if "adaptive_irt" in self.df.columns else None
        self.url_col = "url" if "url" in self.df.columns else None

        self.df["text"] = self.df[self.name_col].fillna("")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

    def recommend(self, query: str, max_results: int = 10):
        if not query or not query.strip():
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except ValueError:
            return []

        top_indices = scores.argsort()[::-1][:max_results]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            row = self.df.iloc[idx]

            results.append({
                "assessment_name": row.get(self.name_col, ""),
                "test_type": row.get(self.test_type_col, "") if self.test_type_col else "",
                "duration": row.get(self.duration_col) if self.duration_col else None,
                "remote_testing": row.get(self.remote_col) if self.remote_col else None,
                "adaptive_irt": row.get(self.adaptive_col) if self.adaptive_col else None,
                "url": row.get(self.url_col) if self.url_col else None,
            })

        return results
