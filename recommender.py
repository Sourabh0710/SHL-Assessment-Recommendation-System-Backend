import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, csv_path: str):

        self.df = pd.read_csv(csv_path)

        self.df.columns = self.df.columns.str.strip().str.lower()

        required_columns = {"name", "url", "test_type"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        self.df["combined_text"] = (
            self.df["name"].fillna("") + " " +
            self.df["test_type"].fillna("")
        )

        self.df = self.df[self.df["combined_text"].str.strip() != ""]
        
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_text"]
        )

    def recommend(self, query: str, max_results: int = 10):
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

        self.df["score"] = similarities

        top_results = self.df.sort_values(
            by="score", ascending=False
        ).head(max_results)

        recommendations = []
        for _, row in top_results.iterrows():
            recommendations.append({
                "assesment_name": row.get("name", ""),
                "test_type": row.get("test_type", ""),
                "duration": row.get("duration"),
                "remote_testing": row.get("remote_testing"),
                "adaptive_irt": row.get("adaptive_irt"),
                "url": row.get("url")
            })

        return recommendations
