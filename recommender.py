import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SHLRecommender:
    def __init__(self, catalog_path="shl_catalog.csv"):
        self.df = pd.read_csv(catalog_path)

        self.df["combined_text"] = (
            self.df["assessment_name"].fillna("") + " " +
            self.df["description"].fillna("")
        )

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            show_progress_bar=True
        )

    def recommend(self, query, top_k=10):
        query_embedding = self.model.encode([query])
        similarity_scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        self.df["score"] = similarity_scores

        results = self.df.sort_values(
            "score",
            ascending=False
        ).head(top_k)

        return results[[
            "assessment_name",
            "test_type",
            "description",
            "url",
            "score"
        ]]
