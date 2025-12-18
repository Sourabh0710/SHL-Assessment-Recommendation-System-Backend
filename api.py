from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import uvicorn

from recommender import SHLRecommender
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API to recommend SHL assessments based on job description text",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = SHLRecommender(csv_path="shl_catalog.csv")


@app.get("/recommend", summary="Get SHL assessment recommendations")
def recommend(
    query: str = Query(..., description="Job description or requirement text"),
    top_k: int = Query(10, ge=1, le=10, description="Number of recommendations (max 10)")
):
    """
    Returns top-k SHL assessment recommendations for a given query.
    """

    results = recommender.recommend(query, top_k=top_k)

    return results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
