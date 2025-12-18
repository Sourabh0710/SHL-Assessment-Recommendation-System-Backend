from fastapi import FastAPI, Query
from recommender import SHLRecommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Generative AI based SHL assessment recommender",
    version="1.0"
)

recommender = SHLRecommender("shl_catalog.csv")

@app.get("/recommend")
def recommend(
    query: str = Query(..., description="Job description or requirement text"),
    top_k: int = Query(5, ge=1, le=10, description="Number of recommendations (max 10)")
):
    return recommender.recommend(query, top_k)
