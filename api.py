from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from recommender import SHLRecommender
import os

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommend SHL assessments based on a text query",
    version="1.0.0"
)
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        csv_path = "shl_catalog.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

        recommender = SHLRecommender(csv_path)

    return recommender

class RecommendRequest(BaseModel):
    text: str
    max_results: int = 10

class RecommendResponse(BaseModel):
    assesment_name: str
    test_type: str
    duration: str | None = None
    remote_testing: str | None = None
    adaptive_irt: str | None = None
    url: str | None = None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=List[RecommendResponse])
def recommend(request: RecommendRequest):
    try:
        rec = get_recommender()
        results = rec.recommend(
            query=request.text,
            max_results=request.max_results
        )
        if not results:
            return []
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )
