from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from recommender import SHLRecommender
import os

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API to recommend SHL assessments based on a job description or query text",
    version="1.0.0"
)

_recommender = None

def get_recommender():
    global _recommender

    if _recommender is None:
        csv_path = "shl_catalog.csv"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

        _recommender = SHLRecommender(csv_path)

    return _recommender

class RecommendRequest(BaseModel):
    text: str
    max_results: int = 10


class RecommendResponse(BaseModel):
    assessment_name: str
    test_type: str
    duration: str | None = None
    remote_testing: str | None = None
    adaptive_irt: str | None = None
    url: str | None = None

@app.get("/")
def root():
    """
    Root endpoint for evaluator clarity
    """
    return {
        "message": "SHL Assessment Recommendation API is running",
        "endpoints": [
            {
                "path": "/recommend",
                "method": "POST",
                "description": "Get assessment recommendations based on a job description or query"
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "Swagger UI"
            }
        ],
        "version": "1.0.0"
    }

@app.get("/recommend")
def recommend_info():
    """
    Friendly GET endpoint to explain how to use POST /recommend
    """
    return {
        "message": "Use POST /recommend to get assessment recommendations",
        "method": "POST",
        "content_type": "application/json",
        "example_request": {
            "text": "Python Developer",
            "max_results": 10
        },
        "example_curl": (
            "curl -X POST https://<your-domain>/recommend "
            "-H 'Content-Type: application/json' "
            "-d '{\"text\": \"Python Developer\", \"max_results\": 10}'"
        )
    }

@app.post("/recommend", response_model=List[RecommendResponse])
def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint
    """
    try:
        recommender = get_recommender()

        results = recommender.recommend(
            query=request.text,
            max_results=request.max_results
        )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )
