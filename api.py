from fastapi import FastAPI, Query
from pydantic import BaseModel
from recommender import SHLRecommender
import uvicorn

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API to recommend SHL assessments based on job description text",
    version="1.0"
)

recommender = SHLRecommender("shl_catalog.csv")

class RecommendationResponse(BaseModel):
    assessment_name: str
    url: str
    test_type: str
    duration: str | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
@app.get("/recommend", response_model=list[RecommendationResponse])
def recommend(
    query: str = Query(..., description="Job description or requirement text"),
    top_k: int = Query(10, ge=1, le=10, description="Number of recommendations (max 10)")
):
    results = recommender.recommend(query, top_k)

    return results.drop(columns=["score"]).to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

