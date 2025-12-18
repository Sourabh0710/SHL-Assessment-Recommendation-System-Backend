from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from recommender import SHLRecommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = None
def get_recommender():
    global recommender
    if recommender is None:
        recommender = SHLRecommender("shl_catalog.csv")
    return recommender

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=10)
):
    model = get_recommender()
    return model.recommend(query, top_k)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)

