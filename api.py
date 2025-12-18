from fastapi import FastAPI
from pydantic import BaseModel
from recommender import SHLRecommender

app = FastAPI()

recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = SHLRecommender("shl_catalog.csv")
    return recommender

@app.get("/")
def health():
    return {"status": "ok"}

class Query(BaseModel):
    text: str
    max_results: int = 10

@app.post("/recommend")
def recommend(query: Query):
    rec = get_recommender()
    results = rec.recommend(query.text, query.max_results)
    return {
        "query": query.text,
        "results": results
    }
