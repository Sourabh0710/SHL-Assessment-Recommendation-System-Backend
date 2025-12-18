from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

class Query(BaseModel):
    text: str
    max_results: int = 10

@app.post("/recommend")
def recommend(query: Query):
    return {
        "query": query.text,
        "results": []
    }
