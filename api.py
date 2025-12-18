from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
