from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from vector_store import FAISSHandler
import os
from pathlib import Path
from typing import Any

app = FastAPI()

class ChunkRequest(BaseModel):
    chunk: str
    filename: str

store_path = Path(os.environ.get('VECTOR_STORE_PATH',"/vector-store/db.pkl"))
store_path.parent.mkdir(exist_ok=True,parents=True)
store = FAISSHandler(store_path=store_path)


@app.post("/end")
def end():
    try:
        store._reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "message": "Conversation Ended"
    }
    

@app.post("/embed")
def embed_chunks(req: ChunkRequest):
    if not req.chunk:
        raise HTTPException(status_code=400, detail="No chunks provided")
    try:
        filename, chunk = req.filename, req.chunk
        store.add_documents([chunk])
        store._save()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Chunks embedded and stored",
        "filename": req.filename,
        "store_count": store.index.ntotal
    }


class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search(req: SearchRequest):
    try:
        top_results = store.search(req.query)
        context = "\n\n".join(top_results)
        context = f"<context>\n\n{context}\n\n</context>"

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "context": context
    }