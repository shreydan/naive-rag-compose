from fastapi import FastAPI, HTTPException
import httpx
import os

app = FastAPI()

@app.get("/rag/{query}")
async def rag(query: str):
    EMBEDDING_SERVICE_URL = os.environ.get('EMBEDDING_API',"http://embedding:8002")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/search", 
                json={"query": query}
            )
            response = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return response