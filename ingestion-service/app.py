from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import os
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

app = FastAPI()

def create_chunks(content: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = splitter.split_text(content)
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), new: bool = Form(...)):

    EMBEDDING_SERVICE_URL = os.environ.get('EMBEDDING_API', "http://embedding:8002")
    
    if new:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/end",
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to create a new conversation")


    if not (file.filename.endswith(".txt") or file.filename.endswith(".md")):
        raise HTTPException(status_code=400, detail="Only text and markdown files supported")

    content = (await file.read()).decode("utf-8")
    chunks = create_chunks(content)

    store_count = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, chunk in enumerate(chunks):
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"chunk": chunk, "filename": file.filename}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Embedding failed at chunk {idx}")
            response = response.json()
            store_count = response['store_count']

    return {
        "message": f"File {file.filename} uploaded and all {len(chunks)} chunks embedded successfully",
        "store_count": store_count
    }