import asyncio
import httpx

async def test_rag():
    query = "hello, I hope this retrieval works!"
    url = f"http://localhost:8003/rag/{query}"  # Adjust port if needed

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    print("Status code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    asyncio.run(test_rag())