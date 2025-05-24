import httpx
import asyncio

async def test_upload(test_file_path, new=True):
    url = "http://localhost:8001/upload"

    with open(test_file_path, "rb") as f:
        files = {"file": (test_file_path, f)}
        data = {"new": str(new).lower()}  # 'true' or 'false' for boolean in form-data

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, data=data, files=files)

    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except Exception:
        print("Raw Response:", response.text)

if __name__ == "__main__":
    asyncio.run(test_upload('testdoc1.md'))
    asyncio.run(test_upload('testdoc2.txt', new=False))