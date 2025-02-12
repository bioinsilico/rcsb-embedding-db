import asyncio
import uvicorn
from fastapi import FastAPI

app = FastAPI()

global_variable = None

@app.get("/")
async def read_root():
    return {"message": f"Global variable value: {global_variable}"}

async def main():
    global global_variable
    global_variable = "Hello, Uvicorn!"
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())