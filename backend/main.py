from fastapi import FastAPI

from src import room, chat
from langchain_community.llms import Ollama


app = FastAPI()

# app.include_router(router=room.router, prefix="/room")
app.include_router(router=chat.router, prefix="/chat", tags=['Chat'])


@app.get("/")
async def hello():
    return {"message": "Welcome to the Room API"}
