from fastapi import FastAPI

from src import room


app = FastAPI()

app.include_router(router=room.router, prefix="/room")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Room API"}
