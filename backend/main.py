import tracemalloc

from fastapi import FastAPI
from src import chat, faq, feedback

app = FastAPI()

tracemalloc.start()

# app.include_router(router=room.router, prefix="/room")
app.include_router(router=chat.router, prefix="/chat", tags=['Chat'])
app.include_router(router=feedback.router,
                   prefix="/feedback", tags=['Feedback'])
app.include_router(router=faq.router, prefix="/faq", tags=['FAQ'])


@app.get("/")
async def hello():
    return {"message": "Welcome to the Room API"}
