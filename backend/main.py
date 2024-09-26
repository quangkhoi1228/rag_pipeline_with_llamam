import tracemalloc

from fastapi import FastAPI
from src.statistic import start_scheduler
from src import chat, faq, feedback, statistic
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động: các hành động như kết nối DB, khởi chạy dịch vụ
    print("App startup")
    scheduler = start_scheduler()
    yield  # Tại đây ứng dụng sẽ chạy
    # Tắt ứng dụng: đóng kết nối, dọn dẹp tài nguyên
    print("App shutdown")
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

tracemalloc.start()

# app.include_router(router=room.router, prefix="/room")
app.include_router(router=chat.router, prefix="/chat", tags=['Chat'])
app.include_router(router=feedback.router,
                   prefix="/feedback", tags=['Feedback'])
app.include_router(router=statistic.router,
                   prefix="/statistic", tags=['Statistic'])
app.include_router(router=faq.router, prefix="/faq", tags=['FAQ'])


@app.get("/")
async def hello():
    return {"message": "Welcome to the Room API"}
