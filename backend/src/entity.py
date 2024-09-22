
from datetime import datetime
from pydantic import BaseModel


class FAQ(BaseModel):
    question: str
    answer: str


class Chat(BaseModel):
    message: str = None  # Auto-generated UUID
    sender: str
    created_date: datetime = datetime.now()


class Feedback(BaseModel):
    question: str
    answer: str
    feedback: str
    created_date: datetime = datetime.now()
