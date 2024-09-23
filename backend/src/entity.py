
import uuid
from datetime import datetime

import numpy as np
from pydantic import BaseModel, conint, field_validator
from src.util import convert_int_to_string


class Room(BaseModel):
    id: str = None  # Auto-generated UUID
    name: str
    created_date: datetime = datetime.now()


class FAQ(BaseModel):
    id: int
    question: str
    answer: str

    @field_validator('id')
    def convert_id(cls, v):
        return convert_int_to_string(v)  # Sử dụng hàm chuyển đổi


class FAQPool(BaseModel):
    id: str
    faq_id: str
    question: str
    answer: str
    created_date: datetime = datetime.now()


class Chat(BaseModel):
    message: str = None  # Auto-generated UUID
    sender: str
    created_date: datetime = datetime.now()


class Feedback(BaseModel):
    question: str
    answer: str
    feedback: str
    created_date: datetime = datetime.now()
