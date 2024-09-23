from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
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


class SendChat(BaseModel):
    message: str = 'Quyết định 720/QĐ-CTN năm 2020'  # Auto-generated UUID
    history_count: int = 6
    faq_id: Optional[str] = None


class Reference(BaseModel):
    url: str
    title: str  # List of


class ChatResponse(BaseModel):
    response: Chat
    references: list[Reference] = []  # List of
    faq_id: Optional[int] = None

    @field_validator('faq_id')
    def convert_id(cls, v):
        return convert_int_to_string(v)  # Sử dụng hàm chuyển đổi


class FAQResponse(BaseModel):
    id: int
    distance: float
    entity: FAQ

    @field_validator('id')
    def convert_id(cls, v):
        return convert_int_to_string(v)  # Sử dụng hàm chuyển đổi


class CreateFAQ(BaseModel):
    question: str
    answer: str


class CreateFAQPool(BaseModel):
    faq_id: str
    answer: str
