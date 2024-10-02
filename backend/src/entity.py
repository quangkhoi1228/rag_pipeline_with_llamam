from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator
from src.util import convert_int_to_string


class FeedbackEnum(str, Enum):
    good = "good"
    bad = "bad"


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
    faq_id: str
    faq_pool_id: str
    feedback: str
    created_date: datetime = datetime.now()
    handled: bool = False
    updated_date: datetime = datetime.now()


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
    faq_pool_id: Optional[str] = None

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


class SendFeedback(BaseModel):
    faq_id: Optional[str] = "452739790167339002"
    faq_pool_id: str = '65519a3b-1e9b-4feb-95ea-16a5ba46b37e'  # Auto-generated UUID
    feedback: FeedbackEnum = FeedbackEnum.good


class Statistic(BaseModel):
    faq_id: str
    faq_pool_id: str
    good_count: int
    bad_count: int
    point: int
    question: str
    answer: str
