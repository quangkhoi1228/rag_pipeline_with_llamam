from pydantic import BaseModel

class DataUpload(BaseModel):
    question: str
    answer: str
    post_time: int
    url: str
