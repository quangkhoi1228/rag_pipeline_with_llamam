from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.entity import FAQ
from src.embedding import embedding_document, embedding_query
from src.database import milvus_db, pg_create_connection

router = APIRouter()


class FAQResponse(BaseModel):
    id: int
    distance: float
    entity: FAQ


class CreateFAQ(BaseModel):
    question: str
    answer: str


@router.get("/", response_model=list[FAQResponse])
def search_faq(message, count=5):
    message_embedding = embedding_query(message)

    res = milvus_db.search(
        collection_name="faq_collection",  # target collection
        data=[message_embedding],  # query vectors
        limit=5,  # number of returned entities
        # specifies fields to be returned
        output_fields=["question", "answer"],
    )
    result_items = res[0]
    print(result_items)

    return result_items


@router.post("/", response_model=FAQ)
def create_faq(faq: FAQ):

    docs_embeddings = embedding_document([faq.question])
    vector = docs_embeddings[0]

    data = [{"question": faq.question, "vector": vector, "answer": faq.answer}]

    res = milvus_db.insert(collection_name="faq_collection", data=data)

    return faq
