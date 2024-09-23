from typing import Any, Optional
from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel, conint, field_validator
from src.util import convert_int_to_string, generate_uuid
from src.entity import FAQ, FAQPool
from src.embedding import embedding_document, embedding_query
from src.database import milvus_db, pg_create_connection

router = APIRouter()


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

    return result_items


@router.get("/", response_model=list[FAQ])
def get_faq(count=5):
    res = milvus_db.query(
        collection_name="faq_collection",  # target collection
        limit=count,  # number of returned entities
        # specifies fields to be returned
        output_fields=["id", "question", "answer"],
    )

    return res


@router.post("/", response_model=FAQ)
def create_faq(faq: FAQ):

    docs_embeddings = embedding_document([faq.question])
    vector = docs_embeddings[0]

    data = [{"question": faq.question, "vector": vector, "answer": faq.answer}]

    res = milvus_db.insert(collection_name="faq_collection", data=data)

    return faq


# FAQ pool
@router.post("/pool", response_model=FAQPool)
async def create_faq_pool(create_faq_pool: CreateFAQPool):

    conn, cur = pg_create_connection()
    try:
        faqs = milvus_db.get(
            collection_name="faq_collection",
            ids=[create_faq_pool.faq_id],
            output_fields=["id", "question", "answer"],
            limit=3
        )

        faq = faqs[0]

        faq_pool = FAQPool(
            id=str(generate_uuid()), faq_id=str(create_faq_pool.faq_id), question=faq['question'],  answer=create_faq_pool.answer)
        # Insert the faq_pool data into the database
        cur.execute(
            "INSERT INTO faq_pool (id,faq_id, question, answer, created_date) VALUES (%s, %s,%s, %s, %s)",
            (faq_pool.id, faq_pool.faq_id, faq_pool.question,
             faq_pool.answer, faq_pool.created_date),
        )
        conn.commit()

        return faq_pool
    except Exception as e:

        raise HTTPException(
            status_code=500, detail=f"Error creating faq_pool: {e}")
    finally:
        cur.close()
        conn.close()


@router.get("/pool", response_model=list[FAQPool])
async def get_all_faq_pools():
    conn, cur = pg_create_connection()
    try:
        cur.execute("SELECT * FROM faq_pool")
        faq_pools_data = cur.fetchall()

        faq_pools = [FAQPool(**faq_pool_data)
                     for faq_pool_data in faq_pools_data]
        return faq_pools
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting all faq_pools: {e}")
    finally:
        cur.close()
        conn.close()


@router.get("/pool/{faq_id}", response_model=list[FAQPool])
async def get_faq_pool_by_id(faq_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute(
            "SELECT * FROM faq_pool WHERE faq_id = %s order by created_date DESC", (faq_id,))
        faq_pool_data = cur.fetchall()

        faq_pools = [FAQPool(id=row[0], faq_id=row[1], question=row[2], answer=row[3], created_date=row[4])
                     for row in faq_pool_data]
        return faq_pools

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting faq_pool: {e}")
    finally:
        cur.close()
        conn.close()


@router.delete("/pool/{faq_id}")
async def delete_faq_pool_by_faq_id(faq_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute("DELETE FROM faq_pool WHERE faq_id = %s", (faq_id,))
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="FAQPool not found")

        return {"message": "FAQPool deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting faq_pool: {e}")
    finally:
        cur.close()
        conn.close()
