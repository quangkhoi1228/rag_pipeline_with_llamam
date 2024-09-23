from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.util import generate_uuid
from src.entity import FAQ, FAQPool
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


# FAQ pool
@router.post("/pool", response_model=FAQPool)
async def create_faq_pool(faq_pool: FAQPool):
    conn, cur = pg_create_connection()
    try:
        # Generate a unique UUID for the faq_pool ID
        faq_pool.id = generate_uuid()

        # Insert the faq_pool data into the database
        cur.execute(
            "INSERT INTO faq_pool (id, question, answer, created_date) VALUES (%s, %s,%s, %s)",
            (faq_pool.id, faq_pool.question, faq_pool.answer, faq_pool.created_date),
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

        faq_pools = [FAQPool(**faq_pool_data) for faq_pool_data in faq_pools_data]
        return faq_pools
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting all faq_pools: {e}")
    finally:
        cur.close()
        conn.close()


@router.get("/{faq_pool_id}", response_model=FAQPool)
async def get_faq_pool(faq_pool_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute("SELECT * FROM faq_pool WHERE id = %s", (faq_pool_id,))
        faq_pool_data = cur.fetchone()

        if faq_pool_data is None:
            raise HTTPException(status_code=404, detail="FAQPool not found")

        faq_pool = FAQPool(**faq_pool_data)
        return faq_pool
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting faq_pool: {e}")
    finally:
        cur.close()
        conn.close()


# @router.put("/{faq_pool_id}", response_model=FAQPool)
# async def update_faq_pool(faq_pool_id: str, faq_pool: FAQPool):
#     conn, cur = pg_create_connection()
#     try:
#         cur.execute(
#             "UPDATE faq_pool SET name = %s WHERE id = %s", (faq_pool.name, faq_pool_id)
#         )
#         conn.commit()

#         cur.execute("SELECT * FROM faq_pool WHERE id = %s", (faq_pool_id,))
#         updated_faq_pool_data = cur.fetchone()

#         if updated_faq_pool_data is None:
#             raise HTTPException(status_code=404, detail="FAQPool not found")

#         updated_faq_pool = FAQPool(**updated_faq_pool_data)
#         return updated_faq_pool
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error updating faq_pool: {e}")
#     finally:
#         cur.close()
#         conn.close()


@router.delete("/{faq_pool_id}")
async def delete_faq_pool(faq_pool_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute("DELETE FROM faq_pool WHERE id = %s", (faq_pool_id,))
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
