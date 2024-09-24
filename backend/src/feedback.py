
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.database import milvus_db, pg_create_connection
from src.embedding import embedding_query
from src.entity import Feedback, SendFeedback
from src.faq import search_faq
from src.llm import llm_model

router = APIRouter()

# Pydantic model for Chat data

# FAQ pool


@router.post("/", response_model=Feedback)
async def send_feedback(create_faq_pool: SendFeedback):

    conn, cur = pg_create_connection()
    try:

        feedback = Feedback(
            faq_pool_id=create_faq_pool.faq_pool_id, faq_id=create_faq_pool.faq_id, feedback=create_faq_pool.feedback)
        cur.execute(
            "INSERT INTO feedback (faq_id,faq_pool_id, feedback, created_date) VALUES (%s, %s,%s, %s)",
            (feedback.faq_id, feedback.faq_pool_id,
             feedback.feedback, feedback.created_date),
        )
        conn.commit()

        return feedback
    except Exception as e:

        raise HTTPException(
            status_code=500, detail=f"Error creating feedback: {e}")
    finally:
        cur.close()
        conn.close()
