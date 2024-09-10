import uuid
from datetime import datetime
import tracemalloc
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.faq import create_faq, search_faq
from src.entity import FAQ, Chat
from src.llm import llm_model
from src.embedding import embedding_query
from src.database import pg_create_connection, milvus_db

router = APIRouter()

# Pydantic model for Chat data


class SendChat(BaseModel):
    message: str = 'Quyết định 720/QĐ-CTN năm 2020'  # Auto-generated UUID
    history_count: int = 6


async def create_context(message: str):
    message_embedding = embedding_query(message)

    res = milvus_db.search(
        collection_name="demo_collection",  # target collection
        data=[message_embedding],  # query vectors
        limit=5,  # number of returned entities
        output_fields=["text", "subject"],  # specifies fields to be returned
    )
    context_items = res[0]
    context = [item['entity']['text'] for item in context_items]
    return context


async def answer_with_rag_pipeline(chat: Chat):
    context = await create_context(chat.message)

    db_chat_history = await get_chat_history(count=chat.history_count)
    chat_history = [(item.sender, item.message)
                    for item in db_chat_history]

    prompt_formatted = '''
      You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question or history of the chat. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
      Question: {question}
      Context: {context}
      History:{history}
      Answer: 

      '''.format(
        question=chat.message, context=context, history=chat_history)

    llm_res = llm_model.invoke(prompt_formatted)
    return llm_res


@router.get("/", response_model=list[Chat])
async def get_chat_history(count=20):
    print(count)
    conn, cur = pg_create_connection()
    try:
        cur.execute(
            f"SELECT * FROM chat ORDER BY created_date DESC LIMIT {count} ")
        chats_data = cur.fetchall()

        chats = [Chat(message=row[0], sender=row[1], created_at=row[2])
                 for row in chats_data]
        return chats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting all chats: {e}")
    finally:
        cur.close()
        conn.close()


async def create_chat(chat: Chat):
    conn, cur = pg_create_connection()
    try:
        # Insert the chat data into the database
        cur.execute(
            "INSERT INTO chat (message, sender, created_date) VALUES (%s, %s, %s)",
            (chat.message, chat.sender, chat.created_date),
        )
        conn.commit()

        return chat
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating chat: {e}")
    finally:
        cur.close()
        conn.close()


@router.post("/send", response_model=Chat)
async def send_chat(chat: SendChat):

    similar_faq = search_faq(chat.message, 1)

    if len(similar_faq) > 0 and similar_faq[0]['distance'] >= 0.9:
        answer = str(similar_faq[0]['entity']['answer'])
    else:

        llm_res = await answer_with_rag_pipeline(chat)
        answer = llm_res
        # save response to FAQ

        faq = FAQ(question=chat.message, answer=answer)
        await create_faq(faq)

    user_chat = Chat(message=chat.message, sender='user')
    system_chat = Chat(message=answer, sender='system')

    await create_chat(user_chat)
    await create_chat(system_chat)

    return system_chat


@router.delete("/clear", response_model=bool)
async def clear_chat():
    conn, cur = pg_create_connection()
    try:
        cur.execute(
            f"TRUNCATE TABLE chat ")
        conn.commit()

        return True
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clear chat: {e}")
    finally:
        cur.close()
        conn.close()

# @router.get("/{chat_id}", response_model=Chat)
# async def get_chat(chat_id: str):
#     conn, cur = pg_create_connection()
#     try:
#         cur.execute("SELECT * FROM chat WHERE id = %s", (chat_id,))
#         chat_data = cur.fetchone()

#         if chat_data is None:
#             raise HTTPException(status_code=404, detail="Chat not found")

#         chat = Chat(**chat_data)
#         return chat
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting chat: {e}")
#     finally:
#         cur.close()
#         conn.close()


# @router.put("/{chat_id}", response_model=Chat)
# async def update_chat(chat_id: str, chat: Chat):
#     conn, cur = pg_create_connection()
#     try:
#         cur.execute(
#             "UPDATE chat SET name = %s WHERE id = %s", (chat.name, chat_id)
#         )
#         conn.commit()

#         cur.execute("SELECT * FROM chat WHERE id = %s", (chat_id,))
#         updated_chat_data = cur.fetchone()

#         if updated_chat_data is None:
#             raise HTTPException(status_code=404, detail="Chat not found")

#         updated_chat = Chat(**updated_chat_data)
#         return updated_chat
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error updating chat: {e}")
#     finally:
#         cur.close()
#         conn.close()


# @router.delete("/{chat_id}")
# async def delete_chat(chat_id: str):
#     conn, cur = pg_create_connection()
#     try:
#         cur.execute("DELETE FROM chat WHERE id = %s", (chat_id,))
#         conn.commit()

#         if cur.rowcount == 0:
#             raise HTTPException(status_code=404, detail="Chat not found")

#         return {"message": "Chat deleted successfully"}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error deleting chat: {e}")
#     finally:
#         cur.close()
#         conn.close()
