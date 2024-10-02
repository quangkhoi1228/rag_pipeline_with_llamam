
from fastapi import APIRouter, HTTPException
from src.config import MAX_FAQ_POOL
from src.database import milvus_db, pg_create_connection
from src.embedding import embedding_query
from src.entity import Chat, ChatResponse, SendChat
from src.faq import (CreateFAQPool, create_faq_pool, get_faq_pool_by_id, random_faq_from_faq_pool,
                     search_faq)
from src.llm import llm_model

router = APIRouter()

# Pydantic model for Chat data


async def create_context(message: str):
    message_embedding = embedding_query(message)

    res = milvus_db.search(
        collection_name="demo_collection",  # target collection
        data=[message_embedding],  # query vectors
        limit=5,  # number of returned entities
        # specifies fields to be returned
        output_fields=["text", "subject", 'url', 'title'],
    )
    context_items = res[0]
    reference = [
        {"url": context_item['entity']['url'],
         'title': context_item['entity']['title']}
        for context_item in context_items]
    context = [f"""post {index}:
               - context {item['entity']['text']}
               - url: {item['entity']['url']}
               ---
               """ for index, item in enumerate(context_items)]
    return [context, reference]


async def answer_with_rag_pipeline(chat: Chat):
    [context, reference] = await create_context(chat.message)

    db_chat_history = await get_chat_history(count=chat.history_count)
    chat_history = [(item.sender, item.message)
                    for item in db_chat_history]

    prompt_formatted = '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question or history of the chat. If you don't know the answer, just say that you don't know. If you know return both answer and reference document (title and document url link) in user language. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        History:{history}
        Answer:

        '''.format(
        question=chat.message, context=context, history=chat_history)

    llm_res = llm_model.invoke(prompt_formatted)
    return [llm_res, reference]


@router.get("/", response_model=list[Chat])
async def get_chat_history(count=20):
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


@router.post("/send", response_model=ChatResponse)
async def send_chat(chat: SendChat):
    reference = []
    faq_id = None
    similar_faq = search_faq(chat.message, 1)

    if len(similar_faq) > 0 and similar_faq[0]['distance'] >= 0.9:
        answer = str(similar_faq[0]['entity']['answer'])
        faq_id = similar_faq[0]['id']
    else:

        [llm_res, reference] = await answer_with_rag_pipeline(chat)
        answer = llm_res

        # save response to FAQ
        # Không nên vì câu hỏi nào của người dùng cũng đưa vào FAQ được
        # faq = FAQ(question=chat.message, answer=answer)
        # create_faq(faq)

        # data null
    user_chat = Chat(message=chat.message, sender='user')
    system_chat = Chat(message=answer, sender='system')

    await create_chat(user_chat)
    await create_chat(system_chat)

    return ChatResponse(response=system_chat, references=reference, faq_id=faq_id)


@router.post("/regenerate", response_model=ChatResponse)
async def regenerate_chat(chat: SendChat):
    context_items = []
    prev_faq_id = chat.faq_id
    faq_id = prev_faq_id
    faq_pool_id = None
    conn, cur = pg_create_connection()
    try:
        faq_pools = await get_faq_pool_by_id(faq_id=prev_faq_id)

        # return random faq from pool if reaching max faq bool, else using rag
        if len(faq_pools) <= MAX_FAQ_POOL - 1:
            [llm_res, context_items] = await answer_with_rag_pipeline(chat)
            answer = llm_res
            # insert faq bool
            faq_pool = await create_faq_pool(CreateFAQPool(faq_id=prev_faq_id, answer=answer))
        else:
            faq_pool = await random_faq_from_faq_pool(faq_id=prev_faq_id)

            answer = faq_pool.answer
        faq_pool_id = faq_pool.id
        # remove 2 lastest chat
        cur.execute(
            """WITH RowsToDelete AS (
                SELECT ctid,
                    ROW_NUMBER() OVER (ORDER BY created_date DESC) AS rn
                FROM chat
            )
            DELETE FROM chat
            WHERE ctid IN (
                SELECT ctid
                FROM RowsToDelete
                WHERE rn <= 2
            );""")
        conn.commit()

        # log history
        user_chat = Chat(message=chat.message, sender='user')
        system_chat = Chat(message=answer, sender='system')

        await create_chat(user_chat)
        await create_chat(system_chat)

        return ChatResponse(response=system_chat, references=context_items, faq_id=faq_id, faq_pool_id=faq_pool_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error regenerate chat: {e}")
    finally:
        cur.close()
        conn.close()


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
