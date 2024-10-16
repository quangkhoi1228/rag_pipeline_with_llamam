from fastapi import APIRouter, HTTPException
from src.config import MAX_FAQ_POOL
from src.database import get_retrieve_context, milvus_db, pg_create_connection
from src.embedding import embedding_query, embedding_query2
from src.entity import Chat, ChatResponse, SendChat, RetrievedDocument
from src.faq import (
    CreateFAQPool,
    create_faq_pool,
    get_faq_pool_by_id,
    random_faq_from_faq_pool,
    search_faq,
)
from src.llm import llm_model
import ollama
from groq import Groq

router = APIRouter()

# Pydantic model for Chat data

client_groq = Groq(
    api_key="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",
)

# Enhancement Fuction start ==========
def llm_completion(system_prompt, user_query):
    chat_completion = client_groq.chat.completions.create(
        messages= [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

def parse_qac_response(response):
    yes_sub = """yes""" ""
    no_sub = """no""" ""
    if yes_sub in response:
        return "yes"
    if no_sub in response:
        return "no"
    return "not found"


def parse_search_query(prompt):
    res = []
    for line in prompt.split("\n"):
        res.append(line.replace("-", "").strip())

    return res

def format_snippets(list_docs: list[RetrievedDocument]):
    text = ""
    for doc in list_docs:
        text += "\n---\n"
        text += doc.content.strip() + "\n\n"
        
    return text

# Enhancement Fuction end =============


async def create_context(message: str):
    message_embedding = embedding_query2(message)

    res = get_retrieve_context(message_embedding)
    context_items = res[0]
    reference = [
        {"url": context_item.url, "title": context_item.title_text}
        for context_item in context_items
    ]
    context = [
        f"""post {index}:
               - context {item.content_text}
               - url: {item.url}
               ---
               """
        for index, item in enumerate(context_items)
    ]
    return [context, reference]


async def answer_with_rag_pipeline(chat: SendChat, version: str = "simple"):
    if version == "simple":
        [context, reference] = await create_context(chat.message)

        db_chat_history = await get_chat_history(count=chat.history_count)
        chat_history = [(item.sender, item.message) for item in db_chat_history]
        prompt_formatted = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question or history of the chat. If you don't know the answer, just say that you don't know. If you know return both answer and reference document (title and document url link) in user language. Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            History:{history}
            Answer:

            """.format(
            question=chat.message, context=context, history=chat_history
        )

        llm_res = llm_model.invoke(prompt_formatted)
        return [llm_res, reference]
    elif version == "enhancement":
        QUESTION_PARSING_SYS_MSG_PROMPT_LINE = """You are a legal assistant. Your task is to analyze questions from the Human.
        Instructions:
        1. You need to analyze the question below and break it down into 3 most correct smallest questions.
        2. Each small question is displayed in a line folowing the format, just give list question only not additional info:
        - <question1>
        - <question2>
        ..."""
        USER_QAC_MESSAGE_TEMPLATE = """'''Cho đoạn văn:
        {d}.
        Câu hỏi: \"{q}\"

        Đoạn văn trên có chứa câu trả lời cho câu hỏi không?'''"""
        QAC_SYS_TEMPLATE = """Bạn được cung cấp một đoạn văn và một câu hỏi, hãy xác định xem đoạn văn có chứa câu trả lời cho câu hỏi không. Response {"is_answer": "yes"} nếu đoạn văn chứa câu trả lời cho câu hỏi.Response {"is_answer": "no"} nếu đoạn văn không chứa câu trả lời cho câu hỏi."""  # noqa: E501
        USER_MSG_TEMPLATE = """Câu hỏi: {q}
        Search result: ```
        {snippet}
        ```"""

        FINAL_SYS_MSG = (
            "<role>\nYou are a legal question answering assistant. Your role is to provide accurate,"
            " relevant and well-structured responses to user queries by leveraging information"
            " from search results.\n\nTo generate a high-quality response:\n\n1. Carefully"
            " analyze the user's question to understand their information needs\n2. Review the"
            " search results returned by the system to identify relevant information \n3."
            " Synthesize the key details from the search results into a coherent, informative"
            " response that directly addresses the user's question\n4. Structure the response"
            " as follows:\n   - Opening (1-3 sentences): Provide a general overview of the"
            " topic without including specific numerical details \n   - Answer (detailed"
            " response): Present the main findings and details from the search results, using"
            " bullet points (-) to clearly list key information and penalties\n   - Citation:"
            " Include references to the search results at the end of the response\n5. Aim for a"
            " response length of 500-1000 words\n6. Write the response in Vietnamese  \n7."
            " Focus only on information directly relevant to answering the question\n8. Avoid"
            ' using generic phrases like "Based on the information found in the related'
            ' documents," "Your question is unclear," or "based on the information from'
            ' [QA-2]"\n\n<response_format>\n # Mở đầu\n...\n\n# Trả lời\n\n...\n</response_format>\n</role>'
        )

        parse_question = llm_completion(system_prompt=QUESTION_PARSING_SYS_MSG_PROMPT_LINE, user_query=message)
        search_query = parse_search_query(parse_question)
        print(search_query)
        contexts = []
        message_embedding = embedding_query2(message)
        res = get_retrieve_context(message_embedding)
        retrieved_documents = [ RetrievedDocument(url = item.url, title = item.title_text, content = item.content_text) for index, item in enumerate(res[0])]
        for doc in retrieved_documents:
            qac_message = llm_completion(system_prompt=QAC_SYS_TEMPLATE, user_query=message)
            if parse_qac_response(qac_res) == "yes":
                contexts.append(doc)
                continue
            else:
                for query in search_query:
                    # When query don't have anything
                    if len(query.strip()) == 0:
                        continue
                        
                    user_query = USER_QAC_MESSAGE_TEMPLATE.format(q=query.strip(), d=doc.content.strip())
                    ### QAC the search response
                    qac_res = llm_completion(system_prompt=QAC_SYS_TEMPLATE, user_query=user_query)
                    if parse_qac_response(qac_res) == "yes":
                        contexts.append(doc)
                        break
                    else:
                        print(parse_qac_response(qac_res))

        print(f'After QAC remains {len(contexts)} docs')
        # if len(contexts) <= 0:
        #     return ["Tôi không có thông tin về  câu hỏi này.", []]
        snippet_text: str = format_snippets(contexts)
        user_msg: str = USER_MSG_TEMPLATE.format(
            q=query,
            snippet=snippet_text
        )

        references = "\n".join([context.url for context in contexts])
        final_response = llm_completion(system_prompt=FINAL_SYS_MSG, user_query=user_msg)
        final_response = f'{final_response} /n References: /n {references}'

        return [final_response, []]


@router.get("/", response_model=list[Chat])
async def get_chat_history(count=20):
    conn, cur = pg_create_connection()
    try:
        cur.execute(f"SELECT * FROM chat ORDER BY created_date DESC LIMIT {count} ")
        chats_data = cur.fetchall()

        chats = [
            Chat(message=row[0], sender=row[1], created_at=row[2]) for row in chats_data
        ]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting all chats: {e}")
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
        raise HTTPException(status_code=500, detail=f"Error creating chat: {e}")
    finally:
        cur.close()
        conn.close()


@router.post("/send", response_model=ChatResponse)
async def send_chat(chat: SendChat, version: str | None = "simple"):
    reference = []
    faq_id = None
    similar_faq = search_faq(chat.message, 1)

    if len(similar_faq) > 0 and similar_faq[0]["distance"] >= 0.9:
        answer = str(similar_faq[0]["entity"]["answer"])
        faq_id = similar_faq[0]["id"]
    else:

        [llm_res, reference] = await answer_with_rag_pipeline(chat, version)
        answer = llm_res

        # save response to FAQ
        # Không nên vì câu hỏi nào của người dùng cũng đưa vào FAQ được
        # faq = FAQ(question=chat.message, answer=answer)
        # create_faq(faq)

        # data null
    user_chat = Chat(message=chat.message, sender="user")
    system_chat = Chat(message=answer, sender="system")

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
            faq_pool = await create_faq_pool(
                CreateFAQPool(faq_id=prev_faq_id, answer=answer)
            )
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
            );"""
        )
        conn.commit()

        # log history
        user_chat = Chat(message=chat.message, sender="user")
        system_chat = Chat(message=answer, sender="system")

        await create_chat(user_chat)
        await create_chat(system_chat)

        return ChatResponse(
            response=system_chat,
            references=context_items,
            faq_id=faq_id,
            faq_pool_id=faq_pool_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerate chat: {e}")
    finally:
        cur.close()
        conn.close()


@router.delete("/clear", response_model=bool)
async def clear_chat():
    conn, cur = pg_create_connection()
    try:
        cur.execute(f"TRUNCATE TABLE chat ")
        conn.commit()

        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clear chat: {e}")
    finally:
        cur.close()
        conn.close()
