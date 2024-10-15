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

router = APIRouter()

# Pydantic model for Chat data


# Enhancement Fuction start ==========
def llm_completion(system_prompt, user_query):
    messages = [
        {"role": "System", "content": system_prompt},
        {"role": "User", "content": user_query},
    ]
    response = ollama.chat(model="llama3.1:latest", messages=messages)
    return response["message"]["content"]


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

def build_chat_message(
        user_msg: str,
        system_msg: str | None = None,
        previous_conversation: str | None = None,
) -> str:
    messages: list[str] = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    text: str = ""
    for turn in messages:
        turn["content"] = turn["content"].strip()
        text += "<|im_start|>{role}\n{content}<|im_end|>\n".format(**turn)

    text += "<|im_start|>assistant\n"

    if previous_conversation:
        if not previous_conversation.endswith("\n"):
            previous_conversation += "\n"
        text = previous_conversation + text
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
2. Each small question is displayed in a line folowing the format:
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
        answer_sample = '''
        {
            "câu hỏi": "Mức xử phạt đối với hành vi cản trở, gây khó khăn cho việc sử dụng đất của người khác là bao nhiêu tiền?",
            "câu trả lời": """
                Căn cứ theo Điều 15 Nghị định 123/2024/NĐ-CP quy định về cản trở, gây khó khăn cho việc sử dụng đất của người khác như sau:

                    `Điều 15. Cản trở, gây khó khăn cho việc sử dụng đất của người khác

                    1. Phạt tiền từ 1.000.000 đồng đến 3.000.000 đồng đối với hành vi đưa vật liệu xây dựng hoặc các vật khác lên thửa đất thuộc quyền sử dụng của người khác hoặc thửa đất thuộc quyền sử dụng của mình mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

                    2. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi đào bới, xây tường, làm hàng rào trên đất thuộc quyền sử dụng của mình hoặc của người khác mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

                    3. Biện pháp khắc phục hậu quả:

                    Buộc khôi phục lại tình trạng ban đầu của đất trước khi vi phạm.

                Căn cứ tại khoản 2 Điều 5 Nghị định 123/2024/NĐ-CP quy định về mức phạt tiền như sau:

                    Điều 5. Mức phạt tiền và thẩm quyền xử phạt

                    [...]

                    2. Mức phạt tiền quy định tại Chương II của Nghị định này áp dụng đối với cá nhân (trừ khoản 4, 5, 6 Điều 18, khoản 1 Điều 19, điểm b khoản 1 và khoản 4 Điều 20, Điều 22, khoản 2 và khoản 3 Điều 29 Nghị định này). Mức phạt tiền đối với tổ chức bằng 02 lần mức phạt tiền đối với cá nhân có cùng một hành vi vi phạm hành chính.

                    [...]`

                Như vậy, mức xử phạt đối với hành vi cản trở, gây khó khăn cho việc sử dụng đất của người khác cụ thể là:

                - Phạt tiền từ 1.000.000 đồng đến 3.000.000 đồng đối với hành vi đưa vật liệu xây dựng hoặc các vật khác lên thửa đất thuộc quyền sử dụng của người khác hoặc thửa đất thuộc quyền sử dụng của mình mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

                - Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi đào bới, xây tường, làm hàng rào trên đất thuộc quyền sử dụng của mình hoặc của người khác mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

                - Đồng thời, biện pháp khắc phục hậu quả là buộc khôi phục lại tình trạng ban đầu của đất trước khi vi phạm.

                Lưu ý: Mức phạt tiền quy định trên áp dụng đối với cá nhân. Mức phạt tiền đối với tổ chức bằng 02 lần mức phạt tiền đối với cá nhân có cùng một hành vi vi phạm hành chính.
            """
        }, 
        {
            "câu hỏi": "Các phương pháp định giá đất bao gồm những phương pháp nào?",
            "câu trả lời": """
                Căn cứ tại khoản 5 Điều 158 Luật Đất đai 2024 quy định về phương pháp định giá đất như sau:

                (1) Phương pháp so sánh được thực hiện bằng cách điều chỉnh mức giá của các thửa đất có cùng mục đích sử dụng đất, tương đồng nhất định về các yếu tố có ảnh hưởng đến giá đất đã chuyển nhượng trên thị trường, đã trúng đấu giá quyền sử dụng đất mà người trúng đấu giá đã hoàn thành nghĩa vụ tài chính theo quyết định trúng đấu giá thông qua việc phân tích, so sánh các yếu tố ảnh hưởng đến giá đất sau khi đã loại trừ giá trị tài sản gắn liền với đất (nếu có) để xác định giá của thửa đất cần định giá;

                (2) Phương pháp thu nhập được thực hiện bằng cách lấy thu nhập ròng bình quân năm trên một diện tích đất chia cho lãi suất tiền gửi tiết kiệm bình quân của loại tiền gửi bằng tiền Việt Nam kỳ hạn 12 tháng tại các ngân hàng thương mại do Nhà nước nắm giữ trên 50% vốn điều lệ hoặc tổng số cổ phần có quyền biểu quyết trên địa bàn cấp tỉnh của 03 năm liền kề tính đến hết quý gần nhất có số liệu trước thời điểm định giá đất;

                (3) Phương pháp thặng dư được thực hiện bằng cách lấy tổng doanh thu phát triển ước tính trừ đi tổng chi phí phát triển ước tính của thửa đất, khu đất trên cơ sở sử dụng đất có hiệu quả cao nhất (hệ số sử dụng đất, mật độ xây dựng, số tầng cao tối đa của công trình) theo quy hoạch sử dụng đất, quy hoạch chi tiết xây dựng đã được cơ quan có thẩm quyền phê duyệt;

                (4) Phương pháp hệ số điều chỉnh giá đất được thực hiện bằng cách lấy giá đất trong bảng giá đất nhân với hệ số điều chỉnh giá đất. Hệ số điều chỉnh giá đất được xác định thông qua việc so sánh giá đất trong bảng giá đất với giá đất thị trường;

                (5) Chính phủ quy định phương pháp định giá đất khác chưa được quy định tại (1), (2), (3) và (4) sau khi được sự đồng ý của Ủy ban Thường vụ Quốc hội.
            """
        }
        '''

        prompt_template = """
        Bạn đang đóng vai là một luật sư rất giỏi chuyên về luật bất động sản. Bạn đang tư vấn cho khách hàng về vấn đề pháp lý liên quan đến bất động sản.
        Bạn được cung cấp một phần văn bản pháp luật. Trả lời bằng tiếng Việt. 

        Dưới đây là 2 ví dụ về câu hỏi từ khách hàng và cách bạn trả lời: {answer_sample}


        Tên văn bản pháp luật: {document_name}
        Văn bản pháp luật: {document}
        Câu trả lời:
        """

        FINAL_SYS_MSG = prompt_template.format(answer_sample=answer_sample, document_name="", document="")
        parse_question = llm_completion(system_prompt=QUESTION_PARSING_SYS_MSG_PROMPT_LINE, user_query=chat.message)
        search_query = parse_search_query(parse_question)
        contexts = []
        message_embedding = embedding_query2(chat.message)
        res = get_retrieve_context(message_embedding)
        retrieved_documents = [ RetrievedDocument(url = item.url, title = item.title_text, content = item.content_text) for index, item in enumerate(res[0])]
        for doc in retrieved_documents:
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
        
        snippet_text: str = format_snippets(contexts)
        user_msg: str = USER_MSG_TEMPLATE.format(
            q=query,
            snippet=snippet_text
        )
            
        final_response = llm_completion(system_prompt=FINAL_SYS_MSG, user_query=user_msg)
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
