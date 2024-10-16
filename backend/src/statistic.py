import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from groq import Groq
from sklearn.cluster import DBSCAN


from fastapi import APIRouter, HTTPException

from src.chat import answer_with_rag_pipeline
from src.embedding import embedding_document
from src.faq import (
    check_faq_exist,
    create_faq,
    delete_faq,
    delete_faq_pool_by_faq_id,
    get_faq,
)
from src.entity import Chat, CreateFAQ, SendChat, Statistic
from src.database import pg_create_connection
from src.config import STATISTIC_INTERVAL

# For local LLM
# from src.llm import llm_model
import json


router = APIRouter()

# Pydantic model for Chat data

# FAQ pool

client_groq = Groq(
    api_key="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",
)


@router.get("/", response_model=list[Statistic])
async def get_statistic_data():
    conn, cur = pg_create_connection()
    try:

        all_faq = get_faq(1000)
        faq_ids = [f"{str(faq['id'])}" for faq in all_faq]
        # print(faq_ids)
        cur.execute(
            f"""
                    with statistic as (
                    SELECT f.faq_id ,f.faq_pool_id ,
                        SUM(CASE WHEN feedback = 'good' THEN 1 ELSE 0 END) AS good_count,
                        SUM(CASE WHEN feedback = 'bad' THEN 1 ELSE 0 END) AS bad_count,
                        SUM(CASE WHEN feedback = 'good' THEN 1 ELSE 0 END) - 
                        SUM(CASE WHEN feedback = 'bad' THEN 1 ELSE 0 END) AS point
                    FROM feedback f
                    group by f.faq_pool_id , f.faq_id 
                    order by f.faq_id desc, point desc
                    ) 
                    select s.faq_id, s.faq_pool_id,  s.good_count,s.bad_count,s.point, fp.question , fp.answer 
                    from statistic s
                    join faq_pool fp on fp.id = s.faq_pool_id
                    where s.faq_id in %s
                    """,
            (tuple(faq_ids),),
        )
        statistic_point_from_feedback_data = cur.fetchall()

        statistic_data = [
            Statistic(
                faq_id=row[0],
                faq_pool_id=row[1],
                good_count=row[2],
                bad_count=row[3],
                point=row[4],
                question=row[5],
                answer=row[6],
            )
            for row in statistic_point_from_feedback_data
        ]
        return statistic_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistic: {e}")
    finally:
        cur.close()
        conn.close()


@router.patch("/update-faq", response_model=bool)
async def update_faq_from_statistic_data():
    statistic_data = await get_statistic_data()

    top_1 = {}
    for statistic in statistic_data:
        if statistic.faq_id not in top_1:
            top_1[statistic.faq_id] = statistic

        if statistic.point > top_1[statistic.faq_id].point:
            top_1[statistic.faq_id] = statistic

    for top_1_item in top_1.values():
        # print(top_1_item)
        faq = CreateFAQ(question=top_1_item.question, answer=top_1_item.answer)
        delete_faq(top_1_item.faq_id)
        await delete_faq_pool_by_faq_id(top_1_item.faq_id)
        create_faq(faq)
    return True


async def get_user_chat_history():
    conn, cur = pg_create_connection()
    try:
        cur.execute(f"SELECT * FROM chat where sender = 'user' ")
        chats_data = cur.fetchall()

        chats = [
            Chat(message=row[0], sender=row[1], created_at=row[2]) for row in chats_data
        ]
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user chats: {e}")
    finally:
        cur.close()
        conn.close()


def cluster_user_messages(messages):
    question_embeddings = embedding_document(messages)
    # Create a DBSCAN clustering model
    clustering_model = DBSCAN(eps=0.5, min_samples=2, metric="cosine")

    # Fit the model and predict the cluster for each question
    cluster_labels = clustering_model.fit_predict(question_embeddings)

    cluster_dict = {}
    # Print out the clusters
    for i, cluster in enumerate(cluster_labels):
        key = str(cluster)
        if key not in cluster_dict:
            cluster_dict[key] = []

        if messages[i] not in cluster_dict[key]:
            cluster_dict[key].append(messages[i])

    return cluster_dict.values()


def detect_new_faq(clusters):
    system_prompt = """You are an assistant for decision tasks based on the provided questions. Always respond in strict JSON format. Do not provide any explanations or text outside the JSON structure."""
    prompt = """
    Look at all the questions: {questions}. Analyze their content to determine their overall theme and relevance to the subject matter.

    Return a JSON object (not an array) in the following format:
    {{
    "valid": true if all questions are related to law and systems, else return false,
    "question": "The most related question (one of them)  in these provided questions"
    }}
    """
    new_faqs_question = []
    for cluster in clusters:
        prompt_formatted = prompt.format(questions=cluster)
        message = [
            {"role": "system", "content": system_prompt},
            # Set a user message for the assistant to respond to.
            {"role": "user", "content": prompt_formatted},
        ]
        res = (
            client_groq.chat.completions.create(
                messages=message,
                model="llama3-8b-8192",
            )
            .choices[0]
            .message.content.strip()
            .replace("\n", "")
        )
        print(res)
        try:
            json_data = json.loads(res)
            # print(prompt_formatted)
            if (
                json_data["valid"] == True
                and check_faq_exist(json_data["question"]) == False
            ):
                new_faqs_question.append(json_data)
        except Exception as e:
            print(e)
    return new_faqs_question


@router.patch("/widen-faq", response_model=bool)
async def widen_faq_from_user_chat():
    user_history = await get_user_chat_history()
    messages = [chat.message.strip().replace("\n", "") for chat in user_history]
    clusters = cluster_user_messages(messages)
    new_faqs_question = detect_new_faq(clusters)
    print(new_faqs_question)

    for faq_question in new_faqs_question:
        [llm_res, reference] = await answer_with_rag_pipeline(
            SendChat(message=faq_question["question"])
        )
        answer = llm_res
        create_faq(CreateFAQ(question=faq_question["question"], answer=answer))
    return True


async def scheduled_task():
    print(f"Task executed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    await update_faq_from_statistic_data()
    await widen_faq_from_user_chat()


# Create an APScheduler instance


def start_scheduler():
    scheduler = AsyncIOScheduler()
    # Add the scheduled task with an interval of 10 seconds
    scheduler.add_job(scheduled_task, "interval", seconds=STATISTIC_INTERVAL)
    scheduler.start()
    return scheduler
