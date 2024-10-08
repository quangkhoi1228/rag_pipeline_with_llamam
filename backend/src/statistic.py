from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd


from fastapi import APIRouter, HTTPException

from src.faq import create_faq, delete_faq, delete_faq_pool_by_faq_id, get_faq
from src.entity import FAQ, CreateFAQ, Statistic
from src.database import pg_create_connection
from src.config import STATISTIC_INTERVAL

router = APIRouter()

# Pydantic model for Chat data

# FAQ pool


@router.get("/", response_model=list[Statistic])
async def get_statistic_data():
    conn, cur = pg_create_connection()
    try:

        all_faq = get_faq(1000)
        faq_ids = [f"{str(faq['id'])}" for faq in all_faq]
        # print(faq_ids)
        cur.execute(f"""
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
                    """, (tuple(faq_ids),))
        statistic_point_from_feedback_data = cur.fetchall()

        statistic_data = [Statistic(faq_id=row[0], faq_pool_id=row[1], good_count=row[2], bad_count=row[3], point=row[4], question=row[5], answer=row[6])
                          for row in statistic_point_from_feedback_data]
        return statistic_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting statistic: {e}")
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


async def scheduled_task():
    print(f"Task executed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    await update_faq_from_statistic_data()


# Create an APScheduler instance


def start_scheduler():
    scheduler = AsyncIOScheduler()
    # Add the scheduled task with an interval of 10 seconds
    scheduler.add_job(scheduled_task, "interval", minutes=STATISTIC_INTERVAL)
    scheduler.start()
    return scheduler
