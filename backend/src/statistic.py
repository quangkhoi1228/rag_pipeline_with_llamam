from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time


# Define the task to be run periodically
def scheduled_task():
    print(f"Task executed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    
scheduler = BackgroundScheduler()

# Add a cron job (runs every minute)
scheduler.add_job(scheduled_task, CronTrigger(minute="*/1"))  # Every 1 minute
scheduler.start()