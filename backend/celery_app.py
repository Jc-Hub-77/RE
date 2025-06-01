from celery import Celery
import os

# Configure Celery
# Using Redis as the broker and result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "trading_platform",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Optional configuration, see the Celery user guide for more details.
celery_app.conf.update(
    task_ignore_result=False,
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# Auto-discover tasks in the 'tasks' module (we will create this later)
celery_app.autodiscover_tasks(['backend.tasks'])

if __name__ == '__main__':
    celery_app.start()
