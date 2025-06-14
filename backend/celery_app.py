from celery import Celery
import os

# Configure Celery
# Using Redis as the broker and result backend
# IMPORTANT: For production, this URL must be set to your actual Redis instance.
REDIS_URL = os.getenv("REDIS_URL", "YOUR_PRODUCTION_REDIS_URL_CHANGE_ME")
# NOTE: For enhanced production safety, consider adding a runtime check here
# to ensure REDIS_URL is not the placeholder value when settings.ENVIRONMENT == "production",
# similar to the critical validations in config.py.

# Ensure broker_connection_retry_on_startup is set if using Celery 5+
# For Celery 4.x, this was broker_connection_retry, broker_connection_max_retries
celery_app = Celery(
    "TradingPlatformTasks", # Changed name for clarity
    broker=REDIS_URL,
    result_backend=REDIS_URL,
    include=['backend.tasks'] # List of modules to import when the worker starts.
)

# Optional configuration, see the Celery user guide for more details.
celery_app.conf.update(
    result_expires=3600, # Time (in seconds) for results to be kept.
    task_track_started=True, # Keep track_started if needed, or remove if not used.
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True # For Celery 5+
)

# Auto-discover tasks in the 'tasks' module (we will create this later)
# celery_app.autodiscover_tasks(['backend.tasks']) # Covered by 'include' in Celery constructor

if __name__ == '__main__':
    celery_app.start()

# Import settings at the end to avoid circular import issues if settings needs celery_app
# This is primarily for the NOTE comment logic, actual check would be more involved.
from backend.config import settings
