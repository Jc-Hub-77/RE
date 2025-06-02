# backend/services/live_trading_service.py
import datetime
import time
import ccxt
import json
import os
import importlib.util
import logging
import datetime # Import datetime
from celery.result import AsyncResult

from sqlalchemy.orm import Session

from backend.models import UserStrategySubscription, ApiKey, User, Strategy as StrategyModel
from backend.utils import _load_strategy_class_from_db_obj # Import from utils
from backend.services.exchange_service import _decrypt_data
from backend.config import settings # For logging or other configs if needed
from backend.celery_app import celery_app # Import celery app
from backend.tasks import run_live_strategy # Import the Celery task

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s') # Changed to processName for Celery workers
logger = logging.getLogger(__name__)

# --- Service Functions ---
def deploy_strategy(db: Session, user_strategy_subscription_id: int): # Use DB session directly
    """
    Deploys a live trading strategy by sending a task to the Celery queue.
    """
    user_sub = db.query(UserStrategySubscription).filter(
        UserStrategySubscription.id == user_strategy_subscription_id
    ).first()

    if not user_sub:
        logger.error(f"Subscription ID {user_strategy_subscription_id} not found for deployment.")
        return {"status": "error", "message": "Subscription not found."}
    if not user_sub.is_active:
         logger.warning(f"Subscription ID {user_strategy_subscription_id} is not active. Cannot deploy.")
         return {"status": "error", "message": "Subscription is not active."}
    if user_sub.expires_at and user_sub.expires_at <= datetime.datetime.utcnow():
        user_sub.is_active = False # Mark as inactive
        user_sub.status_message = "Stopped: Subscription expired before deployment attempt."
        db.commit()
        logger.warning(f"Subscription ID {user_strategy_subscription_id} has expired. Cannot deploy.")
        return {"status": "error", "message": "Subscription has expired."}

    # Check if a task for this subscription is already running in Celery
    if user_sub.celery_task_id:
        task_result = AsyncResult(user_sub.celery_task_id, app=celery_app)
        # Define states that are considered "active" or "not yet finished"
        active_states = ['PENDING', 'RECEIVED', 'STARTED', 'RETRY']
        # Define states that mean the task is finished and can be replaced
        # CRITICAL is not a standard Celery state, but good to include if used. Standard is FAILURE.
        terminal_states = ['SUCCESS', 'FAILURE', 'REVOKED', 'CRITICAL']

        if task_result.state in active_states:
            logger.info(f"Subscription ID {user_strategy_subscription_id} already has an active Celery task {user_sub.celery_task_id} in state {task_result.state}.")
            return {"status": "info", "message": f"Strategy is already running or queued (Task ID: {user_sub.celery_task_id}, Status: {task_result.state})."}
        elif task_result.state in terminal_states:
            logger.info(f"Previous Celery task {user_sub.celery_task_id} for subscription ID {user_strategy_subscription_id} found in terminal state {task_result.state}. Proceeding with new deployment.")
            user_sub.celery_task_id = None # Clear the old task ID
            # Relying on the later commit to persist this change
        else: # Task state is unknown or something else (e.g. custom state)
            logger.warning(f"Previous Celery task {user_sub.celery_task_id} for subscription ID {user_strategy_subscription_id} in unexpected state {task_result.state}. Caution advised. Proceeding with new deployment.")
            user_sub.celery_task_id = None # Clear the old task ID
            # Relying on the later commit

    try:
        # Send the task to the Celery queue
        # The task will handle fetching subscription details and running the strategy loop
        task = run_live_strategy.delay(user_strategy_subscription_id)

        # Store the task.id in the UserStrategySubscription model
        user_sub.celery_task_id = task.id
        user_sub.status_message = f"Queued - Task ID: {task.id}" # Update status to indicate task is queued
        db.commit()
        logger.info(f"Queued strategy deployment task for subscription ID: {user_strategy_subscription_id} with Task ID: {task.id}")
        return {"status": "success", "message": f"Strategy deployment task queued. Task ID: {task.id}"}

    except Exception as e: # Catch errors during task sending
         logger.error(f"Failed to send Celery task for subscription {user_strategy_subscription_id}: {e}", exc_info=True)
         user_sub.status_message = f"Deployment Error: Failed to queue task - {str(e)[:100]}"
         db.commit()
         return {"status": "error", "message": f"Internal server error during deployment: {e}"}


def stop_strategy(db: Session, user_strategy_subscription_id: int): # Use DB session directly
    """
    Stops a live trading strategy by revoking its Celery task.
    """
    user_sub = db.query(UserStrategySubscription).filter(
        UserStrategySubscription.id == user_strategy_subscription_id
    ).first()

    if not user_sub:
        return {"status": "error", "message": "Subscription not found."}

    # Retrieve the stored celery_task_id from the UserStrategySubscription model
    celery_task_id = user_sub.celery_task_id

    if celery_task_id:
        try:
            # Revoke the task. terminate=True sends SIGTERM, which the task can catch.
            # If the task doesn't handle termination signals, it might finish its current cycle.
            # A more immediate stop might use terminate=True and signal='SIGKILL', but this is harsher.
            celery_app.control.revoke(celery_task_id, terminated=True)
            logger.info(f"Sent revoke signal for Celery task ID: {celery_task_id} (Subscription ID: {user_strategy_subscription_id})")
            message = f"Stop signal sent to strategy task {celery_task_id}."

            # Update DB status immediately
            user_sub.is_active = False # Mark as inactive upon manual stop
            user_sub.status_message = f"Stop signal sent at {datetime.datetime.utcnow().isoformat()}"
            user_sub.celery_task_id = None # Clear the task ID once stop is attempted
            db.commit()

            return {"status": "success", "message": message}

        except Exception as e:
            logger.error(f"Failed to send revoke signal for task {celery_task_id}: {e}", exc_info=True)
            # Update DB status to reflect potential issue
            user_sub.status_message = f"Stop Error: Failed to revoke task - {str(e)[:100]}"
            # Keep the task ID for potential manual intervention/debugging
            db.commit()
            return {"status": "error", "message": f"Failed to stop strategy task: {e}"}
    else:
        # If no task ID is stored, just update the DB status
        user_sub.is_active = False
        user_sub.status_message = f"Stopped (Task ID not found) at {datetime.datetime.utcnow().isoformat()}"
        db.commit()
        logger.warning(f"No Celery task ID found for subscription {user_strategy_subscription_id}. Updated DB status only.")
        return {"status": "info", "message": "No running task found for this subscription. Status updated in DB."}


def get_running_strategies_status():
    """
    Gets the status of running strategies by querying the Celery backend.
    """
    # TODO: Implement querying Celery's active/scheduled/reserved tasks to get status.
    # This can be complex and might be better handled by Celery monitoring tools (like Flower)
    # or by relying primarily on the status stored in the UserStrategySubscription model,
    # which the Celery task updates.

    # For a large number of tasks (5000+), querying all task statuses from the backend
    # frequently might be inefficient. Relying on the DB status updated by the task
    # is generally more scalable for displaying status in the UI.

    # Placeholder implementation:
    logger.info("Fetching running strategy statuses from Celery backend (placeholder)...")
    statuses = []
    try:
        # Example (requires Celery worker and broker running):
        # active_tasks = celery_app.control.inspect().active()
        # scheduled_tasks = celery_app.control.inspect().scheduled()
        # reserved_tasks = celery_app.control.inspect().reserved()

        # Combine and process results to extract relevant info (task ID, state, etc.)
        # Map task IDs back to subscription IDs if possible (requires storing task ID in DB)

        # For now, return a placeholder message
        statuses.append({"message": "Status retrieval from Celery backend is a TODO.", "detail": "Implement querying Celery inspect API or rely on DB status."})

    except Exception as e:
        logger.error(f"Error fetching running strategy statuses from Celery: {e}", exc_info=True)
        statuses.append({"message": "Error fetching status from Celery.", "detail": str(e)})

    return {"status": "info", "running_strategies": statuses}


def auto_stop_expired_subscriptions(db: Session): # Use DB session directly
    """
    Checks for expired subscriptions and attempts to stop their associated Celery tasks.
    """
    logger.info("Checking for expired subscriptions to stop...")
    expired_subs = db.query(UserStrategySubscription).filter(
        UserStrategySubscription.is_active == True,
        UserStrategySubscription.expires_at <= datetime.datetime.utcnow()
    ).all()

    for sub in expired_subs:
        logger.info(f"Subscription ID {sub.id} for user {sub.user_id} has expired. Attempting to stop Celery task.")
        # Call the stop_strategy function, which now revokes the task
        stop_response = stop_strategy(db, sub.id)
        logger.info(f"Stop response for sub {sub.id}: {stop_response}")

    if expired_subs:
        logger.info(f"Processed {len(expired_subs)} expired subscriptions for stopping.")

# Note: Background scheduler for auto_stop_expired_subscriptions would be in main.py or a dedicated app_setup.py
# This scheduler would need to create a DB session to pass to auto_stop_expired_subscriptions.

# TODO: Implement a secure secrets management solution for API_ENCRYPTION_KEY in production.
# This key is critical for decrypting API keys for live trading.
