# backend/services/live_trading_service.py
import datetime
import time
import ccxt
import json
import os
import importlib.util
import logging
# import datetime # Already imported below, ensure only one
from celery.result import AsyncResult

from sqlalchemy.orm import Session
import datetime # Ensure datetime is imported

from backend.models import UserStrategySubscription, ApiKey, User, Strategy as StrategyModel
from backend.utils import _load_strategy_class_from_db_obj 
from backend.services.exchange_service import _decrypt_data
from backend.config import settings 
from backend.celery_app import celery_app 
from backend.tasks import run_live_strategy 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Service Functions ---
def deploy_strategy(db: Session, user_strategy_subscription_id: int):
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
        user_sub.is_active = False 
        user_sub.status_message = "Stopped: Subscription expired before deployment attempt."
        db.commit()
        logger.warning(f"Subscription ID {user_strategy_subscription_id} has expired. Cannot deploy.")
        return {"status": "error", "message": "Subscription has expired."}

    if user_sub.celery_task_id:
        task_result = AsyncResult(user_sub.celery_task_id, app=celery_app)
        active_states = ['PENDING', 'RECEIVED', 'STARTED', 'RETRY'] 
        terminal_states = ['SUCCESS', 'FAILURE', 'REVOKED', 'CRITICAL'] 

        if task_result.state in active_states:
            logger.info(f"Subscription ID {user_strategy_subscription_id} already has an active Celery task {user_sub.celery_task_id} in state {task_result.state}.")
            return {"status": "info", "message": f"Strategy is already running or queued (Task ID: {user_sub.celery_task_id}, Status: {task_result.state})."}
        elif task_result.state in terminal_states:
            logger.info(f"Previous Celery task {user_sub.celery_task_id} for sub ID {user_strategy_subscription_id} found in terminal state {task_result.state}. Proceeding with new deployment.")
            user_sub.celery_task_id = None 
        else: 
            logger.warning(f"Previous Celery task {user_sub.celery_task_id} for sub ID {user_strategy_subscription_id} in unexpected state {task_result.state}. Proceeding with new deployment.")
            user_sub.celery_task_id = None 
    try:
        task = run_live_strategy.delay(user_strategy_subscription_id)
        user_sub.celery_task_id = task.id
        user_sub.status_message = f"Queued - Task ID: {task.id}" 
        db.commit()
        logger.info(f"Queued strategy deployment task for subscription ID: {user_strategy_subscription_id} with Task ID: {task.id}")
        return {"status": "success", "message": f"Strategy deployment task queued. Task ID: {task.id}", "task_id": task.id} # Added task_id to response

    except Exception as e: 
         logger.error(f"Failed to send Celery task for subscription {user_strategy_subscription_id}: {e}", exc_info=True)
         user_sub.status_message = f"Deployment Error: Failed to queue task - {str(e)[:100]}"
         db.commit()
         return {"status": "error", "message": f"Internal server error during deployment: {e}"}


def stop_strategy(db: Session, user_strategy_subscription_id: int):
    """
    Stops a live trading strategy by revoking its Celery task.
    """
    user_sub = db.query(UserStrategySubscription).filter(
        UserStrategySubscription.id == user_strategy_subscription_id
    ).first()

    if not user_sub:
        return {"status": "error", "message": "Subscription not found."}

    celery_task_id = user_sub.celery_task_id
    if celery_task_id:
        try:
            celery_app.control.revoke(celery_task_id, terminate=True)
            logger.info(f"Sent revoke signal for Celery task ID: {celery_task_id} (Subscription ID: {user_strategy_subscription_id})")
            message = f"Stop signal sent to strategy task {celery_task_id}."
            user_sub.is_active = False 
            user_sub.status_message = f"Stop signal sent at {datetime.datetime.utcnow().isoformat()}"
            user_sub.celery_task_id = None 
            db.commit()
            return {"status": "success", "message": message}
        except Exception as e:
            logger.error(f"Failed to send revoke signal for task {celery_task_id}: {e}", exc_info=True)
            user_sub.status_message = f"Stop Error: Failed to revoke task - {str(e)[:100]}"
            db.commit()
            return {"status": "error", "message": f"Failed to stop strategy task: {e}"}
    else:
        user_sub.is_active = False
        user_sub.status_message = f"Stopped (Task ID not found) at {datetime.datetime.utcnow().isoformat()}"
        db.commit()
        logger.warning(f"No Celery task ID found for subscription {user_strategy_subscription_id}. Updated DB status only.")
        return {"status": "info", "message": "No running task found for this subscription. Status updated in DB."}

def restart_strategy_admin(db: Session, user_strategy_subscription_id: int):
    logger.info(f"Admin attempting to restart strategy for subscription ID: {user_strategy_subscription_id}")

    stop_result = stop_strategy(db, user_strategy_subscription_id)
    
    if stop_result["status"] == "error":
        logger.warning(f"Error during stop phase of restart for sub ID {user_strategy_subscription_id}: {stop_result['message']}")
        # If subscription not found during stop, it definitely won't be found for deploy either.
        if "Subscription not found" in stop_result["message"]: # More robust check
            return {"status": "error", "message": "Cannot restart: Subscription not found."}
        # For other stop errors, we might still attempt deploy if user wants to force it,
        # but it's safer to return error. For now, let's assume we want to proceed cautiously.
        # However, the original logic implies proceeding, so we log and continue.
    elif stop_result["status"] == "info":
        logger.info(f"Stop phase for sub ID {user_strategy_subscription_id}: {stop_result['message']}")
    else: # success
        logger.info(f"Stop phase successful for sub ID {user_strategy_subscription_id}: {stop_result['message']}")

    logger.info(f"Proceeding to deploy phase of restart for sub ID {user_strategy_subscription_id}")
    # Re-fetch subscription to ensure we have the latest state after stop_strategy might have changed it
    user_sub = db.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_strategy_subscription_id).first()
    if not user_sub:
        # This should ideally not happen if stop_strategy didn't return "Subscription not found" error.
        logger.error(f"Subscription {user_strategy_subscription_id} disappeared before deploy phase.")
        return {"status": "error", "message": "Subscription not found prior to deploy phase."}
    
    if user_sub.expires_at and user_sub.expires_at <= datetime.datetime.utcnow():
        logger.warning(f"Subscription ID {user_strategy_subscription_id} is expired. Cannot restart.")
        # Ensure is_active is False if expired, stop_strategy might have already done this.
        if user_sub.is_active: user_sub.is_active = False 
        user_sub.status_message = "Stopped: Subscription expired."
        db.commit()
        return {"status": "error", "message": "Cannot restart: Subscription is expired."}
    
    # Ensure the subscription is marked active before deploying. 
    # stop_strategy sets is_active=False.
    if not user_sub.is_active:
        logger.info(f"Marking subscription {user_strategy_subscription_id} as active before attempting deploy for restart.")
        user_sub.is_active = True
        # Status message will be overwritten by deploy_strategy or error handling below
        user_sub.status_message = "Restarting: Marked active for deploy." 
        db.commit() # Commit the is_active=True change

    deploy_result = deploy_strategy(db, user_strategy_subscription_id)

    if deploy_result["status"] == "error":
        logger.error(f"Error during deploy phase of restart for sub ID {user_strategy_subscription_id}: {deploy_result['message']}")
        # Re-fetch user_sub to update its status_message after deploy failure
        current_sub_after_deploy_fail = db.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_strategy_subscription_id).first()
        if current_sub_after_deploy_fail: 
            # Preserve is_active status from deploy_strategy logic (it might set to false if expired during deploy)
            # but update status_message to reflect restart failure.
            current_sub_after_deploy_fail.status_message = f"Restart failed: Deploy error - {deploy_result['message']}"
            db.commit()
        return {"status": "error", "message": f"Restart failed during deployment: {deploy_result['message']}"}
    
    final_message = f"Restart process initiated for subscription {user_strategy_subscription_id}. Stop status: '{stop_result['message']}'. Deploy status: '{deploy_result['message']}'."
    if deploy_result.get("task_id"): # deploy_strategy now returns task_id on success
        final_message += f" New Task ID: {deploy_result['task_id']}."
            
    logger.info(final_message)
    # The status_message on user_sub should reflect the deploy_strategy outcome ("Queued - Task ID: ...")
    return {"status": "success", "message": final_message, "task_id": deploy_result.get("task_id")}


def get_running_strategies_status(): # Same as original
    logger.info("Fetching running strategy statuses from Celery backend (placeholder)...")
    statuses = []
    try:
        statuses.append({"message": "Status retrieval from Celery backend is a TODO.", "detail": "Implement querying Celery inspect API or rely on DB status."})
    except Exception as e:
        logger.error(f"Error fetching running strategy statuses from Celery: {e}", exc_info=True)
        statuses.append({"message": "Error fetching status from Celery.", "detail": str(e)})
    return {"status": "info", "running_strategies": statuses}


def auto_stop_expired_subscriptions(db: Session): # Same as original
    logger.info("Checking for expired subscriptions to stop...")
    expired_subs = db.query(UserStrategySubscription).filter(
        UserStrategySubscription.is_active == True,
        UserStrategySubscription.expires_at <= datetime.datetime.utcnow()
    ).all()
    for sub in expired_subs:
        logger.info(f"Subscription ID {sub.id} for user {sub.user_id} has expired. Attempting to stop Celery task.")
        stop_response = stop_strategy(db, sub.id)
        logger.info(f"Stop response for sub {sub.id}: {stop_response}")
    if expired_subs:
        logger.info(f"Processed {len(expired_subs)} expired subscriptions for stopping.")

```
