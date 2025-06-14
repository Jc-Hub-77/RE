# backend/services/strategy_service.py
import os
import importlib.util
import json
import datetime
import logging # Added logging
from sqlalchemy.orm import Session
from sqlalchemy import desc 
import sqlalchemy.orm # For joinedload

from backend.models import Strategy as StrategyModel, UserStrategySubscription, User, ApiKey 
from backend.config import settings
from backend.services import live_trading_service 
from backend.utils import _load_strategy_class_from_db_obj # Import from utils
from backend.schemas.strategy_schemas import PaymentOption
from typing import Optional, List, Dict, Any
from fastapi import HTTPException # Added for parameter validation, though will return dict for consistency

# Initialize logger
logger = logging.getLogger(__name__)

# STRATEGIES_DIR is now primarily sourced from settings.STRATEGIES_DIR.
# The utils._load_strategy_class_from_db_obj function will use settings.STRATEGIES_DIR.

def list_available_strategies(db_session: Session) -> Dict[str, Any]:
    """Lists all active strategies available to users from the database."""
    try:
        strategies_from_db = db_session.query(StrategyModel).filter(StrategyModel.is_active == True).order_by(StrategyModel.name).all()
        
        available_strategies_data = []
        for s_db in strategies_from_db:
            available_strategies_data.append({
                "id": s_db.id, 
                "name": s_db.name, 
                "description": s_db.description,
                "category": s_db.category, 
                "risk_level": s_db.risk_level,
                "historical_performance_summary": s_db.historical_performance_summary
            })
        logger.info(f"Listed {len(available_strategies_data)} available strategies.")
        return {"status": "success", "strategies": available_strategies_data}
    except Exception as e:
        logger.error(f"Error listing available strategies from DB: {e}", exc_info=True)
        return {"status": "error", "message": "Could not retrieve strategies."}


def get_strategy_details(db_session: Session, strategy_db_id: int) -> Dict[str, Any]:
    """Gets detailed information about a specific strategy from DB, including its parameters."""
    strategy_db_obj = db_session.query(StrategyModel).filter(StrategyModel.id == strategy_db_id, StrategyModel.is_active == True).first()
    if not strategy_db_obj:
        logger.warning(f"Attempt to get details for non-existent or inactive strategy ID {strategy_db_id}.")
        return {"status": "error", "message": "Active strategy not found or ID is invalid."}

    StrategyClass = _load_strategy_class_from_db_obj(strategy_db_obj)
    if not StrategyClass:
        return {"status": "error", "message": f"Could not load strategy class for '{strategy_db_obj.name}'."}
    
    params_def = {}
    default_params_method_name = "get_parameters_definition" 
    try:
        if hasattr(StrategyClass, default_params_method_name) and callable(getattr(StrategyClass, default_params_method_name)):
            method_to_call = getattr(StrategyClass, default_params_method_name)
            params_def = method_to_call() 
        else:
            logger.warning(f"Strategy class {StrategyClass.__name__} does not have '{default_params_method_name}' method. Using DB defaults.")
            params_def = json.loads(strategy_db_obj.default_parameters) if strategy_db_obj.default_parameters else {}
            if not params_def:
                 params_def = {"info": "No parameter definition method found and no default parameters in DB."}
    except Exception as e:
        logger.error(f"Error getting parameter definition for strategy '{strategy_db_obj.name}': {e}", exc_info=True)
        params_def = {"error": f"Could not load parameter definitions: {str(e)}"}

    payment_options_data = []
    if strategy_db_obj.payment_options_json:
        try:
            options_list = json.loads(strategy_db_obj.payment_options_json)
            if isinstance(options_list, list):
                for option_dict in options_list:
                    # Ensure keys match PaymentOption model fields
                    # Pydantic will validate types
                    payment_options_data.append(PaymentOption(**option_dict))
            else:
                logger.warning(f"payment_options_json for strategy {strategy_db_obj.id} is not a list.")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in payment_options_json for strategy {strategy_db_obj.id}.", exc_info=True)
        except Exception as e: # Catch errors during Pydantic model instantiation
            logger.error(f"Error processing payment_options_json for strategy {strategy_db_obj.id}: {e}", exc_info=True)


    details = {
        "id": strategy_db_obj.id, 
        "name": strategy_db_obj.name, 
        "description": strategy_db_obj.description,
        "category": strategy_db_obj.category, 
        "risk_level": strategy_db_obj.risk_level,
        "python_code_path": strategy_db_obj.python_code_path, # Include for admin/debug
        "parameters_definition": params_def, 
        "default_parameters_db": json.loads(strategy_db_obj.default_parameters) if strategy_db_obj.default_parameters else {},
        "payment_options": payment_options_data
    }
    logger.info(f"Fetched details for strategy ID {strategy_db_id}: {strategy_db_obj.name}.")
    return {"status": "success", "details": details}


def create_or_update_strategy_subscription(db_session: Session, user_id: int, strategy_db_id: int, 
                                           api_key_id: int, custom_parameters: dict, 
                                           subscription_months: int = 1):
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user: 
        logger.warning(f"User not found (ID: {user_id}) for subscription.")
        return {"status": "error", "message": "User not found."}
    
    strategy_db_obj = db_session.query(StrategyModel).filter(StrategyModel.id == strategy_db_id, StrategyModel.is_active == True).first()
    if not strategy_db_obj: 
        logger.warning(f"Active strategy not found (ID: {strategy_db_id}) for subscription by user {user_id}.")
        return {"status": "error", "message": "Active strategy not found."}
    
    api_key = db_session.query(ApiKey).filter(ApiKey.id == api_key_id, ApiKey.user_id == user_id).first()
    if not api_key: 
        logger.warning(f"API key not found (ID: {api_key_id}) for user {user_id}.")
        return {"status": "error", "message": "API key not found or does not belong to user."}
    if api_key.status != "active": 
        logger.warning(f"API key (ID: {api_key_id}) is not active for user {user_id}.")
        return {"status": "error", "message": "Selected API key is not active."}

    # Validate custom_parameters before creating/updating subscription
    StrategyClass = _load_strategy_class_from_db_obj(strategy_db_obj)
    if not StrategyClass: # Should have been caught earlier, but double check
        return {"status": "error", "message": f"Could not load strategy class for '{strategy_db_obj.name}'."}

    if hasattr(StrategyClass, 'validate_parameters') and callable(getattr(StrategyClass, 'validate_parameters')):
        try:
            # Validate and get coerced parameters
            validated_custom_parameters = getattr(StrategyClass, 'validate_parameters')(custom_parameters)
            # If validation passes, proceed with validated_custom_parameters
            # The validator raises ValueError on failure, so no need to check 'is_valid is False'
            logger.info(f"Parameters validated successfully for strategy {strategy_db_obj.name}.")
            custom_parameters_to_save = json.dumps(validated_custom_parameters)
        except ValueError as ve: # Catch specific validation errors from strategy
            logger.warning(f"Subscription creation/update: Parameter validation error for strategy {strategy_db_obj.name}: {ve}")
            # Consider raising HTTPException here if this service directly handles HTTP responses
            # e.g., raise HTTPException(status_code=400, detail=f"Invalid parameters: {ve}")
            return {"status": "error", "message": f"Invalid parameters: {ve}"}
        except Exception as e:
            logger.error(f"Subscription creation/update: Unexpected error during parameter validation for strategy {strategy_db_obj.name}: {e}", exc_info=True)
            return {"status": "error", "message": "An unexpected error occurred during parameter validation."}
    else:
        logger.info(f"Strategy {strategy_db_obj.name} has no 'validate_parameters' method. Proceeding with provided custom_parameters as is.")
        custom_parameters_to_save = json.dumps(custom_parameters) # Save original if no validator


    existing_sub = db_session.query(UserStrategySubscription).filter(
        UserStrategySubscription.user_id == user_id,
        UserStrategySubscription.strategy_id == strategy_db_id,
        UserStrategySubscription.api_key_id == api_key_id 
    ).order_by(desc(UserStrategySubscription.expires_at)).first()

    now = datetime.datetime.utcnow()
    action_message = ""
    
    if existing_sub:
        current_expiry = existing_sub.expires_at if existing_sub.expires_at else now
        start_from = max(now, current_expiry) 
        new_expiry = start_from + datetime.timedelta(days=30 * subscription_months) 
        
        existing_sub.expires_at = new_expiry
        existing_sub.is_active = True # Will be set by deploy_strategy logic
        existing_sub.custom_parameters = custom_parameters_to_save # Use validated parameters
        existing_sub.status_message = "Subscription extended and active." # Reset status message
        subscribed_item = existing_sub
        action_message = "Subscription extended"
        logger.info(f"Extending subscription for user {user_id}, strategy {strategy_db_id}, API key {api_key_id}. New expiry: {new_expiry}.")
    else:
        new_expiry = now + datetime.timedelta(days=30 * subscription_months)
        new_subscription = UserStrategySubscription(
            user_id=user_id, strategy_id=strategy_db_id, api_key_id=api_key_id,
            custom_parameters=custom_parameters_to_save, # Use validated parameters
            is_active=False, # Start as inactive, deployment will activate
            subscribed_at=now, expires_at=new_expiry,
            status_message="Subscription created, pending deployment."
        )
        db_session.add(new_subscription)
        subscribed_item = new_subscription
        action_message = "Subscription created"
        logger.info(f"Creating new subscription for user {user_id}, strategy {strategy_db_id}, API key {api_key_id}. Expiry: {new_expiry}.")
    
    try:
        db_session.commit()
        db_session.refresh(subscribed_item)
        
        logger.info(f"Attempting to deploy strategy for subscription ID: {subscribed_item.id}")
        deployment_result = live_trading_service.deploy_strategy(db_session, subscribed_item.id)

        if deployment_result["status"] == "error":
             subscribed_item.status_message = f"Subscription active, but deployment failed: {deployment_result['message']}"
             subscribed_item.is_active = False 
             logger.error(f"Deployment failed for sub ID {subscribed_item.id}: {deployment_result['message']}")
        else:
            subscribed_item.is_active = True # Mark active if deployment was queued
            subscribed_item.status_message = f"Subscription active. Task ID: {deployment_result.get('task_id', 'N/A')}"
            logger.info(f"Deployment successful (queued) for sub ID {subscribed_item.id}. Task ID: {deployment_result.get('task_id')}")
        db_session.commit()

        return {
            "status": "success", 
            "message": f"{action_message} for '{strategy_db_obj.name}'. Status: {subscribed_item.status_message}",
            "subscription_id": subscribed_item.id,
            "expires_at": subscribed_item.expires_at.isoformat()
        }
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error during subscription DB commit for strategy '{strategy_db_obj.name}' (User {user_id}): {e}", exc_info=True)
        return {"status": "error", "message": "Database error during subscription processing."}


def list_user_subscriptions(db_session: Session, user_id: int) -> Dict[str, Any]:
    subscriptions = db_session.query(UserStrategySubscription).filter(
        UserStrategySubscription.user_id == user_id
    ).join(StrategyModel).options(sqlalchemy.orm.joinedload(UserStrategySubscription.strategy)).order_by(desc(UserStrategySubscription.expires_at)).all()

    user_subs_display = []
    now = datetime.datetime.utcnow()
    for sub in subscriptions:
        strategy_info = sub.strategy 
        is_currently_active = sub.is_active and (sub.expires_at > now if sub.expires_at else True)
        
        time_remaining_seconds = 0
        if sub.expires_at and sub.expires_at > now:
            time_remaining_seconds = (sub.expires_at - now).total_seconds()

        user_subs_display.append({
            "subscription_id": sub.id,
            "strategy_id": sub.strategy_id, 
            "strategy_name": strategy_info.name if strategy_info else "Unknown Strategy",
            "api_key_id": sub.api_key_id, 
            "custom_parameters": json.loads(sub.custom_parameters) if isinstance(sub.custom_parameters, str) else sub.custom_parameters,
            "is_active": is_currently_active, # This is the calculated current operational status
            "db_is_active_flag": sub.is_active, # This is the raw DB flag
            "status_message": sub.status_message,
            "subscribed_at": sub.subscribed_at.isoformat() if sub.subscribed_at else None,
            "expires_at": sub.expires_at.isoformat() if sub.expires_at else "Never (or lifetime)",
            "time_remaining_seconds": int(time_remaining_seconds),
            "celery_task_id": sub.celery_task_id
        })
    logger.info(f"Listed {len(user_subs_display)} subscriptions for user ID {user_id}.")
    return {"status": "success", "subscriptions": user_subs_display}


def get_user_subscription_details(db_session: Session, user_id: int, subscription_id: int) -> Dict[str, Any]:
    """
    Retrieves detailed information for a specific user subscription, including strategy name and API key label.
    """
    try:
        subscription = db_session.query(UserStrategySubscription).filter(
            UserStrategySubscription.id == subscription_id,
            UserStrategySubscription.user_id == user_id
        ).options(
            sqlalchemy.orm.joinedload(UserStrategySubscription.strategy),
            sqlalchemy.orm.joinedload(UserStrategySubscription.api_key)
        ).first()

        if not subscription:
            logger.warning(f"Subscription ID {subscription_id} not found for user ID {user_id} or access denied.")
            return {"status": "error", "message": "Subscription not found or access denied."}

        now = datetime.datetime.utcnow()
        
        is_currently_active = subscription.is_active and \
                              (subscription.expires_at > now if subscription.expires_at else True)
        
        time_remaining_seconds = 0
        if subscription.expires_at and subscription.expires_at > now:
            time_remaining_seconds = int((subscription.expires_at - now).total_seconds())

        custom_params_dict = {}
        if subscription.custom_parameters:
            try:
                custom_params_dict = json.loads(subscription.custom_parameters)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse custom_parameters JSON for subscription {subscription.id}: {subscription.custom_parameters}", exc_info=True)
                custom_params_dict = {"error": "Could not parse parameters"}

        formatted_details = {
            "id": subscription.id,
            "user_id": subscription.user_id,
            "strategy_id": subscription.strategy_id,
            "strategy_name": subscription.strategy.name if subscription.strategy else "N/A",
            "api_key_id": subscription.api_key_id,
            "api_key_label": subscription.api_key.label if subscription.api_key else "N/A",
            "custom_parameters": custom_params_dict,
            "is_active": is_currently_active, # Calculated current operational status
            "db_is_active_flag": subscription.is_active, # Raw DB flag
            "status_message": subscription.status_message,
            "subscribed_at": subscription.subscribed_at, # Direct datetime object
            "expires_at": subscription.expires_at, # Direct datetime object
            "time_remaining_seconds": time_remaining_seconds,
            "celery_task_id": subscription.celery_task_id,
            "is_currently_active": is_currently_active # Explicitly added as per schema
        }
        
        logger.info(f"Fetched details for subscription ID {subscription_id} for user ID {user_id}.")
        return {"status": "success", "subscription": formatted_details}

    except Exception as e:
        logger.error(f"Error retrieving subscription details for sub ID {subscription_id}, user ID {user_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


def deactivate_strategy_subscription(db_session: Session, user_id: int, subscription_id: int, by_admin: bool = False):
    """Deactivates a user's strategy subscription."""
    query = db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == subscription_id)
    if not by_admin: # If not admin, ensure user owns the subscription
        query = query.filter(UserStrategySubscription.user_id == user_id)
    
    subscription = query.first()

    if not subscription:
        logger.warning(f"Subscription ID {subscription_id} not found or access denied for user {user_id} (admin: {by_admin}).")
        return {"status": "error", "message": "Subscription not found or access denied."}

    if not subscription.is_active:
        logger.info(f"Subscription ID {subscription_id} is already inactive.")
        return {"status": "info", "message": "Subscription is already inactive."}

    try:
        # Stop the Celery task associated with the subscription
        stop_result = live_trading_service.stop_strategy(db_session, subscription.id) # stop_strategy handles DB updates for task_id and status
        
        if stop_result["status"] == "error":
            logger.error(f"Failed to stop Celery task for subscription ID {subscription_id}: {stop_result['message']}")
            # Proceed to mark as inactive in DB anyway, but log the task stop failure.
            subscription.status_message = f"Deactivation requested, but task stop failed: {stop_result['message']}"
        else:
            subscription.status_message = "Deactivated by user/admin."
            logger.info(f"Celery task for subscription ID {subscription_id} stop signal sent. Status: {stop_result['message']}")
            
        subscription.is_active = False
        # Optionally, can set expires_at to now if deactivation means immediate expiry
        # subscription.expires_at = datetime.datetime.utcnow() 
        
        db_session.commit()
        logger.info(f"Subscription ID {subscription_id} for user {subscription.user_id} deactivated {'by admin' if by_admin else 'by user'}.")
        return {"status": "success", "message": "Subscription deactivated successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error deactivating subscription ID {subscription_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}


def update_user_subscription_parameters(db_session: Session, user_id: int, subscription_id: int, new_custom_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates custom parameters for a user's active strategy subscription.
    This involves stopping the current strategy, updating parameters, and restarting the strategy.
    """
    now = datetime.datetime.utcnow()
    subscription = db_session.query(UserStrategySubscription).filter(
        UserStrategySubscription.id == subscription_id,
        UserStrategySubscription.user_id == user_id
    ).first()

    if not subscription:
        logger.warning(f"Update params: Subscription ID {subscription_id} not found for user ID {user_id}.")
        return {"status": "error", "message": "Subscription not found or access denied."}

    if not subscription.is_active:
        logger.warning(f"Update params: Subscription ID {subscription_id} for user {user_id} is not active.")
        return {"status": "error", "message": "Subscription is not active. Parameters can only be updated for active subscriptions."}

    if subscription.expires_at and subscription.expires_at < now:
        logger.warning(f"Update params: Subscription ID {subscription_id} for user {user_id} has expired.")
        return {"status": "error", "message": "Subscription has expired."}

    # Parameter Validation (Conceptual)
    strategy_db_obj = db_session.query(StrategyModel).filter(StrategyModel.id == subscription.strategy_id).first()
    if not strategy_db_obj:
        # This should ideally not happen if subscription integrity is maintained
        logger.error(f"Update params: Strategy ID {subscription.strategy_id} not found for active subscription {subscription_id}.")
        return {"status": "error", "message": "Associated strategy not found. Cannot validate parameters."}

    StrategyClass = _load_strategy_class_from_db_obj(strategy_db_obj)
    validated_new_custom_parameters = new_custom_parameters # Default to original if no validator
    if StrategyClass and hasattr(StrategyClass, 'validate_parameters') and callable(getattr(StrategyClass, 'validate_parameters')):
        try:
            validated_new_custom_parameters = getattr(StrategyClass, 'validate_parameters')(new_custom_parameters)
            logger.info(f"Parameters validated successfully for strategy {strategy_db_obj.name} during update.")
        except ValueError as ve: # Catch specific validation errors
            logger.warning(f"Update params: Parameter validation error for strategy {strategy_db_obj.name}: {ve}")
            return {"status": "error", "message": f"Invalid parameters: {ve}"}
        except Exception as e:
            logger.error(f"Update params: Unexpected error during parameter validation for strategy {strategy_db_obj.name}: {e}", exc_info=True)
            return {"status": "error", "message": "An unexpected error occurred during parameter validation."}
    else:
        logger.warning(f"Update params: Strategy {strategy_db_obj.name} has no 'validate_parameters' method. Skipping custom validation.")
        # validated_new_custom_parameters remains as new_custom_parameters (original input)

    # Stop the current strategy task
    logger.info(f"Update params: Stopping strategy for subscription ID {subscription_id} (user {user_id}).")
    stop_result = live_trading_service.stop_strategy(db_session=db_session, subscription_id=subscription_id, user_id=user_id)
    
    # stop_strategy should set is_active=False. If it failed but didn't error out (e.g. task not found), we proceed.
    # If stop_strategy itself errored (e.g. DB issue), we'd catch it here if it raises.
    # Assuming stop_strategy returns a dict like {"status": "success/error/info", "message": "..."}
    if stop_result.get("status") == "error":
        # If it's an error that means the task couldn't be signalled to stop, it might be risky to proceed.
        # However, stop_strategy is also responsible for DB updates (is_active=False).
        # Let's assume if status is "error", the state of the subscription might be uncertain or unchanged.
        logger.error(f"Update params: Failed to stop strategy for subscription {subscription_id}. Reason: {stop_result.get('message')}")
        return {"status": "error", "message": f"Failed to stop the current strategy task: {stop_result.get('message')}. Parameters not updated."}

    # Update custom_parameters and status_message
    try:
        subscription.custom_parameters = json.dumps(validated_new_custom_parameters) # Use validated
        subscription.status_message = "Parameters updated, preparing to restart strategy."
        # is_active should have been set to False by stop_strategy. If not, ensure it here.
        subscription.is_active = False 
        db_session.commit()
        db_session.refresh(subscription)
        logger.info(f"Update params: Parameters for subscription {subscription_id} (user {user_id}) updated in DB.")
    except Exception as e:
        db_session.rollback()
        logger.error(f"Update params: DB error updating parameters for subscription {subscription_id}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error while updating parameters."}

    # Restart the strategy task with new parameters
    logger.info(f"Update params: Restarting strategy for subscription ID {subscription_id} (user {user_id}).")
    # deploy_strategy should use the new custom_parameters from the DB
    # It also needs user_id to correctly identify the subscription if it re-fetches.
    deploy_result = live_trading_service.deploy_strategy(db_session=db_session, subscription_id=subscription_id, user_id=user_id)

    if deploy_result.get("status") == "error":
        logger.error(f"Update params: Failed to restart strategy for subscription {subscription_id} after param update. Reason: {deploy_result.get('message')}")
        # The subscription parameters are updated, but the strategy isn't running.
        # The status_message in the subscription should reflect this from deploy_strategy.
        # If deploy_strategy doesn't update status_message on failure, do it here.
        subscription.status_message = f"Parameters updated, but strategy restart failed: {deploy_result.get('message')}"
        subscription.is_active = False # Ensure it's marked as inactive
        try:
            db_session.commit()
        except Exception as e_commit:
            db_session.rollback()
            logger.error(f"Update params: DB error saving post-deploy-failure status for sub {subscription_id}: {e_commit}", exc_info=True)

        return {"status": "error", "message": f"Parameters updated, but failed to restart the strategy: {deploy_result.get('message')}"}

    # Success: deploy_strategy should have set is_active=True and updated celery_task_id and status_message
    logger.info(f"Update params: Strategy for subscription {subscription_id} (user {user_id}) restarted successfully after parameter update.")
    return {
        "status": "success", 
        "message": "Subscription parameters updated and strategy restarted successfully.",
        "subscription_id": subscription_id, # Added for consistency
        "new_task_id": deploy_result.get("celery_task_id") # Or task_id, depends on deploy_strategy response
    }

def admin_update_subscription_details(db_session: Session, subscription_id: int, 
                                   new_status_message: Optional[str] = None, 
                                   new_is_active: Optional[bool] = None,
                                   new_expires_at_str: Optional[str] = None):
    """Admin function to manually update subscription details."""
    subscription = db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == subscription_id).first()
    if not subscription:
        logger.warning(f"Admin: Subscription ID {subscription_id} not found for update.")
        return {"status": "error", "message": "Subscription not found."}

    updated_fields = []
    if new_status_message is not None:
        subscription.status_message = new_status_message
        updated_fields.append("status_message")

    if new_expires_at_str is not None:
        try:
            subscription.expires_at = datetime.datetime.fromisoformat(new_expires_at_str)
            updated_fields.append("expires_at")
        except ValueError:
            return {"status": "error", "message": "Invalid ISO format for new_expires_at_str."}

    # Handle is_active change carefully, as it involves Celery tasks
    if new_is_active is not None and subscription.is_active != new_is_active:
        subscription.is_active = new_is_active
        updated_fields.append("is_active")
        if new_is_active:
            # Deploy the strategy if it's being activated
            logger.info(f"Admin: Activating subscription {subscription_id}, attempting to deploy strategy.")
            deploy_result = live_trading_service.deploy_strategy(db_session, subscription.id)
            if deploy_result["status"] == "error":
                subscription.status_message = (subscription.status_message or "") + f" | Admin activation: Deployment failed: {deploy_result['message']}"
                subscription.is_active = False # Revert if deployment fails
                logger.error(f"Admin: Deployment failed for reactivated subscription {subscription_id}: {deploy_result['message']}")
            else:
                 subscription.status_message = (subscription.status_message or "") + f" | Admin activated. Task ID: {deploy_result.get('task_id')}"
        else:
            # Stop the strategy if it's being deactivated
            logger.info(f"Admin: Deactivating subscription {subscription_id}, attempting to stop strategy.")
            stop_result = live_trading_service.stop_strategy(db_session, subscription.id)
            if stop_result["status"] == "error":
                subscription.status_message = (subscription.status_message or "") + f" | Admin deactivation: Task stop failed: {stop_result['message']}"
                logger.error(f"Admin: Failed to stop task for deactivated subscription {subscription_id}: {stop_result['message']}")
            else:
                 subscription.status_message = (subscription.status_message or "") + " | Admin deactivated."
    
    if not updated_fields:
        return {"status": "info", "message": "No changes provided for subscription."}

    try:
        db_session.commit()
        logger.info(f"Admin: Subscription ID {subscription_id} updated. Changed fields: {', '.join(updated_fields)}.")
        return {"status": "success", "message": f"Subscription details updated for fields: {', '.join(updated_fields)}."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Admin: Error updating subscription ID {subscription_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}
