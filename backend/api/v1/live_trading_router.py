# backend/api/v1/live_trading_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header, Query # Added Query
from sqlalchemy.orm import Session
from typing import List, Optional # Added List, Optional

from backend.schemas import live_trading_schemas # Ensure this schema is defined and has StrategyActionResponse
from backend.services import live_trading_service
from backend.models import User, UserStrategySubscription # Added UserStrategySubscription
from backend.db import get_db
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes
from backend.dependencies import get_current_active_admin_user # Corrected import for admin dependency


router = APIRouter()

@router.post("/subscriptions/{user_strategy_subscription_id}/deploy", response_model=live_trading_schemas.StrategyActionResponse)
async def deploy_live_strategy(
    user_strategy_subscription_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Deploys a live trading strategy for a given user subscription.
    """
    # Fetch the subscription object
    subscription = db.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_strategy_subscription_id).first()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found."
        )

    # Authorization check: User must be admin or own the subscription
    if not current_user.is_admin and current_user.id != subscription.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to deploy this strategy subscription."
        )
    
    # Original logic for deploying the strategy
    # The service function might also need the user_id, so we pass current_user.id
    # Ensure live_trading_service.deploy_strategy is adapted if it needs user_id for non-admin cases
    result = live_trading_service.deploy_strategy(
        db_session=db, # Pass db_session as per typical service function signature
        subscription_id=user_strategy_subscription_id,
        user_id=current_user.id # Pass user_id to the service for context
    )
    
    if result["status"] == "error":
        # Consider if specific error messages from deploy_strategy should map to different HTTP status codes
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.post("/subscriptions/{user_strategy_subscription_id}/stop", response_model=live_trading_schemas.StrategyActionResponse)
async def stop_live_strategy(
    user_strategy_subscription_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Stops a running live trading strategy for a given user subscription.
    """
    # Fetch the subscription object
    subscription = db.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_strategy_subscription_id).first()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found."
        )

    # Authorization check: User must be admin or own the subscription
    if not current_user.is_admin and current_user.id != subscription.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to stop this strategy subscription."
        )
        
    # Original logic for stopping the strategy
    # Pass user_id for context, especially if stop_strategy needs to know who initiated for non-admin
    result = live_trading_service.stop_strategy(
        db_session=db, # Pass db_session
        subscription_id=user_strategy_subscription_id,
        user_id=current_user.id # Pass user_id
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/strategies/status", response_model=live_trading_schemas.RunningStrategiesResponse) # Ensure live_trading_schemas is imported
async def get_running_strategies_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    #"""
    #Gets the status of running strategies for the currently authenticated user.
    #"""
    # Pass current_user.id to filter for the user's strategies
    result = live_trading_service.get_running_strategies_status(db=db, user_id=current_user.id)
    return result


# --- Admin-only Endpoints for Live Strategy Management ---

@router.get("/admin/running-strategies", response_model=live_trading_schemas.RunningStrategiesResponse, dependencies=[Depends(get_current_active_admin_user)]) # Ensure get_current_active_admin_user is imported
async def admin_list_running_strategies(
    db: Session = Depends(get_db) 
):
    #"""
    #Admin endpoint to list all currently running live strategies.
    #"""
    # Call without user_id to get all strategies
    return live_trading_service.get_running_strategies_status(db=db)

@router.post("/admin/subscriptions/{user_strategy_subscription_id}/force-stop", response_model=live_trading_schemas.StrategyActionResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_force_stop_live_strategy(
    user_strategy_subscription_id: int,
    db: Session = Depends(get_db) 
):
    """
    Admin endpoint to force-stop a running live trading strategy by subscription ID.
    """
    result = live_trading_service.stop_strategy(db, user_strategy_subscription_id) # Corrected to pass db session directly
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.post("/admin/subscriptions/{user_strategy_subscription_id}/restart", response_model=live_trading_schemas.StrategyActionResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_restart_live_strategy(
    user_strategy_subscription_id: int,
    db: Session = Depends(get_db)
):
    """
    Admin endpoint to restart a live trading strategy.
    This will attempt to stop the existing task (if any) and then deploy a new one.
    """
    result = live_trading_service.restart_strategy_admin(db, user_strategy_subscription_id)
    if result.get("status") == "error": # Use .get() for safer access
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("message", "Unknown error during restart."))
    return result
