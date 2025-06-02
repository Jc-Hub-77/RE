# backend/api/v1/live_trading_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header, Query # Added Query
from sqlalchemy.orm import Session
from typing import List, Optional # Added List, Optional

from backend.schemas import live_trading_schemas # Ensure this schema is defined and has StrategyActionResponse
from backend.services import live_trading_service
from backend.models import User
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
    # TODO: Ensure that current_user.id matches user_id on the subscription, or that user is admin
    result = live_trading_service.deploy_strategy(db, user_strategy_subscription_id)
    if result["status"] == "error":
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
    # TODO: Ensure that current_user.id matches user_id on the subscription, or that user is admin
    result = live_trading_service.stop_strategy(db, user_strategy_subscription_id) # Passed db directly
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/strategies/status", response_model=live_trading_schemas.RunningStrategiesResponse)
async def get_running_strategies_status(
    db: Session = Depends(get_db), # db might be used by service in future
    current_user: User = Depends(get_current_active_user) # Auth for this endpoint
):
    """
    Gets the status of all running strategies. (Currently placeholder, might be restricted or enhanced)
    """
    result = live_trading_service.get_running_strategies_status() # db not used by current service func
    return result


# --- Admin-only Endpoints for Live Strategy Management ---

@router.get("/admin/running-strategies", response_model=live_trading_schemas.RunningStrategiesResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_running_strategies(
    db: Session = Depends(get_db) 
):
    """
    Admin endpoint to list all currently running live strategies.
    """
    return live_trading_service.get_running_strategies_status()

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
