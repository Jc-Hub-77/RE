# backend/api/v1/live_trading_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.schemas import live_trading_schemas
from backend.services import live_trading_service
from backend.models import User
from backend.db import get_db
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes

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
    result = live_trading_service.stop_strategy(user_strategy_subscription_id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/strategies/status", response_model=live_trading_schemas.RunningStrategiesResponse)
async def get_running_strategies_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Gets the status of all running strategies.
    """
    result = live_trading_service.get_running_strategies_status()
    return result


# --- Admin-only Endpoints for Live Strategy Management ---

from .admin_router import get_current_active_admin_user # Import admin dependency

@router.get("/admin/running-strategies", response_model=live_trading_schemas.RunningStrategiesResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_running_strategies(
    db: Session = Depends(get_db) # db session might be needed for more detailed status later
):
    """
    Admin endpoint to list all currently running live strategies.
    """
    # Re-use the existing service function
    return live_trading_service.get_running_strategies_status()

@router.post("/admin/subscriptions/{user_strategy_subscription_id}/force-stop", response_model=live_trading_schemas.StrategyActionResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_force_stop_live_strategy(
    user_strategy_subscription_id: int,
    db: Session = Depends(get_db) # db session might be needed for status update
):
    """
    Admin endpoint to force-stop a running live trading strategy by subscription ID.
    """
    # Re-use the existing stop_strategy service function
    result = live_trading_service.stop_strategy(db.get_bind()._sessionmaker, user_strategy_subscription_id) # Pass the session factory
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

# TODO: Consider an admin restart endpoint (could call stop then deploy)
