# backend/api/v1/strategy_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.schemas.strategy_schemas import (
    StrategyAvailableListResponse,
    StrategyDetailResponse,
    UserStrategySubscriptionCreateRequest,
    UserStrategySubscriptionActionResponse,
    UserStrategySubscriptionListResponse
)
from backend.services import strategy_service
from backend.models import User
from backend.db import get_db
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes

router = APIRouter()

# --- Public Strategy Endpoints ---
@router.get("/", response_model=StrategyAvailableListResponse)
async def list_strategies_available_to_users(db: Session = Depends(get_db)):
    """
    Lists all active strategies available for users to view and potentially subscribe to.
    """
    result = strategy_service.list_available_strategies(db)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error retrieving strategies"))
    return result

@router.get("/{strategy_db_id}", response_model=StrategyDetailResponse)
async def get_single_strategy_details(strategy_db_id: int, db: Session = Depends(get_db)):
    """
    Gets detailed information about a specific active strategy, including its parameter definitions.
    """
    result = strategy_service.get_strategy_details(db, strategy_db_id)
    if result["status"] == "error":
        # Distinguish between not found and other errors
        if "not found" in result.get("message", "").lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error retrieving strategy details"))
    return result

# --- User Subscription Endpoints (Protected) ---
@router.post("/subscriptions", response_model=UserStrategySubscriptionActionResponse, status_code=status.HTTP_201_CREATED)
async def create_new_subscription(
    subscription_data: UserStrategySubscriptionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Creates a new strategy subscription for the currently authenticated user.
    This endpoint assumes payment has been handled separately or is not required for this action.
    """
    result = strategy_service.create_or_update_strategy_subscription(
        db_session=db,
        user_id=current_user.id,
        strategy_db_id=subscription_data.strategy_db_id,
        api_key_id=subscription_data.api_key_id,
        custom_parameters=subscription_data.custom_parameters,
        subscription_months=subscription_data.subscription_months
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/subscriptions/me", response_model=UserStrategySubscriptionListResponse)
async def list_my_subscriptions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Lists all strategy subscriptions for the currently authenticated user.
    """
    result = strategy_service.list_user_subscriptions(db, current_user.id)
    # This service function currently always returns success status
    return result

# TODO: Add endpoint to get details of a specific user subscription
# TODO: Add endpoint for user to deactivate/cancel their subscription
# TODO: Add endpoint to update custom_parameters for an existing, active subscription (if allowed)
