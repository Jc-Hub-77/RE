# backend/api/v1/strategy_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.schemas.strategy_schemas import (
    StrategyAvailableListResponse,
    StrategyDetailResponse,
    UserStrategySubscriptionCreateRequest,
    UserStrategySubscriptionActionResponse,
    UserStrategySubscriptionListResponse,
    UserStrategySubscriptionDetailResponse, # Added new response model
    UserSubscriptionUpdateParamsRequest # Added for the new endpoint
)
from backend.schemas import user_schemas # Already here for GeneralResponse
from backend.services import strategy_service # Already here
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

@router.get("/subscriptions/me/{subscription_id}", response_model=UserStrategySubscriptionDetailResponse)
async def get_my_specific_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves details for a specific strategy subscription belonging to the current user.
    """
    result = strategy_service.get_user_subscription_details(
        db_session=db,
        user_id=current_user.id,
        subscription_id=subscription_id
    )

    if result["status"] == "error":
        # Consider if "Subscription not found or access denied." should be 403 if user is wrong,
        # but 404 is fine as it obscures whether the sub exists at all from other users.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    
    return result["subscription"] # The service returns the subscription dict directly under this key

@router.post("/subscriptions/me/{subscription_id}/deactivate", response_model=user_schemas.GeneralResponse)
async def deactivate_my_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Deactivates a specific strategy subscription for the currently authenticated user.
    """
    result = strategy_service.deactivate_strategy_subscription(
        db_session=db,
        user_id=current_user.id,
        subscription_id=subscription_id,
        by_admin=False # User is deactivating their own subscription
    )

    if result["status"] == "error":
        if "not found" in result.get("message", "").lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    
    # For "success" or "info" (e.g., "already inactive"), the service response structure
    # (e.g., {"status": "info", "message": "..."}) matches GeneralResponse.
    return result

@router.put("/subscriptions/me/{subscription_id}/parameters", response_model=user_schemas.GeneralResponse)
async def update_my_subscription_parameters(
    subscription_id: int,
    payload: UserSubscriptionUpdateParamsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Updates the custom parameters for a specific, active strategy subscription
    belonging to the currently authenticated user. This involves restarting the strategy.
    """
    result = strategy_service.update_user_subscription_parameters(
        db_session=db,
        user_id=current_user.id,
        subscription_id=subscription_id,
        new_custom_parameters=payload.custom_parameters
    )

    if result["status"] == "error":
        message = result.get("message", "An unexpected error occurred.")
        if "not found" in message.lower() or "access denied" in message.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
        elif "not active" in message.lower() or "expired" in message.lower() or "invalid parameters" in message.lower():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
        else: # For internal errors like failing to stop/start task, or DB errors during the process
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)
            
    # For "success" status, the service returns a dict that matches GeneralResponse
    # e.g. {"status": "success", "message": "...", "new_task_id": "..." }
    # GeneralResponse only has status and message, so new_task_id won't be typed
    # but will be part of the JSON response if the service includes it.
    # If new_task_id is critical for client, a more specific response_model is needed.
    # For now, per plan, GeneralResponse is used.
    return result
