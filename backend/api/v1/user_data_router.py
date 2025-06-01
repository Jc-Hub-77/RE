# backend/api/v1/user_data_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.schemas import user_schemas, live_trading_schemas, payment_schemas # Assuming necessary schemas
from backend.models import User # Assuming User model is needed
from backend.main import get_db # Assuming get_db is needed
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes
from backend.services import user_service, strategy_service, referral_service # Import necessary services

router = APIRouter()

# --- User Dashboard Data Endpoints (Protected) ---
@router.get("/users/{user_id}/performance-summary", response_model=user_schemas.UserPerformanceSummaryResponse) # Assuming a schema
async def get_user_performance_summary(
    user_id: int = Path(..., description="The ID of the user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves performance summary data for a specific user.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this data")

    # Replace simulated data with actual service call
    result = user_service.get_user_performance_summary(db, user_id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result

@router.get("/users/{user_id}/strategy_subscriptions", response_model=user_schemas.UserStrategySubscriptionListResponse) # Assuming a schema
async def get_user_strategy_subscriptions(
    user_id: int = Path(..., description="The ID of the user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves the strategy subscriptions for a specific user.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this data")

    # Replace simulated data with actual service call
    result = strategy_service.list_user_subscriptions(db, user_id)
    # The service function is expected to return a list of subscriptions
    return result


@router.get("/users/{user_id}/platform_subscription", response_model=user_schemas.UserPlatformSubscriptionResponse) # Assuming a schema
async def get_user_platform_subscription(
    user_id: int = Path(..., description="The ID of the user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves the platform subscription details for a specific user.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this data")

    # Replace simulated data with actual service call
    result = user_service.get_user_platform_subscription(db, user_id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result


@router.get("/users/{user_id}/referral-stats", response_model=user_schemas.UserReferralStatsResponse) # Assuming a schema
async def get_user_referral_stats(
    user_id: int = Path(..., description="The ID of the user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves referral statistics for a specific user.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this data")

    # Replace simulated data with actual service call
    result = referral_service.get_user_referral_stats(db, user_id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result
