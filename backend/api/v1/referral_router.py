# backend/api/v1/referral_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Optional # Adding Optional again just in case

from backend.schemas import referral_schemas, user_schemas # user_schemas for GeneralResponse
from backend.services import referral_service
from backend.models import User
from backend.db import get_db
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes
from backend.dependencies import get_current_active_admin_user # Dependency for admin routes

router = APIRouter()

# --- User-Facing Referral Endpoints (Protected) ---
@router.get("/me/stats", response_model=referral_schemas.UserReferralStatsResponse)
async def get_authenticated_user_referral_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves referral statistics for the currently authenticated user.
    """
    result = referral_service.get_user_referral_stats(db, current_user.id)
    # This service function is expected to always return status: success or error if user not found (shouldn't happen with dependency)
    if result["status"] == "error":
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result

# --- Admin-Facing Referral Endpoints (Admin Protected) ---
@router.get("/", response_model=referral_schemas.AdminReferralListResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_referrals(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("pending_payout", enum=["id", "signed_up", "first_payment", "earned_total", "pending_payout", "paid_out_total", "last_payout", "referrer", "referred"]),
    sort_order: str = Query("desc", enum=["asc", "desc"]),
    referrer_search: Optional[str] = Query(None, description="Search by referrer username"),
    referred_search: Optional[str] = Query(None, description="Search by referred user username")
):
    """
    Lists all referral records for administrators with pagination, sorting, and filtering.
    """
    result = referral_service.list_referrals_for_admin(
        db, page, per_page, sort_by, sort_order, referrer_search, referred_search
    )
    # This service function is expected to always return status: success
    return result

@router.post("/{referral_id}/mark-paid", response_model=referral_schemas.AdminReferralActionResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_mark_referral_commission_paid(
    referral_id: int,
    payout_data: referral_schemas.AdminMarkCommissionPaidRequest,
    db: Session = Depends(get_db)
):
    """
    Admin action to mark a specific amount of pending commission as paid for a referral record.
    """
    result = referral_service.mark_referral_commission_paid_admin(
        db, referral_id, payout_data.amount_paid, payout_data.notes
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

# Note: The endpoint for `process_payment_for_referral_commission` is not included here
# as it's typically triggered internally by a payment processing webhook or service,
# not directly by a user or admin via a standard API call.
