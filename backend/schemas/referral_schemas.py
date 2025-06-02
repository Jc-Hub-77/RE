# backend/schemas/referral_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime

# --- User-Facing Referral Schemas ---
class UserReferralStatsResponse(BaseModel):
    status: str
    message: Optional[str] = None # For errors
    user_id: Optional[int] = None
    referral_code: Optional[str] = None
    total_referrals: Optional[int] = None
    active_referrals: Optional[int] = None # Referrals who made a payment
    total_commission_earned: Optional[float] = None
    pending_commission_payout: Optional[float] = None
    minimum_payout_threshold_usd: Optional[float] = None

# --- Admin-Facing Referral Schemas ---
class AdminReferralView(BaseModel):
    referral_id: int
    referrer_user_id: int
    referrer_username: str
    referred_user_id: int
    referred_username: str
    signed_up_at: Optional[datetime.datetime] = None
    first_payment_at: Optional[datetime.datetime] = None
    is_active_subscriber: bool # True if first_payment_at is not None
    commission_earned_total: float
    commission_pending_payout: float
    commission_paid_out_total: float
    last_payout_date: Optional[datetime.datetime] = None

    class Config:
        model_config = { "from_attributes": True } # Though this is constructed manually in service

class AdminReferralListResponse(BaseModel):
    status: str
    referrals: List[AdminReferralView]
    total_items: int
    page: int
    per_page: int
    total_pages: int
    message: Optional[str] = None # For errors

class AdminMarkCommissionPaidRequest(BaseModel):
    referral_id: int
    amount_paid: float = Field(..., gt=0, description="Amount paid out to the referrer")
    notes: Optional[str] = Field(None, max_length=500, description="Notes for this payout transaction")

class AdminReferralActionResponse(BaseModel): # General response for admin actions
    status: str
    message: str

class ReferralPayoutLogItem(BaseModel):
    log_id: int
    referral_id: int
    admin_user_id: Optional[int] = None
    admin_username: Optional[str] = None
    amount_paid: float
    payout_initiated_at: datetime.datetime
    notes: Optional[str] = None

    class Config:
        model_config = { "from_attributes": True }

class AdminReferralPayoutHistoryResponse(BaseModel):
    status: str = "success" # Default status for successful responses
    items: List[ReferralPayoutLogItem]
    total_items: int
    page: int
    per_page: int
    total_pages: int
    message: Optional[str] = None # For potential errors or info messages

# Schema for the process_payment_for_referral_commission (internal or webhook triggered)
# This might not be directly exposed via an API route in the referral router itself,
# but could be a sub-process called by a payment webhook handler.
# For now, defining it here for completeness if it were to be an input.
class ProcessPaymentReferralRequest(BaseModel):
    referred_user_id: int
    payment_amount_usd: float
    # payment_transaction_id: str # If you link PaymentTransaction to Referral
