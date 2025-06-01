# backend/services/referral_service.py
import datetime
import logging # Added logging
from typing import Optional
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func, desc, or_

from backend.models import User, Referral, PaymentTransaction 
from backend.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

def get_user_referral_stats(db_session: Session, user_id: int):
    """
    Retrieves referral statistics for a given user (who is a referrer).
    """
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"User not found for ID {user_id} when fetching referral stats.")
        return {"status": "error", "message": "User not found."}

    total_referrals_count = db_session.query(func.count(Referral.id)).filter(
        Referral.referrer_user_id == user_id
    ).scalar() or 0

    active_referrals_count = db_session.query(func.count(Referral.id)).filter(
        Referral.referrer_user_id == user_id,
        Referral.first_payment_at != None
    ).scalar() or 0

    total_pending_commission = db_session.query(func.sum(Referral.commission_pending_payout)).filter(
        Referral.referrer_user_id == user_id
    ).scalar() or 0.0
    
    total_commission_earned = db_session.query(func.sum(Referral.commission_earned_total)).filter(
        Referral.referrer_user_id == user_id
    ).scalar() or 0.0

    logger.info(f"Fetched referral stats for user ID {user_id}.")
    return {
        "status": "success",
        "user_id": user_id,
        "referral_code": user.referral_code,
        "total_referrals": total_referrals_count,
        "active_referrals": active_referrals_count,
        "total_commission_earned": round(total_commission_earned, 2),
        "pending_commission_payout": round(total_pending_commission, 2),
        "minimum_payout_threshold_usd": settings.REFERRAL_MINIMUM_PAYOUT_USD
    }

def process_payment_for_referral_commission(db_session: Session, referred_user_id: int, payment_amount_usd: float):
    """
    Processes a successful payment made by a referred user to calculate and assign commission.
    Uses COMMISSION_RATE from settings.
    """
    logger.info(f"Processing payment for potential referral commission. Referred User ID: {referred_user_id}, Payment: ${payment_amount_usd:.2f}")

    referral_record = db_session.query(Referral).filter(Referral.referred_user_id == referred_user_id).first()

    if not referral_record:
        logger.info(f"No referral record found for user ID {referred_user_id}. No commission processed.")
        return {"status": "info", "message": "User was not referred or referral record missing."}

    if referral_record.first_payment_at is not None:
        logger.info(f"Referral ID {referral_record.id}: Commission already processed for first payment.")
        # TODO: Implement logic for recurring commissions if that's a feature based on settings.
        return {"status": "info", "message": "First payment commission already processed for this referral."}

    referral_record.first_payment_at = datetime.datetime.utcnow()
    
    commission_amount = payment_amount_usd * settings.REFERRAL_COMMISSION_RATE
    
    referral_record.commission_earned_total = (referral_record.commission_earned_total or 0.0) + commission_amount
    referral_record.commission_pending_payout = (referral_record.commission_pending_payout or 0.0) + commission_amount
    
    try:
        db_session.commit()
        logger.info(f"Referral ID {referral_record.id}: Commission of ${commission_amount:.2f} processed for referrer {referral_record.referrer_user_id}.")
        # TODO: Notify referrer about earned commission (e.g., via email or in-app notification)
        return {"status": "success", "message": "Referral commission processed."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error updating referral record {referral_record.id} for commission: {e}", exc_info=True)
        return {"status": "error", "message": "Database error processing commission."}


# --- Admin Functions for Referral Management ---
def list_referrals_for_admin(db_session: Session, page: int = 1, per_page: int = 20, 
                             sort_by: str = "pending_payout", 
                             sort_order: str = "desc", 
                             referrer_search: Optional[str] = None, 
                             referred_search: Optional[str] = None):
    """Lists referral records for admin, with sorting and filtering."""
    
    ReferrerUser = aliased(User, name="referrer_user")
    ReferredUser = aliased(User, name="referred_user")

    query = db_session.query(
        Referral, 
        ReferrerUser.username.label("referrer_username"),
        ReferredUser.username.label("referred_username")
    ).join(
        ReferrerUser, Referral.referrer_user_id == ReferrerUser.id
    ).join(
        ReferredUser, Referral.referred_user_id == ReferredUser.id
    )

    if referrer_search:
        query = query.filter(ReferrerUser.username.ilike(f"%{referrer_search}%"))
    if referred_search:
        query = query.filter(ReferredUser.username.ilike(f"%{referred_search}%"))

    sort_column_map = {
        "id": Referral.id,
        "signed_up": Referral.signed_up_at,
        "first_payment": Referral.first_payment_at,
        "earned_total": Referral.commission_earned_total,
        "pending_payout": Referral.commission_pending_payout,
        "paid_out_total": Referral.commission_paid_out_total,
        "last_payout": Referral.last_payout_date,
        "referrer": ReferrerUser.username,
        "referred": ReferredUser.username
    }
    
    sort_attr = sort_column_map.get(sort_by, Referral.commission_pending_payout)
    
    if sort_order.lower() == "desc":
        query = query.order_by(desc(sort_attr))
    else:
        query = query.order_by(sort_attr)
        
    total_referrals = query.count() 
    
    referrals_page_data = query.offset((page - 1) * per_page).limit(per_page).all()

    result_list = []
    for ref, referrer_username, referred_username in referrals_page_data:
        result_list.append({
            "referral_id": ref.id,
            "referrer_user_id": ref.referrer_user_id,
            "referrer_username": referrer_username,
            "referred_user_id": ref.referred_user_id,
            "referred_username": referred_username,
            "signed_up_at": ref.signed_up_at.isoformat() if ref.signed_up_at else None,
            "first_payment_at": ref.first_payment_at.isoformat() if ref.first_payment_at else None,
            "is_active_subscriber": ref.first_payment_at is not None, 
            "commission_earned_total": round(ref.commission_earned_total or 0.0, 2),
            "commission_pending_payout": round(ref.commission_pending_payout or 0.0, 2),
            "commission_paid_out_total": round(ref.commission_paid_out_total or 0.0, 2),
            "last_payout_date": ref.last_payout_date.isoformat() if ref.last_payout_date else None
        })
    
    logger.debug(f"Admin listed referrals page {page}, per_page {per_page}. Found {total_referrals} total.")
    return {
        "status": "success", "referrals": result_list,
        "total_items": total_referrals, "page": page, "per_page": per_page,
        "total_pages": (total_referrals + per_page - 1) // per_page if per_page > 0 else 0
    }

def mark_referral_commission_paid_admin(db_session: Session, referral_id: int, amount_paid: float, notes: Optional[str] = None):
    """Admin action to mark commission as paid for a specific referral record."""
    referral = db_session.query(Referral).filter(Referral.id == referral_id).first()
    if not referral:
        logger.warning(f"Admin: Attempt to mark payout for non-existent referral ID {referral_id}.")
        return {"status": "error", "message": "Referral record not found."}

    if amount_paid <= 0:
        return {"status": "error", "message": "Amount paid must be positive."}
    
    pending_commission = referral.commission_pending_payout or 0.0
    if amount_paid > pending_commission:
        logger.warning(f"Admin: Attempt to pay {amount_paid:.2f} for referral ID {referral_id} but pending is only {pending_commission:.2f}.")
        return {"status": "error", "message": f"Amount paid (${amount_paid:.2f}) exceeds pending commission (${pending_commission:.2f})."}

    referral.commission_pending_payout = pending_commission - amount_paid
    referral.commission_paid_out_total = (referral.commission_paid_out_total or 0.0) + amount_paid
    referral.last_payout_date = datetime.datetime.utcnow()
    
    # TODO: Log this payout action in an admin audit log or separate payout transaction table.
    # Include notes if provided. E.g., create a PayoutLog entry.
    logger.info(f"Admin: Payout of ${amount_paid:.2f} for referral ID {referral_id}. Notes: {notes if notes else 'N/A'}")
    
    try:
        db_session.commit()
        return {"status": "success", "message": "Commission payout recorded successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error marking commission paid for referral {referral_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}

# TODO: Function for admin to adjust commission rates or referral program settings (if stored in DB)
# TODO: Function for admin to view payout history/logs
