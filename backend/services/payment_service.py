# backend/services/payment_service.py
import datetime
from typing import Optional, Dict, Any
import json
import uuid
import os
import logging
from coinbase_commerce.client import Client
from coinbase_commerce.error import SignatureVerificationError, WebhookInvalidPayload
from coinbase_commerce.webhook import Webhook
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models import User, UserStrategySubscription, PaymentTransaction # Adjusted import path
from backend.config import settings # Import global settings
from backend.services import strategy_service # Import strategy service for subscription logic
from backend.services import referral_service # Import referral service for commission processing

# Initialize logger
logger = logging.getLogger(__name__)

# --- Configuration & Setup ---
coinbase_client = None
if settings.COINBASE_COMMERCE_API_KEY:
    try:
        coinbase_client = Client(api_key=settings.COINBASE_COMMERCE_API_KEY)
        logger.info("Coinbase Commerce client initialized.")
    except Exception as e:
        logger.error(f"Error initializing Coinbase Commerce client: {e}. Coinbase Commerce features will be disabled.", exc_info=True)
        coinbase_client = None
else:
    logger.warning("COINBASE_COMMERCE_API_KEY not set in settings. Coinbase Commerce integration will be simulated.")

# --- Payment Gateway Interaction ---
def create_coinbase_commerce_charge(db_session: Session, user_id: int, 
                                   item_id: int, # Can be strategy_id for new sub, or user_strategy_subscription_id for renewal
                                   item_type: str, # e.g., "new_strategy_subscription", "renew_strategy_subscription", "platform_access"
                                   item_name: str, 
                                   item_description: str,
                                   amount_usd: float,
                                   subscription_months: int = 1, # For calculating value if needed
                                   redirect_url: str = None, 
                                   cancel_url: str = None,
                                   # Include necessary metadata for webhook processing, especially for new subscriptions
                                   # For "new_strategy_subscription", metadata MUST include 'api_key_id' and 'custom_parameters_json'.
                                   metadata: Optional[Dict[str, Any]] = None
                                   ):
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user: return {"status": "error", "message": "User not found."}

    internal_transaction_ref = str(uuid.uuid4())

    # Merge provided metadata with essential internal data
    metadata_for_charge = metadata if metadata is not None else {}
    metadata_for_charge.update({
        'internal_transaction_ref': internal_transaction_ref,
        'user_id': str(user_id),
        'item_id': str(item_id),
        'item_type': item_type,
        'subscription_months': str(subscription_months)
    })

    if not coinbase_client:
        logger.info(f"Simulating Coinbase Commerce charge for {item_name}.")
        sim_gateway_charge_id = "sim_charge_cb_" + internal_transaction_ref[:8]
        
        # Create a preliminary PaymentTransaction record with 'pending_gateway_interaction' status
        new_payment = PaymentTransaction(
            internal_reference=internal_transaction_ref, # Use internal_reference for our UUID
            user_id=user_id,
            # Link to subscription if applicable and known at this stage
            user_strategy_subscription_id=item_id if item_type == "renew_strategy_subscription" else None,
            amount_crypto=amount_usd, 
            crypto_currency="USD_PRICED",
            payment_gateway="CoinbaseCommerce_Simulated",
            gateway_transaction_id=sim_gateway_charge_id, 
            status="pending_gateway_interaction",
            created_at=datetime.datetime.utcnow(), 
            updated_at=datetime.datetime.utcnow(),
            description=f"Simulated charge for {item_name}"
        )
        try:
            db_session.add(new_payment)
            db_session.commit()
            db_session.refresh(new_payment)
        except Exception as e:
             db_session.rollback()
             logger.error(f"Error saving simulated payment transaction to DB: {e}", exc_info=True)
             return {"status": "error", "message": "Database error saving simulated payment."}


        return {
            "status": "success_simulated",
            "message": "Simulated Coinbase Commerce charge. Redirect user to payment page.",
            "internal_transaction_ref": internal_transaction_ref,
            "gateway_charge_id": sim_gateway_charge_id,
            "payment_page_url": f"https://commerce.coinbase.com/charges/SIM_{sim_gateway_charge_id}",
            "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat()
        }

    try:
        charge_payload = {
            'name': item_name,
            'description': item_description,
            'local_price': {'amount': f"{amount_usd:.2f}", 'currency': 'USD'},
            'pricing_type': 'fixed_price',
            'redirect_url': redirect_url or settings.APP_PAYMENT_SUCCESS_URL,
            'cancel_url': cancel_url or settings.APP_PAYMENT_CANCEL_URL,
            'metadata': metadata_for_charge
        }
        charge = coinbase_client.charge.create(**charge_payload)

        # Create PaymentTransaction record after successful charge creation
        new_payment = PaymentTransaction(
            internal_transaction_ref=internal_transaction_ref,
            user_id=user_id,
            # Link to subscription if applicable and known at this stage
            user_strategy_subscription_id=item_id if item_type == "renew_strategy_subscription" else None,
            amount_crypto=amount_usd, crypto_currency="USD_PRICED",
            payment_gateway="CoinbaseCommerce",
            gateway_transaction_id=charge.code, 
            status="pending_payment",
            created_at=datetime.datetime.fromisoformat(charge.created_at.replace("Z", "+00:00")),
            updated_at=datetime.datetime.fromisoformat(charge.created_at.replace("Z", "+00:00")),
            description=f"Charge for {item_name} (Gateway ID: {charge.code})"
        )
        try:
            db_session.add(new_payment)
            db_session.commit()
            # db_session.refresh(new_payment) # If needed
        except Exception as e:
             db_session.rollback()
             logger.error(f"Error saving PaymentTransaction after Coinbase charge creation: {e}", exc_info=True)
             # The charge was created, but our DB record failed. This needs alerting/manual fix.
             return {"status": "error", "message": "Payment charge created, but database record failed. Contact support.", "gateway_charge_id": charge.code}


        logger.info(f"Coinbase Commerce charge {charge.code} created for user {user_id} (Internal Ref: {internal_transaction_ref}).")
        return {
            "status": "success",
            "message": "Coinbase Commerce charge created. Redirect user to payment page.",
            "internal_transaction_ref": internal_transaction_ref,
            "gateway_charge_id": charge.code,
            "payment_page_url": charge.hosted_url,
            "expires_at": charge.expires_at
        }
    except Exception as e:
        logger.error(f"Error creating Coinbase Commerce charge: {e}", exc_info=True)
        return {"status": "error", "message": f"Payment gateway error: {str(e)}"}


def handle_coinbase_commerce_webhook(db_session: Session, request_body_str: str, webhook_signature: str):
    if not settings.COINBASE_COMMERCE_WEBHOOK_SECRET:
        logger.critical("COINBASE_COMMERCE_WEBHOOK_SECRET not set in settings. Cannot verify webhook.")
        return {"status": "error", "message": "Webhook secret not configured on server."}, 500

    try:
        event = Webhook.construct_event(request_body_str, webhook_signature, settings.COINBASE_COMMERCE_WEBHOOK_SECRET)
    except WebhookInvalidPayload as e:
        logger.error(f"Webhook Invalid Payload: {e}", exc_info=True)
        return {"status": "error", "message": "Invalid webhook payload."}, 400
    except SignatureVerificationError as e:
        logger.error(f"Webhook Signature Verification Failed: {e}", exc_info=True)
        return {"status": "error", "message": "Webhook signature verification failed."}, 400

    event_type = event.type
    charge_obj_from_webhook = event.data
    gateway_charge_id = charge_obj_from_webhook.code
    internal_ref = charge_obj_from_webhook.metadata.get('internal_transaction_ref')

    logger.info(f"Coinbase Webhook: Type '{event_type}', Charge ID '{gateway_charge_id}', Internal Ref '{internal_ref}'.")

    # Find the payment transaction using either gateway_transaction_id or internal_reference
    payment_transaction = db_session.query(PaymentTransaction).filter(
        (PaymentTransaction.gateway_transaction_id == gateway_charge_id) |
        (PaymentTransaction.internal_reference == internal_ref)
    ).first()

    if not payment_transaction:
        logger.warning(f"Webhook Error: PaymentTransaction not found for Gateway ID {gateway_charge_id} or Internal Ref {internal_ref}")
        # It's often better to return 200 here to avoid webhook retries for unmatchable events
        return {"status": "info", "message": "Transaction not found in DB."}, 200

    if payment_transaction.status == "completed":
        logger.info(f"Webhook Info: Charge {gateway_charge_id} already processed as completed.")
        return {"status": "success", "message": "Already completed."}, 200

    # Update payment transaction status based on webhook event
    new_status = event_type.split(":")[1] if ":" in event_type else event_type # e.g., "confirmed", "failed"
    payment_transaction.status = new_status
    payment_transaction.updated_at = datetime.datetime.utcnow()
    payment_transaction.status_message = f"Webhook event: {event_type}" # Store event type or more details

    try:
        db_session.commit() # Commit status update first
        logger.info(f"Payment {payment_transaction.id} (Gateway: {gateway_charge_id}) status updated to {new_status}.")
    except Exception as e:
        db_session.rollback()
        logger.error(f"DB error updating payment status for {gateway_charge_id}: {e}", exc_info=True)
        return {"status": "error", "message": "DB error updating payment status."}, 500


    # --- Handle Confirmed/Completed Payments ---
    if event_type == "charge:confirmed" or event_type == "charge:completed":
        metadata = charge_obj_from_webhook.metadata
        user_id = int(metadata.get('user_id'))
        item_id = metadata.get('item_id') # This is strategy_id (int) or user_strategy_subscription_id (int)
        item_type = metadata.get('item_type')
        subscription_months = int(metadata.get('subscription_months', 1))
        
        # Extract actual payment amount in USD from webhook data
        payment_amount_usd_str = charge_obj_from_webhook.pricing.get('local', {}).get('amount')
        payment_amount_usd = float(payment_amount_usd_str) if payment_amount_usd_str else 0.0

        # Update PaymentTransaction with actual crypto details if available in webhook
        # payment_details = charge_obj_from_webhook.payments[0] if charge_obj_from_webhook.payments else {}
        # payment_transaction.amount_crypto = payment_details.get('value',{}).get('crypto',{}).get('amount')
        # payment_transaction.crypto_currency = payment_details.get('value',{}).get('crypto',{}).get('currency')
        # db_session.commit() # Commit these details if updated

        if item_type == "new_strategy_subscription":
            # For a new subscription, item_id in metadata is the strategy_db_id.
            # We need api_key_id and custom_parameters from the metadata as well.
            api_key_id = int(metadata.get('api_key_id')) # Assume api_key_id is passed in metadata
            custom_parameters_json = metadata.get('custom_parameters_json', '{}') # Assume params are passed as JSON string
            try:
                custom_parameters = json.loads(custom_parameters_json)
            except json.JSONDecodeError:
                logger.error(f"Webhook Error: Invalid custom_parameters_json in metadata for charge {gateway_charge_id}.", exc_info=True)
                # Log error, potentially set subscription status to error
                custom_parameters = {} # Use empty dict or handle error appropriately

            sub_result = strategy_service.create_or_update_strategy_subscription(
                db_session=db_session,
                user_id=user_id,
                strategy_db_id=int(item_id), # item_id is the strategy_db_id for new subs
                api_key_id=api_key_id,
                custom_parameters=custom_parameters,
                subscription_months=subscription_months
                # payment_transaction_id=payment_transaction.id # Pass our internal payment ID
            )
            if sub_result["status"] == "error":
                 logger.error(f"Webhook Error: Failed to create/update subscription for charge {gateway_charge_id}: {sub_result['message']}", exc_info=True)
                 # Log error, potentially update payment_transaction status to reflect issue
                 payment_transaction.status_message = f"Payment confirmed, but subscription update failed: {sub_result['message']}"
                 db_session.commit() # Commit status message update
                 # Return 200 to Coinbase, but alert admin
                 return {"status": "error", "message": "Payment confirmed, but subscription update failed."}, 200

        elif item_type == "renew_strategy_subscription":
            # For renewal, item_id in metadata is the user_strategy_subscription_id.
            # The create_or_update_strategy_subscription function can find and extend it.
            # It needs the strategy_db_id, api_key_id, and custom_parameters from the existing subscription.
            # A simpler approach is to just update the expiry and active status directly here
            # if create_or_update_strategy_subscription is designed for new subs only.

            # Let's call create_or_update_strategy_subscription, which handles both.
            # It needs strategy_db_id and api_key_id. These should be in metadata for renewal too.
            # Or, fetch the existing subscription by item_id (which is sub ID) and get its details.

            existing_sub = db_session.query(UserStrategySubscription).filter(
                UserStrategySubscription.id == int(item_id), # item_id is the subscription ID for renewal
                UserStrategySubscription.user_id == user_id
            ).first()

            if existing_sub:
                 # Call create_or_update with existing sub details to extend it
                 sub_result = strategy_service.create_or_update_strategy_subscription(
                    db_session=db_session,
                    user_id=user_id,
                    strategy_db_id=existing_sub.strategy_id, # Use strategy ID from existing sub
                    api_key_id=existing_sub.api_key_id, # Use API key ID from existing sub
                    custom_parameters=json.loads(existing_sub.custom_parameters) if isinstance(existing_sub.custom_parameters, str) else existing_sub.custom_parameters, # Use existing params
                    subscription_months=subscription_months
                    # payment_transaction_id=payment_transaction.id
                 )
                 if sub_result["status"] == "error":
                     logger.error(f"Webhook Error: Failed to renew subscription {item_id} for charge {gateway_charge_id}: {sub_result['message']}", exc_info=True)
                     payment_transaction.status_message = f"Payment confirmed, but subscription renewal failed: {sub_result['message']}"
                     db_session.commit()
                     return {"status": "error", "message": "Payment confirmed, but subscription renewal failed."}, 200
            else:
                 logger.error(f"Webhook Error: Renewal requested for non-existent subscription ID {item_id} for user {user_id}.", exc_info=True)
                 payment_transaction.status_message = f"Payment confirmed, but renewal failed: Subscription {item_id} not found."
                 db_session.commit()
                 return {"status": "error", "message": "Payment confirmed, but renewal failed (subscription not found)."}, 200

        # elif item_type == "platform_access": (handle platform-wide subscription activation/renewal)
        # ... other item types

        # After successful payment and subscription update, process for referral commission
        # The charge_obj_from_webhook.metadata should contain 'user_id' of the paying user
        # and 'local_price': {'amount': amount_str, 'currency': 'USD'}
        # We already extracted paying_user_id and payment_amount_usd above.

        # Call referral processing. This service function handles its own DB commit.
        referral_result = referral_service.process_payment_for_referral_commission(
            db_session,
            referred_user_id=user_id, # The user who made the payment is the referred user
            payment_amount_usd=payment_amount_usd
            # payment_transaction_id=payment_transaction.id # Pass our internal payment ID if needed by referral service
        )
        if referral_result["status"] == "error":
             logger.error(f"Webhook Error: Failed to process referral commission for charge {gateway_charge_id}: {referral_result['message']}", exc_info=True)
             # Log error, but don't necessarily return 500 as payment/sub is done.
             # Maybe update payment_transaction status_message with this info.
             payment_transaction.status_message = (payment_transaction.status_message or "") + f" | Referral processing failed: {referral_result['message']}"
             db_session.commit() # Commit status message update

        return {"status": "success", "message": "Payment confirmed, subscription updated, and referral processed (if applicable)."}, 200

    # --- Handle Other Payment Statuses ---
    # For 'charge:failed', 'charge:delayed', 'charge:resolved', etc.
    # The status update was already committed above.
    # You might add logic here to deactivate subscriptions if a payment fails after being pending, etc.
    # For now, just return success as the status update was handled.
    return {"status": "success", "message": f"Payment status updated to {new_status}."}, 200


def get_user_payment_history(db_session: Session, user_id: int, page: int = 1, per_page: int = 10):
    """Retrieves a user's payment history with pagination."""
    payments_query = db_session.query(PaymentTransaction).filter(
        PaymentTransaction.user_id == user_id
    ).order_by(desc(PaymentTransaction.created_at))
    
    total_payments = payments_query.count()
    payments = payments_query.offset((page - 1) * per_page).limit(per_page).all()

    history = [{
        "id": p.id,
        "internal_reference": p.internal_reference,
        "date": p.created_at.isoformat(),
        "description": p.description or f"Transaction ID {p.id}", # Use description if available
        "amount_crypto": p.amount_crypto,
        "crypto_currency": p.crypto_currency,
        "usd_equivalent": p.usd_equivalent,
        "status": p.status,
        "status_message": p.status_message,
        "gateway": p.payment_gateway,
        "gateway_id": p.gateway_transaction_id,
        "subscription_id": p.user_strategy_subscription_id
    } for p in payments]
    
    return {"status": "success", "payment_history": history, "total": total_payments, "page": page, "per_page": per_page, "total_pages": (total_payments + per_page - 1) // per_page if per_page > 0 else 0}

# --- Admin Payment Service Functions ---

def list_all_payment_transactions(db_session: Session, page: int = 1, per_page: int = 20, user_id: Optional[int] = None, status: Optional[str] = None, gateway: Optional[str] = None):
    """Admin function to list all payment transactions with filtering and pagination."""
    query = db_session.query(PaymentTransaction)

    if user_id is not None:
        query = query.filter(PaymentTransaction.user_id == user_id)
    if status:
        query = query.filter(PaymentTransaction.status == status)
    if gateway:
        query = query.filter(PaymentTransaction.payment_gateway == gateway)

    total_transactions = query.count()
    transactions = query.order_by(desc(PaymentTransaction.created_at)).offset((page - 1) * per_page).limit(per_page).all()

    transaction_list = [{
        "id": t.id,
        "internal_reference": t.internal_reference,
        "user_id": t.user_id,
        "date": t.created_at.isoformat(),
        "description": t.description,
        "amount_crypto": t.amount_crypto,
        "crypto_currency": t.crypto_currency,
        "usd_equivalent": t.usd_equivalent,
        "status": t.status,
        "status_message": t.status_message,
        "gateway": t.payment_gateway,
        "gateway_id": t.gateway_transaction_id,
        "subscription_id": t.user_strategy_subscription_id
    } for t in transactions]

    return {"status": "success", "transactions": transaction_list, "total": total_transactions, "page": page, "per_page": per_page, "total_pages": (total_transactions + per_page - 1) // per_page if per_page > 0 else 0}


def get_payment_transaction_by_id(db_session: Session, transaction_id: int):
    """Admin function to view details of a specific payment transaction."""
    transaction = db_session.query(PaymentTransaction).filter(PaymentTransaction.id == transaction_id).first()
    if not transaction:
        return {"status": "error", "message": "Payment transaction not found."}

    return {
        "status": "success",
        "transaction": {
            "id": transaction.id,
            "internal_reference": transaction.internal_reference,
            "user_id": transaction.user_id,
            "date": transaction.created_at.isoformat(),
            "description": transaction.description,
            "amount_crypto": transaction.amount_crypto,
            "crypto_currency": transaction.crypto_currency,
            "usd_equivalent": transaction.usd_equivalent,
            "status": transaction.status,
            "status_message": transaction.status_message,
            "gateway": transaction.payment_gateway,
            "gateway_id": transaction.gateway_transaction_id,
            "subscription_id": transaction.user_strategy_subscription_id,
            "updated_at": transaction.updated_at.isoformat()
        }
    }

# TODO: Admin function to manually update payment status (e.g., for manual payments)
# Placeholder for manual update function
def admin_manual_update_payment_status(db_session: Session, transaction_id: int, new_status: str, status_message: Optional[str] = None):
    """
    Admin function to manually update the status of a payment transaction.
    Use with caution.
    """
    transaction = db_session.query(PaymentTransaction).filter(PaymentTransaction.id == transaction_id).first()
    if not transaction:
        return {"status": "error", "message": "Payment transaction not found."}

    transaction.status = new_status
    if status_message is not None:
        transaction.status_message = status_message
    transaction.updated_at = datetime.datetime.utcnow()

    try:
        db_session.commit()
        logger.info(f"Manually updated payment transaction {transaction_id} status to {new_status}.")
        return {"status": "success", "message": "Payment status updated manually."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error manually updating payment status for transaction {transaction_id}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error manually updating status."}


