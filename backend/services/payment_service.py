# backend/services/payment_service.py
import datetime
from typing import Optional, Dict, Any, List
import json
import uuid
import os
import logging
from coinbase_commerce.client import Client
from coinbase_commerce.error import SignatureVerificationError, WebhookInvalidPayload
from coinbase_commerce.webhook import Webhook
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models import User, UserStrategySubscription, PaymentTransaction
from backend.config import settings
from backend.services import strategy_service
from backend.services import referral_service
from backend.schemas import payment_schemas # Import payment_schemas

# Initialize logger
logger = logging.getLogger(__name__)

VALID_PAYMENT_STATUSES = ['pending', 'pending_gateway_interaction', 'pending_payment', 'confirmed', 'completed', 'failed', 'refunded', 'cancelled', 'expired', 'unresolved', 'charge:created', 'charge:pending', 'charge:confirmed', 'charge:failed', 'charge:delayed', 'charge:resolved']


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

# --- Internal Helper Functions ---
def _process_successful_payment(db_session: Session, payment_transaction: PaymentTransaction, metadata: Dict[str, Any]) -> bool:
    """
    Handles the logic after a payment is confirmed as successful.
    Activates/renews subscriptions and processes referrals.
    """
    logger.info(f"Processing successful payment for transaction ID: {payment_transaction.id}, Internal Ref: {payment_transaction.internal_reference}")

    user_id = int(metadata.get('user_id'))
    item_id_str = metadata.get('item_id') # Can be strategy_id or user_strategy_subscription_id
    item_type = metadata.get('item_type')
    subscription_months = int(metadata.get('subscription_months', 1))
    payment_amount_usd = float(metadata.get('payment_amount_usd', 0.0)) # Ensure this is passed in metadata

    if not all([user_id, item_id_str, item_type, payment_amount_usd > 0]):
        logger.error(f"Missing critical metadata for processing successful payment {payment_transaction.id}. Metadata: {metadata}")
        payment_transaction.status_message = "Processing Error: Missing critical metadata."
        db_session.commit()
        return False

    item_id = int(item_id_str)

    try:
        if item_type == "new_strategy_subscription":
            api_key_id = int(metadata.get('api_key_id'))
            custom_parameters_json = metadata.get('custom_parameters_json', '{}')
            custom_parameters = json.loads(custom_parameters_json)

            sub_result = strategy_service.create_or_update_strategy_subscription(
                db_session=db_session, user_id=user_id, strategy_db_id=item_id,
                api_key_id=api_key_id, custom_parameters=custom_parameters,
                subscription_months=subscription_months, payment_transaction_id=payment_transaction.id
            )
            if sub_result["status"] == "error":
                logger.error(f"Failed to create/update subscription for new payment {payment_transaction.id}: {sub_result['message']}")
                payment_transaction.status_message = f"Subscription creation failed: {sub_result['message']}"
                db_session.commit()
                return False
            payment_transaction.user_strategy_subscription_id = sub_result.get("subscription_id")


        elif item_type == "renew_strategy_subscription":
            existing_sub = db_session.query(UserStrategySubscription).filter(
                UserStrategySubscription.id == item_id,
                UserStrategySubscription.user_id == user_id
            ).first()

            if existing_sub:
                sub_result = strategy_service.create_or_update_strategy_subscription(
                    db_session=db_session, user_id=user_id, strategy_db_id=existing_sub.strategy_id,
                    api_key_id=existing_sub.api_key_id,
                    custom_parameters=json.loads(existing_sub.custom_parameters) if isinstance(existing_sub.custom_parameters, str) else existing_sub.custom_parameters,
                    subscription_months=subscription_months, existing_subscription_id=existing_sub.id,
                    payment_transaction_id=payment_transaction.id
                )
                if sub_result["status"] == "error":
                    logger.error(f"Failed to renew subscription {item_id} for payment {payment_transaction.id}: {sub_result['message']}")
                    payment_transaction.status_message = f"Subscription renewal failed: {sub_result['message']}"
                    db_session.commit()
                    return False
            else:
                logger.error(f"Renewal requested for non-existent subscription ID {item_id} for user {user_id}, payment {payment_transaction.id}.")
                payment_transaction.status_message = f"Renewal failed: Subscription {item_id} not found."
                db_session.commit()
                return False

        # elif item_type == "platform_access": (handle platform-wide subscription activation/renewal)
        else:
            logger.warning(f"Unsupported item_type '{item_type}' for payment {payment_transaction.id}.")
            payment_transaction.status_message = f"Unsupported item type: {item_type}"
            db_session.commit()
            return False # Or True if this is not considered a failure of payment processing itself

        # Process referral commission
        referral_result = referral_service.process_payment_for_referral_commission(
            db_session, referred_user_id=user_id, payment_amount_usd=payment_amount_usd,
            payment_transaction_id=payment_transaction.id
        )
        if referral_result["status"] == "error":
            logger.error(f"Failed to process referral commission for payment {payment_transaction.id}: {referral_result['message']}")
            current_status_message = payment_transaction.status_message or ""
            payment_transaction.status_message = (current_status_message + f" | Referral processing error: {referral_result['message']}").strip(" | ")
            # Not returning False here as primary payment processing (subscription) might have succeeded.

        payment_transaction.status_message = payment_transaction.status_message or "Successfully processed."
        db_session.commit()
        logger.info(f"Successfully processed payment {payment_transaction.id} and associated actions.")
        return True

    except Exception as e:
        logger.error(f"Generic error during _process_successful_payment for transaction {payment_transaction.id}: {e}", exc_info=True)
        payment_transaction.status_message = f"Generic processing error: {str(e)[:100]}"
        db_session.commit()
        return False

# --- Payment Gateway Interaction ---
def create_coinbase_commerce_charge(db_session: Session, user_id: int,
                                   item_id: int,
                                   item_type: str,
                                   item_name: str,
                                   item_description: str,
                                   amount_usd: float,
                                   subscription_months: int = 1,
                                   redirect_url: str = None,
                                   cancel_url: str = None,
                                   metadata: Optional[Dict[str, Any]] = None
                                   ):
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user: return {"status": "error", "message": "User not found."}

    internal_transaction_ref = str(uuid.uuid4())
    metadata_for_charge = metadata if metadata is not None else {}
    metadata_for_charge.update({
        'internal_transaction_ref': internal_transaction_ref,
        'user_id': str(user_id),
        'item_id': str(item_id),
        'item_type': item_type,
        'subscription_months': str(subscription_months),
        'payment_amount_usd': str(amount_usd) # Add payment_amount_usd for _process_successful_payment
    })

    # Store all metadata in description or a dedicated field if model is updated
    # For now, ensure description is comprehensive for manual reconstruction if needed.
    charge_description = f"Charge for {item_name} ({item_type}, Item ID: {item_id}, User: {user_id}, SubMonths: {subscription_months}). Metadata: {json.dumps(metadata_for_charge)}"


    # Create a preliminary PaymentTransaction record
    new_payment = PaymentTransaction(
        internal_reference=internal_transaction_ref,
        user_id=user_id,
        user_strategy_subscription_id=item_id if item_type == "renew_strategy_subscription" else None,
        usd_equivalent=amount_usd, # Store the intended USD amount
        crypto_currency="USD_PRICED", # Indicates this is the target USD value
        payment_gateway="CoinbaseCommerce_Simulated" if not coinbase_client else "CoinbaseCommerce",
        status="pending_gateway_interaction",
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
        description=charge_description, # Store detailed description
        # charge_metadata_json=json.dumps(metadata_for_charge) # IDEAL: if model had this field
    )
    try:
        db_session.add(new_payment)
        db_session.commit()
        db_session.refresh(new_payment)
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error saving initial payment transaction to DB: {e}", exc_info=True)
        return {"status": "error", "message": "Database error saving payment transaction."}

    if not coinbase_client:
        logger.info(f"Simulating Coinbase Commerce charge for {item_name}. Internal Ref: {internal_transaction_ref}")
        sim_gateway_charge_id = "sim_charge_cb_" + internal_transaction_ref[:8]
        new_payment.gateway_transaction_id = sim_gateway_charge_id
        new_payment.status = "pending_payment" # Update status
        db_session.commit()
        return {
            "status": "success_simulated", "message": "Simulated Coinbase Commerce charge.",
            "internal_transaction_ref": internal_transaction_ref, "gateway_charge_id": sim_gateway_charge_id,
            "payment_page_url": f"https://commerce.coinbase.com/charges/SIM_{sim_gateway_charge_id}",
            "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat()
        }

    try:
        charge_payload = {
            'name': item_name, 'description': item_description,
            'local_price': {'amount': f"{amount_usd:.2f}", 'currency': 'USD'},
            'pricing_type': 'fixed_price',
            'redirect_url': redirect_url or settings.APP_PAYMENT_SUCCESS_URL,
            'cancel_url': cancel_url or settings.APP_PAYMENT_CANCEL_URL,
            'metadata': metadata_for_charge
        }
        charge = coinbase_client.charge.create(**charge_payload)

        new_payment.gateway_transaction_id = charge.code
        new_payment.status = "pending_payment" # Charge created, waiting for user payment
        new_payment.created_at = datetime.datetime.fromisoformat(charge.created_at.replace("Z", "+00:00")) # Use actual charge creation time
        new_payment.updated_at = new_payment.created_at
        db_session.commit()

        logger.info(f"Coinbase Commerce charge {charge.code} created for user {user_id} (Internal Ref: {internal_transaction_ref}).")
        return {
            "status": "success", "message": "Coinbase Commerce charge created.",
            "internal_transaction_ref": internal_transaction_ref, "gateway_charge_id": charge.code,
            "payment_page_url": charge.hosted_url, "expires_at": charge.expires_at
        }
    except Exception as e:
        logger.error(f"Error creating Coinbase Commerce charge: {e}", exc_info=True)
        new_payment.status = "failed"; new_payment.status_message = f"Gateway error: {str(e)[:100]}"
        db_session.commit()
        return {"status": "error", "message": f"Payment gateway error: {str(e)}"}


def handle_coinbase_commerce_webhook(db_session: Session, request_body_str: str, webhook_signature: str):
    if not settings.COINBASE_COMMERCE_WEBHOOK_SECRET:
        logger.critical("COINBASE_COMMERCE_WEBHOOK_SECRET not set. Cannot verify webhook.")
        return {"status": "error", "message": "Webhook secret not configured."}, 500

    try:
        event = Webhook.construct_event(request_body_str, webhook_signature, settings.COINBASE_COMMERCE_WEBHOOK_SECRET)
    except (WebhookInvalidPayload, SignatureVerificationError) as e:
        logger.error(f"Webhook validation failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Webhook validation failed: {e}"}, 400

    event_type = event.type
    charge_obj_from_webhook = event.data
    gateway_charge_id = charge_obj_from_webhook.code
    internal_ref = charge_obj_from_webhook.metadata.get('internal_transaction_ref')

    logger.info(f"Coinbase Webhook: Type '{event_type}', Charge ID '{gateway_charge_id}', Internal Ref '{internal_ref}'.")

    payment_transaction = db_session.query(PaymentTransaction).filter(
        (PaymentTransaction.gateway_transaction_id == gateway_charge_id) |
        (PaymentTransaction.internal_reference == internal_ref)
    ).first()

    if not payment_transaction:
        logger.warning(f"Webhook: PaymentTransaction not found for Gateway ID {gateway_charge_id} or Internal Ref {internal_ref}. Ignoring.")
        return {"status": "info", "message": "Transaction not found, webhook ignored."}, 200 # Acknowledge to prevent retries

    if payment_transaction.status == "completed": # Final state, no further processing needed
        logger.info(f"Webhook Info: Charge {gateway_charge_id} (Payment ID: {payment_transaction.id}) already processed as completed.")
        return {"status": "success", "message": "Already completed."}, 200

    # Update payment transaction based on webhook
    # Example: pricing.local.amount, pricing.local.currency for final amount
    # payments[0].value.local.amount for actual payment amount
    # payments[0].value.crypto.amount, payments[0].value.crypto.currency for crypto details

    timeline_statuses = [s['status'].upper() for s in charge_obj_from_webhook.timeline]
    new_status_from_webhook = "pending_payment" # default
    if 'COMPLETED' in timeline_statuses or 'CONFIRMED' in timeline_statuses: # CONFIRMED is usually the key for success
        new_status_from_webhook = "completed"
    elif 'PENDING' in timeline_statuses:
         new_status_from_webhook = "pending_payment" # Could be pending confirmation
    elif 'EXPIRED' in timeline_statuses:
        new_status_from_webhook = "expired"
    elif 'CANCELED' in timeline_statuses: # Note: Coinbase uses CANCELED
        new_status_from_webhook = "cancelled" # Our consistent spelling
    elif 'UNRESOLVED' in timeline_statuses: # e.g. overpaid, underpaid
        new_status_from_webhook = "unresolved"
        if 'UNDERPAID' in timeline_statuses: payment_transaction.status_message = "Underpaid"
        if 'OVERPAID' in timeline_statuses: payment_transaction.status_message = "Overpaid"

    # Update crypto details if available from the webhook
    if charge_obj_from_webhook.payments:
        last_payment_event = charge_obj_from_webhook.payments[-1] # Get the most recent payment event
        if last_payment_event.value and last_payment_event.value.crypto:
            payment_transaction.amount_crypto = float(last_payment_event.value.crypto.amount)
            payment_transaction.crypto_currency = last_payment_event.value.crypto.currency
            logger.info(f"Webhook: Updated crypto payment details for TxID {payment_transaction.id}: {payment_transaction.amount_crypto} {payment_transaction.crypto_currency}")


    payment_transaction.status = new_status_from_webhook
    payment_transaction.updated_at = datetime.datetime.utcnow()
    # status_message can be more specific if needed from webhook details
    if not payment_transaction.status_message or "Webhook event" not in payment_transaction.status_message : # Avoid overwriting specific error messages
        payment_transaction.status_message = payment_transaction.status_message + f" | Webhook event: {event_type}" if payment_transaction.status_message else f"Webhook event: {event_type}"


    try:
        db_session.commit()
        logger.info(f"Payment {payment_transaction.id} (Gateway: {gateway_charge_id}) status updated to {new_status_from_webhook} via webhook.")
    except Exception as e:
        db_session.rollback()
        logger.error(f"DB error updating payment status for {gateway_charge_id} via webhook: {e}", exc_info=True)
        return {"status": "error", "message": "DB error during webhook status update."}, 500

    if new_status_from_webhook == "completed":
        # Pass extracted metadata to the helper function
        # Ensure metadata from charge object is complete
        webhook_metadata = charge_obj_from_webhook.metadata

        # Add payment_amount_usd to metadata if not already there (should be from charge creation)
        if 'payment_amount_usd' not in webhook_metadata:
            payment_amount_usd_str = charge_obj_from_webhook.pricing.get('local', {}).get('amount')
            if payment_amount_usd_str:
                webhook_metadata['payment_amount_usd'] = payment_amount_usd_str # Keep as string, _process converts
            else: # Fallback if not in pricing (should not happen for fixed_price)
                 webhook_metadata['payment_amount_usd'] = str(payment_transaction.usd_equivalent or 0.0)


        if not _process_successful_payment(db_session, payment_transaction, webhook_metadata):
            # Error already logged by _process_successful_payment, status_message updated
            return {"status": "error", "message": "Payment processed, but post-payment actions failed."}, 200 # Acknowledge webhook
        else:
            return {"status": "success", "message": "Payment completed and processed successfully via webhook."}, 200

    return {"status": "success", "message": f"Webhook event {event_type} processed, status set to {new_status_from_webhook}."}, 200


def get_user_payment_history(db_session: Session, user_id: int, page: int = 1, per_page: int = 10):
    payments_query = db_session.query(PaymentTransaction).filter(PaymentTransaction.user_id == user_id).order_by(desc(PaymentTransaction.created_at))
    total_payments = payments_query.count()
    payments = payments_query.offset((page - 1) * per_page).limit(per_page).all()
    history = [{"id": p.id, "internal_reference": p.internal_reference, "date": p.created_at.isoformat(),
                  "description": p.description or f"Transaction ID {p.id}", "amount_crypto": p.amount_crypto,
                  "crypto_currency": p.crypto_currency, "usd_equivalent": p.usd_equivalent, "status": p.status,
                  "status_message": p.status_message, "gateway": p.payment_gateway, "gateway_id": p.gateway_transaction_id,
                  "subscription_id": p.user_strategy_subscription_id} for p in payments]
    return {"status": "success", "payment_history": history, "total": total_payments, "page": page, "per_page": per_page, "total_pages": (total_payments + per_page - 1) // per_page if per_page > 0 else 0}

# --- Admin Payment Service Functions ---
def list_all_payment_transactions(db_session: Session, page: int = 1, per_page: int = 20, user_id: Optional[int] = None, status: Optional[str] = None, gateway: Optional[str] = None):
    query = db_session.query(PaymentTransaction)
    if user_id is not None: query = query.filter(PaymentTransaction.user_id == user_id)
    if status: query = query.filter(PaymentTransaction.status == status)
    if gateway: query = query.filter(PaymentTransaction.payment_gateway == gateway)
    total_transactions = query.count()
    transactions = query.order_by(desc(PaymentTransaction.created_at)).offset((page - 1) * per_page).limit(per_page).all()
    transaction_list = [{"id": t.id, "internal_reference": t.internal_reference, "user_id": t.user_id,
                           "date": t.created_at.isoformat(), "description": t.description, "amount_crypto": t.amount_crypto,
                           "crypto_currency": t.crypto_currency, "usd_equivalent": t.usd_equivalent, "status": t.status,
                           "status_message": t.status_message, "gateway": t.payment_gateway, "gateway_id": t.gateway_transaction_id,
                           "subscription_id": t.user_strategy_subscription_id} for t in transactions]
    return {"status": "success", "transactions": transaction_list, "total": total_transactions, "page": page, "per_page": per_page, "total_pages": (total_transactions + per_page - 1) // per_page if per_page > 0 else 0}

def get_payment_transaction_by_id(db_session: Session, transaction_id: int):
    transaction = db_session.query(PaymentTransaction).filter(PaymentTransaction.id == transaction_id).first()
    if not transaction: return {"status": "error", "message": "Payment transaction not found."}
    return {"status": "success", "transaction": {
            "id": transaction.id, "internal_reference": transaction.internal_reference, "user_id": transaction.user_id,
            "date": transaction.created_at.isoformat(), "description": transaction.description,
            "amount_crypto": transaction.amount_crypto, "crypto_currency": transaction.crypto_currency,
            "usd_equivalent": transaction.usd_equivalent, "status": transaction.status,
            "status_message": transaction.status_message, "gateway": transaction.payment_gateway,
            "gateway_id": transaction.gateway_transaction_id, "subscription_id": transaction.user_strategy_subscription_id,
            "updated_at": transaction.updated_at.isoformat()
        }}

def admin_manual_update_payment_status(db_session: Session, transaction_id: int, update_request: payment_schemas.AdminPaymentStatusUpdateRequest, performing_admin_id: Optional[int] = None) -> Dict[str, Any]:
    transaction = db_session.query(PaymentTransaction).filter(PaymentTransaction.id == transaction_id).first()
    if not transaction:
        return {"status": "error", "message": "Payment transaction not found."}

    if update_request.new_status not in VALID_PAYMENT_STATUSES:
        return {"status": "error", "message": f"Invalid new status '{update_request.new_status}'. Allowed: {', '.join(VALID_PAYMENT_STATUSES)}"}

    old_status = transaction.status
    logger.info(f"Admin (ID: {performing_admin_id or 'Unknown'}) manually updating payment transaction {transaction_id} from status '{old_status}' to '{update_request.new_status}'. Notes: {update_request.admin_notes or 'N/A'}")

    transaction.status = update_request.new_status
    transaction.status_message = update_request.status_message if update_request.status_message is not None else f"Manually updated to {update_request.new_status} by admin {performing_admin_id or 'Unknown'}."
    if update_request.admin_notes:
        transaction.description = (transaction.description or "") + f" | Admin Note ({datetime.datetime.utcnow().isoformat()} by {performing_admin_id or 'Unknown'}): {update_request.admin_notes}"
    transaction.updated_at = datetime.datetime.utcnow()

    processed_successfully = True
    if update_request.new_status == 'completed' and old_status != 'completed':
        logger.info(f"Transaction {transaction_id} marked as 'completed' by admin. Attempting post-payment processing.")

        reconstructed_metadata = {}
        try:
            # Attempt to reconstruct metadata. This is highly dependent on how it was stored.
            # Ideal: transaction.charge_metadata_json (if model was updated)
            # Fallback: parse from transaction.description
            if transaction.description:
                # This is a basic attempt; real parsing might be more complex
                desc_metadata_prefix = "Metadata: {"
                if desc_metadata_prefix in transaction.description:
                    metadata_json_str = transaction.description[transaction.description.find(desc_metadata_prefix) + len(desc_metadata_prefix)-1:]
                    # Ensure closing brace is found and extract
                    brace_level = 0
                    end_index = -1
                    for i, char in enumerate(metadata_json_str):
                        if char == '{': brace_level += 1
                        elif char == '}': brace_level -=1
                        if brace_level == 0 and char == '}': end_index = i; break
                    if end_index != -1:
                        metadata_json_str = metadata_json_str[:end_index+1]
                        try:
                            reconstructed_metadata = json.loads(metadata_json_str)
                            logger.info(f"Reconstructed metadata for TxID {transaction.id} from description: {reconstructed_metadata}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse metadata from description for TxID {transaction.id}: {e}. Description: {metadata_json_str}")
                            reconstructed_metadata = {} # Ensure it's a dict
                    else: # Fallback if metadata not in description or format is unexpected
                        logger.warning(f"Could not find complete metadata JSON in description for TxID {transaction.id}. Using default/fallback values if possible.")
                else:
                     logger.warning(f"Metadata marker not found in description for TxID {transaction.id}")

            # Ensure essential fields for _process_successful_payment are present
            if 'user_id' not in reconstructed_metadata and transaction.user_id:
                reconstructed_metadata['user_id'] = str(transaction.user_id)
            if 'payment_amount_usd' not in reconstructed_metadata and transaction.usd_equivalent:
                 reconstructed_metadata['payment_amount_usd'] = str(transaction.usd_equivalent)

            # If critical metadata like item_id, item_type, api_key_id (for new sub) is missing,
            # _process_successful_payment might fail or partially succeed.
            if not all(k in reconstructed_metadata for k in ['user_id', 'item_id', 'item_type', 'payment_amount_usd']):
                 logger.warning(f"Critical metadata missing after reconstruction for TxID {transaction.id}. Post-payment processing might fail or be incomplete. Metadata found: {reconstructed_metadata}")
                 transaction.status_message += " | Post-processing may fail due to missing metadata."
                 # Allow to proceed, _process_successful_payment will log specific errors.


            if not _process_successful_payment(db_session, transaction, reconstructed_metadata):
                processed_successfully = False
                # Error logged and status_message updated within _process_successful_payment
                logger.error(f"Post-payment processing failed for manually completed transaction {transaction_id}.")
                # transaction.status_message already updated by _process_successful_payment on failure
            else:
                logger.info(f"Post-payment processing successful for manually completed transaction {transaction_id}.")

    try:
        db_session.commit()
        message = "Payment status updated manually."
        if update_request.new_status == 'completed' and old_status != 'completed':
            message += " Post-payment processing attempted: " + ("Succeeded." if processed_successfully else "Failed (check logs and transaction status message).")
        return {"status": "success", "message": message}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database error manually updating payment status for transaction {transaction_id}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error manually updating status."}
