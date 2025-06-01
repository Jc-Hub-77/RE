# backend/api/v1/payment_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.schemas import payment_schemas
from backend.services import payment_service
from backend.models import User
from backend.db import get_db
from backend.api.v1.auth_router import get_current_active_user # Dependency for protected routes

router = APIRouter()

# --- User-Facing Payment Endpoints (Protected) ---
@router.post("/charges", response_model=payment_schemas.CreateChargeResponse, status_code=status.HTTP_201_CREATED)
async def create_payment_charge(
    charge_data: payment_schemas.CreateChargeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Initiates a payment charge via the configured payment gateway (e.g., Coinbase Commerce).
    Returns details needed to redirect the user to the payment page.
    """
    # Ensure necessary metadata is included for webhook processing, especially for new subscriptions
    # The frontend should provide item_id (strategy_db_id), item_type, item_name, item_description, amount_usd, subscription_months
    # For new strategy subscriptions, the frontend also needs to provide api_key_id and custom_parameters
    
    # Add api_key_id and custom_parameters_json to metadata if item_type is new_strategy_subscription
    metadata_to_pass = charge_data.metadata if charge_data.metadata is not None else {}
    if charge_data.item_type == "new_strategy_subscription":
        if 'api_key_id' not in metadata_to_pass or 'custom_parameters_json' not in metadata_to_pass:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="api_key_id and custom_parameters_json are required in metadata for new_strategy_subscription.")
        # Ensure custom_parameters_json is a string
        if not isinstance(metadata_to_pass.get('custom_parameters_json'), str):
             metadata_to_pass['custom_parameters_json'] = json.dumps(metadata_to_pass.get('custom_parameters_json')) # Convert dict to JSON string

    result = payment_service.create_coinbase_commerce_charge(
        db_session=db,
        user_id=current_user.id,
        item_id=charge_data.item_id,
        item_type=charge_data.item_type,
        item_name=charge_data.item_name,
        item_description=charge_data.item_description,
        amount_usd=charge_data.amount_usd,
        subscription_months=charge_data.subscription_months,
        redirect_url=str(charge_data.redirect_url) if charge_data.redirect_url else None,
        cancel_url=str(charge_data.cancel_url) if charge_data.cancel_url else None,
        metadata=metadata_to_pass # Pass the metadata
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    elif result["status"] == "success_simulated":
         # For simulated success, return 200 OK
         return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    return result

@router.get("/history/me", response_model=payment_schemas.UserPaymentHistoryResponse)
async def get_authenticated_user_payment_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50)
):
    """
    Retrieves the payment transaction history for the currently authenticated user.
    """
    result = payment_service.get_user_payment_history(db, current_user.id, page, per_page)
    # This service function is expected to always return status: success
    return result


# --- Payment Gateway Webhook Endpoint (Public - requires signature verification) ---
@router.post("/webhooks/coinbase-commerce")
async def handle_coinbase_commerce_webhook_event(
    request: Request,
    coinbase_signature: str = Header(None, alias='X-CC-Webhook-Signature'),
    db: Session = Depends(get_db)
):
    """
    Handles incoming webhook events from Coinbase Commerce.
    Verifies the signature and processes the event (e.g., confirms payment, updates status).
    """
    if not coinbase_signature:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-CC-Webhook-Signature header")

    request_body = await request.body()
    request_body_str = request_body.decode('utf-8')

    # The service function handles signature verification and event processing
    result, status_code = payment_service.handle_coinbase_commerce_webhook(
        db_session=db,
        request_body_str=request_body_str,
        webhook_signature=coinbase_signature
    )
    
    # Return the status code determined by the service function
    return JSONResponse(status_code=status_code, content=result)

# --- Admin-only Payment Endpoints ---

from backend.api.v1.admin_router import get_current_active_admin_user # Import admin dependency

@router.get("/admin/transactions", response_model=payment_schemas.UserPaymentHistoryResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_all_payment_transactions(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = Query(None), # Filter by user
    status: Optional[str] = Query(None), # Filter by status
    gateway: Optional[str] = Query(None) # Filter by gateway
):
    """
    Admin endpoint to list all payment transactions with filtering and pagination.
    """
    result = payment_service.list_all_payment_transactions(db, page, per_page, user_id, status, gateway)
    # This service function is expected to always return status: success
    return result

@router.get("/admin/transactions/{transaction_id}", response_model=payment_schemas.PaymentTransactionView, dependencies=[Depends(get_current_active_admin_user)])
async def admin_get_payment_transaction_details(
    transaction_id: int,
    db: Session = Depends(get_db)
):
    """
    Admin endpoint to view details of a specific payment transaction.
    """
    result = payment_service.get_payment_transaction_by_id(db, transaction_id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result

# TODO: Admin endpoint to manually update payment status (e.g., for manual payments)

from fastapi.responses import JSONResponse # For returning JSONResponse with specific status codes
import json # Needed for json.dumps
