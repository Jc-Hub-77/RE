# backend/schemas/payment_schemas.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
import datetime

class CreateChargeRequest(BaseModel):
    item_id: int # e.g., strategy_db_id for a new subscription, or user_strategy_subscription_id for renewal
    item_type: str = Field(..., description="Type of item being paid for (e.g., 'new_strategy_subscription', 'renew_strategy_subscription')")
    item_name: str # Will be displayed on payment page
    item_description: str # Displayed on payment page
    amount_usd: float = Field(..., gt=0, description="Amount in USD")
    subscription_months: Optional[int] = Field(1, ge=1, le=24, description="Number of months for subscription, if applicable")
    
    # Optional metadata to pass to payment gateway, which will be returned in webhooks
    # Crucial for linking webhook events back to specific actions or entities
    # For new strategy subscriptions, this should include api_key_id and custom_parameters_json
    metadata: Optional[Dict[str, Any]] = None 
    
    redirect_url: Optional[HttpUrl] = None # Override default success URL
    cancel_url: Optional[HttpUrl] = None   # Override default cancel URL

class CreateChargeResponse(BaseModel):
    status: str
    message: str
    internal_transaction_ref: Optional[str] = None # Our internal UUID for the transaction attempt
    gateway_charge_id: Optional[str] = None # ID from the payment gateway (e.g., Coinbase charge code)
    payment_page_url: Optional[HttpUrl] = None # URL for the user to complete payment
    expires_at: Optional[datetime.datetime] = None # When the charge/payment link expires

class PaymentTransactionView(BaseModel):
    id: int # Our DB primary key for the PaymentTransaction record
    internal_reference: Optional[str] = None # Our internal UUID reference
    date: datetime.datetime
    description: Optional[str] = None
    amount_crypto: Optional[float] = None # Actual crypto amount paid, if available
    crypto_currency: Optional[str] = None # Actual crypto currency used, if available
    usd_equivalent: Optional[float] = None # USD value at time of transaction, or initial USD price
    status: str
    status_message: Optional[str] = None
    gateway: str
    gateway_id: Optional[str] = None # Transaction ID from the payment gateway
    subscription_id: Optional[int] = None # Link to UserStrategySubscription if applicable

    class Config:
        model_config = { "from_attributes": True }

class UserPaymentHistoryResponse(BaseModel):
    status: str
    payment_history: List[PaymentTransactionView]
    total: int
    page: int
    per_page: int
    total_pages: int
    message: Optional[str] = None

# Webhook response is just a simple acknowledgement, actual processing is internal
class WebhookAcknowledgeResponse(BaseModel):
    status: str
    message: str

class AdminPaymentStatusUpdateRequest(BaseModel):
    new_status: str = Field(..., description="The new status for the payment transaction (e.g., 'completed', 'failed', 'pending', 'refunded', 'cancelled').")
    status_message: Optional[str] = Field(None, description="Optional message to set for the transaction's status.")
    admin_notes: Optional[str] = Field(None, description="Administrator notes regarding this manual update.")

class GeneralExchangeResponse(BaseModel): # As suggested by the prompt for generic responses
    status: str
    message: str
