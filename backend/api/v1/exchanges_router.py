# backend/api/v1/exchanges_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.schemas import exchange_schemas
from backend.models import User, ApiKey
from backend.main import get_db
from backend.api.v1.auth_router import get_current_active_user
from datetime import datetime

router = APIRouter()

# --- Public Exchange Endpoints ---
@router.get("/exchanges/supported", response_model=List[str])
async def get_supported_exchanges():
    """
    Lists all supported cryptocurrency exchanges.
    """
    # For now, return a hardcoded list of exchanges supported on the frontend
    return ["binance", "bybit", "phemex", "binanceus", "bitget", "coinbasepro"]

# --- User Exchange Management Endpoints (Protected) ---
@router.get("/users/{user_id}/exchange_keys", response_model=exchange_schemas.ApiKeyListResponse)
async def get_user_exchange_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    user_id: int = Path(..., description="The ID of the user")
):
    """
    Retrieves the exchange API keys connected by a specific user.
    """
    # Ensure user is requesting their own keys or is an admin (admin check omitted for simplicity here)
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view these keys")

    api_keys = db.query(ApiKey).filter(ApiKey.user_id == user_id).all()

    keys_display = []
    for key in api_keys:
        keys_display.append({
            "id": key.id,
            "exchange_name": key.exchange_name.capitalize(),
            "label": key.label,
            "api_key_preview": key.api_key_public_preview,
            "status": key.status,
            "status_message": key.status_message,
            "last_tested_at": key.last_tested_at,
            "created_at": key.created_at
        })

    return {"status": "success", "keys": keys_display}

@router.get("/users/{user_id}/exchange_keys/active", response_model=exchange_schemas.ApiKeyListResponse)
async def get_user_active_exchange_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    user_id: int = Path(..., description="The ID of the user")
):
    """
    Retrieves the active exchange API keys connected by a specific user.
    """
    # Ensure user is requesting their own keys or is an admin (admin check omitted for simplicity here)
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view these keys")

    active_api_keys = db.query(ApiKey).filter(ApiKey.user_id == user_id, ApiKey.status == "active").all()

    active_keys_display = []
    for key in active_api_keys:
        active_keys_display.append({
            "id": key.id,
            "exchange_name": key.exchange_name.capitalize(),
            "label": key.label,
            "api_key_preview": key.api_key_public_preview,
            "status": key.status,
            "status_message": key.status_message,
            "last_tested_at": key.last_tested_at,
            "created_at": key.created_at
        })

    return {"status": "success", "keys": active_keys_display}


@router.post("/users/{user_id}/exchange_keys", response_model=exchange_schemas.ApiKeyCreateResponse)
async def add_user_exchange_key(
    key_data: exchange_schemas.ApiKeyCreateRequest,
    user_id: int = Path(..., description="The ID of the user"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Adds a new exchange API key for a user.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to add keys for this user")

    print(f"Received new API key for user {user_id}: {key_data.label} on {key_data.exchange_name}")
    
    # Create a new ApiKey object
    new_api_key = ApiKey(
        user_id=user_id,
        exchange_name=key_data.exchange_name.lower(), # Store in lowercase
        label=key_data.label,
        api_key_public=key_data.api_key_public,
        api_key_private=key_data.api_key_private, # This should be encrypted in a real app
        api_key_passphrase=key_data.api_key_passphrase, # This should be encrypted in a real app
        api_key_public_preview=key_data.api_key_public[-4:] if key_data.api_key_public else None, # Store last 4 chars
        status="pending_test", # Initial status
        status_message="Awaiting connection test.",
        created_at=datetime.utcnow()
    )

    try:
        db.add(new_api_key)
        db.commit()
        db.refresh(new_api_key) # Refresh to get the generated ID

        return {
            "status": "success",
            "message": f"API Key for {key_data.exchange_name} added. Test pending.",
            "api_key_id": new_api_key.id
        }
    except Exception as e:
        db.rollback()
        print(f"Error adding API key for user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add API key to database.")


@router.post("/users/{user_id}/exchange_keys/{api_key_id}/test", response_model=exchange_schemas.ApiKeyTestResponse)
async def test_user_exchange_key(
    user_id: int = Path(..., description="The ID of the user"),
    api_key_id: str = Path(..., description="The ID of the API key"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Tests a user's exchange API key connection.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to test keys for this user")

    print(f"Testing API Key ID {api_key_id} for user {user_id}")
    
    # TODO: Implement actual API key testing logic here or call a service function
    # This would involve fetching the key from the DB, using an exchange library (like CCXT)
    # to attempt a connection/API call, and updating the key's status in the DB.
    
    # For now, return a placeholder response indicating test is conceptual
    return {
        "status": "test_conceptual",
        "message": "API key testing is conceptual and not fully implemented.",
        "api_key_id": api_key_id
    }


@router.delete("/users/{user_id}/exchange_keys/{api_key_id}", response_model=exchange_schemas.GeneralExchangeResponse)
async def remove_user_exchange_key(
    user_id: int = Path(..., description="The ID of the user"),
    api_key_id: str = Path(..., description="The ID of the API key"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Removes a user's exchange API key.
    """
    if user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to remove keys for this user")

    print(f"Removing API Key ID {api_key_id} for user {user_id}")
    
    # Retrieve the API key
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id, ApiKey.user_id == user_id).first()
    
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API Key not found or does not belong to user.")

    # TODO: Add a check here to prevent deleting an API key if it's actively used by a strategy subscription (User's point 3)

    try:
        db.delete(api_key)
        db.commit()
        return {
            "status": "success",
            "message": f"API Key {api_key_id} removed successfully.",
            "api_key_id": api_key_id
        }
    except Exception as e:
        db.rollback()
        print(f"Error removing API Key {api_key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to remove API key from database.")
