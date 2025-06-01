# backend/api/v1/exchange_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.schemas import exchange_schemas
from backend.services import exchange_service
from backend.models import User
from backend.db import get_db
from .auth_router import get_current_active_user # Dependency for protected routes

router = APIRouter()

@router.post("/api-keys", response_model=exchange_schemas.ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def add_new_exchange_api_key(
    api_key_data: exchange_schemas.ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Adds a new exchange API key for the authenticated user.
    The key is encrypted before storage.
    """
    result = exchange_service.add_exchange_api_key(
        db_session=db,
        user_id=current_user.id,
        exchange_name=api_key_data.exchange_name,
        api_key_public=api_key_data.api_key_public,
        secret_key=api_key_data.secret_key,
        passphrase=api_key_data.passphrase,
        label=api_key_data.label
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/api-keys", response_model=exchange_schemas.ApiKeyListResponse)
async def list_user_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Lists all exchange API keys for the authenticated user (display format, no sensitive data).
    """
    result = exchange_service.get_user_exchange_api_keys_display(db, current_user.id)
    # This service function is expected to always return status: success with a list (possibly empty)
    return result

@router.delete("/api-keys/{api_key_id}", response_model=exchange_schemas.GeneralExchangeResponse)
async def delete_exchange_api_key(
    api_key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Removes an exchange API key for the authenticated user.
    """
    result = exchange_service.remove_exchange_api_key(db, current_user.id, api_key_id)
    if result["status"] == "error":
        # Distinguish between not found and other errors if necessary
        if "not found" in result.get("message", "").lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.post("/api-keys/{api_key_id}/test-connectivity", response_model=exchange_schemas.ApiKeyTestResponse)
async def test_exchange_api_key_connectivity(
    api_key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Tests the connectivity of a specified API key for the authenticated user.
    Updates the key's status in the database based on the test result.
    """
    result = exchange_service.test_api_connectivity(db, current_user.id, api_key_id)
    # The status in the result directly reflects the outcome of the test
    # No specific HTTP exception mapping here unless a systemic error occurs before the test logic
    if result.get("status") == "error_decryption" or "System error" in result.get("message", ""):
         # These are more like server-side issues with the setup or key itself
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"])
    return result

@router.get("/supported-exchanges", response_model=List[str])
async def list_supported_exchanges():
    """
    Lists all exchange IDs supported by the CCXT library.
    """
    return exchange_service.SUPPORTED_EXCHANGES
