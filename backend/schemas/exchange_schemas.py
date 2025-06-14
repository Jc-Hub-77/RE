# backend/schemas/exchange_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime

class ApiKeyBase(BaseModel):
    exchange_name: str = Field(..., description="Name of the exchange (e.g., 'binance', 'coinbasepro')")
    label: Optional[str] = Field(None, description="User-defined label for the API key")
    api_key_public: str = Field(..., description="The public API key")
    secret_key: str = Field(..., description="The secret API key")
    passphrase: Optional[str] = Field(None, description="Passphrase, if required by the exchange")

class ApiKeyCreateRequest(ApiKeyBase):
    pass

class ApiKeyCreateResponse(BaseModel):
    status: str
    message: str
    api_key_id: Optional[int] = None

class ApiKeyDisplay(BaseModel):
    id: int
    exchange_name: str
    label: Optional[str] = None
    api_key_preview: Optional[str] = None
    status: str
    status_message: Optional[str] = None
    last_tested_at: Optional[datetime.datetime] = None
    created_at: datetime.datetime

    class Config:
        model_config = { "from_attributes": True }

class ApiKeyListResponse(BaseModel):
    status: str
    keys: List[ApiKeyDisplay]

class ApiKeyTestResponse(BaseModel):
    status: str # Will reflect the outcome: 'active', 'error_authentication', etc.
    message: str
    api_key_id: Optional[int] = None # To confirm which key was tested

class GeneralExchangeResponse(BaseModel): # For simple actions like delete
    status: str
    message: str
