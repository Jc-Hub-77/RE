# backend/schemas/user_schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import datetime

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    is_admin: Optional[bool] = False


# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    referral_code_used: Optional[str] = None

class UserRegisterResponse(BaseModel):
    status: str
    message: str
    user_id: Optional[int] = None

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class UserLoginResponse(BaseModel):
    status: str
    access_token: Optional[str] = None
    token_type: Optional[str] = "bearer"
    user_id: Optional[int] = None
    username: Optional[str] = None
    is_admin: Optional[bool] = None
    message: Optional[str] = None # For error messages

class UserInDBBase(UserBase):
    id: int
    email_verified: bool = False
    is_admin: bool = False
    referral_code: Optional[str] = None
    referred_by_user_id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime] = None # Made optional as it might not always be set

    class Config:
        model_config = { "from_attributes": True } # For FastAPI to map SQLAlchemy models to Pydantic models

# --- Profile Schemas ---
class ProfileBase(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)

class ProfileCreate(ProfileBase):
    pass # No extra fields needed for creation beyond base

class ProfileUpdate(ProfileBase):
    pass # Same as base for now, can be extended

class ProfileResponse(ProfileBase):
    user_id: int
    
    class Config:
        model_config = { "from_attributes": True }

# --- User Response (combining User and Profile for some endpoints) ---
class UserPublicResponse(UserInDBBase):
    profile: Optional[ProfileResponse] = None
    # Exclude sensitive fields like password_hash, email_verification_token

    # Override fields from UserInDBBase if needed for public view
    # For example, we might not want to expose referred_by_user_id directly
    # Or ensure email_verified is always present

class UserProfileResponse(BaseModel): # Used by get_user_profile
    status: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_admin: Optional[bool] = None
    email_verified: Optional[bool] = None
    created_at: Optional[str] = None # ISO format string
    full_name: Optional[str] = None
    bio: Optional[str] = None
    referral_code: Optional[str] = None
    message: Optional[str] = None # For error messages


# --- Password Management Schemas ---
class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=8)

class PasswordReset(BaseModel):
    email: EmailStr # Added email field for user lookup
    token: str # This will be the plain token received by the user
    new_password: str = Field(..., min_length=8)

class EmailVerificationRequest(BaseModel):
    token: str

class EmailRequest(BaseModel):
    email: EmailStr

class GeneralResponse(BaseModel):
    status: str
    message: str
