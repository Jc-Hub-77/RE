# backend/services/user_service.py
import datetime
import uuid
import logging
import random
import string
# import smtplib # For email sending - No longer directly used here
# from email.mime.text import MIMEText # For email sending - No longer directly used here

from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import timedelta # Explicitly ensure timedelta is imported
from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.models import User, Profile, Referral # Adjusted to relative import
from backend.config import settings # Adjusted to relative import
from backend.tasks import send_email_task # Import the new Celery task

# Initialize logger
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Helper Functions ---

# def send_email_async(to_email: str, subject: str, body: str):
#     """
#     Sends an email using SMTP. This is a basic synchronous implementation.
#     For production, consider using a task queue and a dedicated email service.
#     """
#     if not settings.SMTP_HOST or not settings.SMTP_USER or not settings.SMTP_PASSWORD or not settings.EMAILS_FROM_EMAIL:
#         logger.warning(f"Email not sent to {to_email}: SMTP settings are not fully configured. Simulating email.")
#         logger.info(f"--- Simulated Email to {to_email} ---")
#         logger.info(f"Subject: {subject}")
#         logger.info(f"Body:\n{body}")
#         logger.info(f"--- End of Simulated Email ---")
#         return {"status": "warning_simulated", "message": "Email settings not configured; email simulated."}
# 
#     try:
#         msg = MIMEText(body)
#         msg['Subject'] = subject
#         msg['From'] = settings.EMAILS_FROM_EMAIL
#         msg['To'] = to_email
# 
#         smtp_port = settings.SMTP_PORT if settings.SMTP_PORT is not None else 587
# 
#         with smtplib.SMTP(settings.SMTP_HOST, smtp_port) as server:
#             if settings.SMTP_TLS:
#                 server.starttls()
#             server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
#             server.sendmail(settings.EMAILS_FROM_EMAIL, to_email, msg.as_string())
# 
#         logger.info(f"Successfully sent email to {to_email} with subject '{subject}'.")
#         return {"status": "success", "message": "Email sent successfully."}
#     except Exception as e:
#         logger.error(f"Failed to send email to {to_email}: {e}", exc_info=True)
#         return {"status": "error", "message": f"Failed to send email: {e}"}

def _generate_secure_token_data(expire_hours: int):
    token = str(uuid.uuid4())
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=expire_hours)
    return token, expires_at

def _generate_unique_referral_code(db_session: Session) -> str:
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        existing_user = db_session.query(User).filter(User.referral_code == code).first()
        if not existing_user:
            return code

# --- Password Hashing ---
def _get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def _verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# --- Token Creation ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    current_time = datetime.datetime.utcnow()
    if expires_delta:
        expire = current_time + expires_delta
    else:
        expire = current_time + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"iat": current_time, "exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

# --- User Service Functions ---

def get_user_by_id(db_session: Session, user_id: int) -> User | None:
    return db_session.query(User).filter(User.id == user_id).first()

def get_user_by_email(db_session: Session, email: str) -> User | None:
    return db_session.query(User).filter(User.email == email).first()

def get_user_by_username(db_session: Session, username: str) -> User | None:
    return db_session.query(User).filter(User.username == username).first()


def register_user(db_session: Session, email: str, username: str, password: str, referral_code_used: str = None):
    logger.info(f"Attempting registration for username: {username}, email: {email}, referral code used: {referral_code_used or 'None'}")

    if not email or "@" not in email or "." not in email:
        return {"status": "error", "message": "Invalid email format."}
    if not username or len(username) < 3:
        return {"status": "error", "message": "Username must be at least 3 characters."}
    if not password or len(password) < 8: # Basic password length check
        return {"status": "error", "message": "Password must be at least 8 characters."}

    existing_user_by_email = get_user_by_email(db_session, email)
    if existing_user_by_email:
        logger.warning(f"Registration failed: Email '{email}' already registered.")
        return {"status": "error", "message": "Email already registered."}
    
    existing_user_by_username = get_user_by_username(db_session, username)
    if existing_user_by_username:
        logger.warning(f"Registration failed: Username '{username}' already exists.")
        return {"status": "error", "message": "Username already exists."}

    hashed_password = _get_password_hash(password)
    email_verification_token, email_verification_token_expires_at = _generate_secure_token_data(settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS)
    user_own_referral_code = _generate_unique_referral_code(db_session)

    referrer_user_id_to_store = None
    if referral_code_used:
        referrer = db_session.query(User).filter(User.referral_code == referral_code_used).first()
        if referrer:
            referrer_user_id_to_store = referrer.id
            logger.info(f"Valid referral code '{referral_code_used}' used by {username}, referrer ID: {referrer.id}")
        else:
            logger.warning(f"Invalid referral code '{referral_code_used}' provided by {username}.")
            # Not returning error for invalid referral code, just not applying it.

    new_user = User(
        username=username, email=email, password_hash=hashed_password,
        email_verified=False, is_admin=False, referral_code=user_own_referral_code,
        referred_by_user_id=referrer_user_id_to_store, created_at=datetime.datetime.utcnow(),
        email_verification_token=email_verification_token,
        email_verification_token_expires_at=email_verification_token_expires_at,
        is_active=True, # Users are active by default upon registration
        last_password_change_at=datetime.datetime.utcnow() # Set initial password change time
    )
    # Profile is created via relationship back_populates if Profile model has user_id FK
    # If Profile must be explicitly created:
    new_profile = Profile(user=new_user, full_name="") # Initialize with empty full_name if needed

    try:
        db_session.add(new_user)
        # db_session.add(new_profile) # Only if Profile isn't cascaded or needs explicit add
        db_session.flush() 

        if referrer_user_id_to_store and new_user.id:
            new_referral_record = Referral(
                referrer_user_id=referrer_user_id_to_store,
                referred_user_id=new_user.id,
                signed_up_at=new_user.created_at
            )
            db_session.add(new_referral_record)
        
        db_session.commit()
        # db_session.refresh(new_user) # Not strictly needed if not immediately using refreshed fields

        verification_link = f"{settings.FRONTEND_URL}/verify-email?token={email_verification_token}"
        email_subject = f"Verify your email for {settings.PROJECT_NAME}"
        email_body = (
            f"Hi {username},\n\n"
            f"Thanks for registering for {settings.PROJECT_NAME}! Please click the link below to verify your email address:\n"
            f"{verification_link}\n\n"
            f"This link will expire in {settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS} hours.\n\n"
            f"Thanks,\nThe {settings.PROJECT_NAME} Team"
        )
        send_email_task.delay(new_user.email, email_subject, email_body)
        
        logger.info(f"User {username} (ID: {new_user.id}) registered successfully. Verification email queued.")
        return {"status": "success", "message": "Registration successful. Please check your email for verification.", "user_id": new_user.id}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error during user registration for {username}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error during registration."}


def verify_email(db_session: Session, verification_token: str):
    user_to_verify = db_session.query(User).filter(User.email_verification_token == verification_token).first()
    
    if not user_to_verify:
        logger.warning(f"Invalid email verification token used: {verification_token}")
        return {"status": "error", "message": "Invalid verification token."}
    if user_to_verify.email_verified:
        logger.info(f"Email for user {user_to_verify.username} already verified.")
        return {"status": "info", "message": "Email already verified."}
    if user_to_verify.email_verification_token_expires_at < datetime.datetime.utcnow():
        logger.warning(f"Expired email verification token used for user {user_to_verify.username}.")
        return {"status": "error", "message": "Verification token has expired. Please request a new one."}

    user_to_verify.email_verified = True
    user_to_verify.email_verification_token = None # Clear token
    user_to_verify.email_verification_token_expires_at = None # Clear expiry
    try:
        db_session.commit()
        logger.info(f"Email successfully verified for user {user_to_verify.username} (ID: {user_to_verify.id}).")
        return {"status": "success", "message": "Email verified successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database error during email verification for user {user_to_verify.username}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error during email verification."}


def login_user(db_session: Session, username_or_email: str, password: str):
    user = db_session.query(User).filter(
        or_(User.username == username_or_email, User.email == username_or_email)
    ).first()

    if not user:
        logger.warning(f"Login attempt failed: User '{username_or_email}' not found.")
        return {"status": "error", "message": "Invalid username/email or password."}
    if not user.is_active:
        logger.warning(f"Login attempt failed: User '{username_or_email}' (ID: {user.id}) is inactive.")
        return {"status": "error", "message": "Account is inactive. Please contact support."}
    if not _verify_password(password, user.password_hash):
        logger.warning(f"Login attempt failed: Invalid password for user '{username_or_email}' (ID: {user.id}).")
        return {"status": "error", "message": "Invalid username/email or password."}
    if not user.email_verified:
        logger.warning(f"Login attempt failed: Email not verified for user '{username_or_email}' (ID: {user.id}).")
        return {"status": "error", "message": "Email not verified. Please verify your email before logging in."}
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "is_admin": user.is_admin}, # Ensure sub is string for JWT standard
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user.username} (ID: {user.id}) logged in successfully.")
    return {
        "status": "success", "access_token": access_token, "token_type": "bearer",
        "user_id": user.id, "username": user.username, "is_admin": user.is_admin
    }


def get_user_profile(db_session: Session, user_id: int):
    user = get_user_by_id(db_session, user_id)
    if not user:
        return {"status": "error", "message": "User not found."}
    
    profile_data = user.profile # Assumes user.profile relationship is loaded or use joinedload if needed frequently

    return {
        "status": "success", "user_id": user.id, "username": user.username, "email": user.email,
        "is_admin": user.is_admin, "email_verified": user.email_verified,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "full_name": profile_data.full_name if profile_data else None,
        "bio": profile_data.bio if profile_data else None,
        "referral_code": user.referral_code,
    }

def update_user_profile(db_session: Session, user_id: int, data_to_update: dict):
    user = get_user_by_id(db_session, user_id)
    if not user: return {"status": "error", "message": "User not found."}

    profile_updated_fields = []
    if not user.profile: 
        user.profile = Profile(user_id=user.id) # Ensure user_id is set if creating new
        db_session.add(user.profile)
        profile_updated_fields.append("profile_created")

    if "email" in data_to_update and data_to_update["email"] != user.email:
        new_email = data_to_update["email"]
        if not new_email or "@" not in new_email or "." not in new_email:
             return {"status": "error", "message": "Invalid new email format."}
        
        existing_email_user = get_user_by_email(db_session, new_email)
        if existing_email_user and existing_email_user.id != user_id:
            return {"status": "error", "message": "New email address is already in use."}
        
        user.email = new_email
        user.email_verified = False # Email change requires re-verification
        token, expires_at = _generate_secure_token_data(settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS)
        user.email_verification_token = token
        user.email_verification_token_expires_at = expires_at
        profile_updated_fields.append("email")
        
        verification_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"
        email_subject = f"Verify your new email address for {settings.PROJECT_NAME}"
        email_body = f"Hi {user.username},\n\nPlease click the link to verify your new email address: {verification_link}"
        send_email_task.delay(user.email, email_subject, email_body)
        
    if "full_name" in data_to_update and user.profile.full_name != data_to_update["full_name"]:
        user.profile.full_name = data_to_update["full_name"]
        profile_updated_fields.append("full_name")
    
    if "bio" in data_to_update and user.profile.bio != data_to_update["bio"]:
        user.profile.bio = data_to_update["bio"]
        profile_updated_fields.append("bio")
    
    if not profile_updated_fields:
         return {"status": "info", "message": "No changes detected to update."}

    try:
        db_session.commit()
        logger.info(f"Profile for user {user.username} (ID: {user.id}) updated. Fields: {', '.join(profile_updated_fields)}.")
        return {"status": "success", "message": "Profile updated successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database error during profile update for user {user.username}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error during profile update."}


def change_password(db_session: Session, user_id: int, old_password: str, new_password: str):
    user = get_user_by_id(db_session, user_id)
    if not user: return {"status": "error", "message": "User not found."}
    if not _verify_password(old_password, user.password_hash):
        return {"status": "error", "message": "Incorrect old password."}
    if len(new_password) < 8: # Basic password length check
        return {"status": "error", "message": "New password must be at least 8 characters."}

    user.password_hash = _get_password_hash(new_password)
    user.last_password_change_at = datetime.datetime.utcnow()
    try:
        db_session.commit()
        logger.info(f"Password changed successfully for user {user.username} (ID: {user.id}).")
        # TODO: Invalidate other active sessions/tokens for this user (e.g., by managing a token blacklist or session store)
        return {"status": "success", "message": "Password changed successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database error during password change for user {user.username}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error during password change."}

def forgot_password_request(db_session: Session, email: str):
    user = get_user_by_email(db_session, email)
    if not user:
        logger.warning(f"Password reset requested for non-existent email: {email}")
        # Still return success to prevent email enumeration
        return {"status": "success", "message": "If an account with this email exists, a password reset link has been sent."}

    token, expires_at = _generate_secure_token_data(1) # Password reset tokens usually have shorter expiry (e.g., 1 hour)
    user.password_reset_token = token # Store the plain token for lookup; consider hashing if desired for extra security layer
    user.password_reset_token_expires_at = expires_at
    
    try:
        db_session.commit()
        reset_link = f"{settings.FRONTEND_URL}/reset-password?token={token}"
        email_subject = f"Password Reset Request for {settings.PROJECT_NAME}"
        email_body = (
            f"Hi {user.username},\n\n"
            f"You requested a password reset. Click the link below to reset your password:\n"
            f"{reset_link}\n\n"
            f"This link will expire in 1 hour. If you did not request this, please ignore this email.\n\n"
            f"Thanks,\nThe {settings.PROJECT_NAME} Team"
        )
        send_email_task.delay(user.email, email_subject, email_body)
        logger.info(f"Password reset email queued for {user.email} for user {user.username} (ID: {user.id}).")
        return {"status": "success", "message": "If an account with this email exists, a password reset link has been sent."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error processing password reset request for {email}: {e}", exc_info=True)
        return {"status": "error", "message": "Error processing password reset request."}

def reset_password_with_token(db_session: Session, token: str, new_password: str):
    if not token or not new_password:
        return {"status": "error", "message": "Token and new password are required."}
    if len(new_password) < 8:
        return {"status": "error", "message": "New password must be at least 8 characters."}

    user = db_session.query(User).filter(User.password_reset_token == token).first()

    if not user:
        logger.warning(f"Invalid or non-existent password reset token used: {token}")
        return {"status": "error", "message": "Invalid or expired password reset token."}
    if user.password_reset_token_expires_at < datetime.datetime.utcnow():
        logger.warning(f"Expired password reset token used for user {user.username} (ID: {user.id}).")
        user.password_reset_token = None # Clear expired token
        user.password_reset_token_expires_at = None
        db_session.commit()
        return {"status": "error", "message": "Password reset token has expired."}

    user.password_hash = _get_password_hash(new_password)
    user.last_password_change_at = datetime.datetime.utcnow()
    user.password_reset_token = None # Invalidate token after use
    user.password_reset_token_expires_at = None
    user.email_verified = True # Resetting password often implies email ownership
    
    try:
        db_session.commit()
        logger.info(f"Password reset successfully for user {user.username} (ID: {user.id}) using token.")
        # TODO: Consider sending a confirmation email that password was changed.
        # TODO: Invalidate other active sessions/tokens for this user.
        return {"status": "success", "message": "Password has been reset successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Database error during password reset for user {user.username} (ID: {user.id}): {e}", exc_info=True)
        return {"status": "error", "message": "Database error during password reset."}

def request_new_verification_email(db_session: Session, email: str):
    user = get_user_by_email(db_session, email)
    if not user:
        return {"status": "error", "message": "User with this email not found."}
    if user.email_verified:
        return {"status": "info", "message": "Email is already verified."}

    # Generate new token and expiry
    email_verification_token, email_verification_token_expires_at = _generate_secure_token_data(settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS)
    user.email_verification_token = email_verification_token
    user.email_verification_token_expires_at = email_verification_token_expires_at

    try:
        db_session.commit()
        verification_link = f"{settings.FRONTEND_URL}/verify-email?token={email_verification_token}"
        email_subject = f"Verify your email for {settings.PROJECT_NAME}"
        email_body = (
            f"Hi {user.username},\n\n"
            f"Please click the link below to verify your email address:\n"
            f"{verification_link}\n\n"
            f"This link will expire in {settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS} hours.\n\n"
            f"Thanks,\nThe {settings.PROJECT_NAME} Team"
        )
        send_email_task.delay(user.email, email_subject, email_body)
        logger.info(f"New verification email queued for {user.email} for user {user.username} (ID: {user.id}).")
        return {"status": "success", "message": "New verification email sent. Please check your inbox."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error resending verification email for {email}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error resending verification email."}

# `manage_security_settings` placeholder - kept for conceptual completeness from original file
def manage_security_settings(user_id, settings_data): 
    logger.info(f"Placeholder: Managing security settings for user_id: {user_id} with settings: {settings_data}")
    # Actual implementation would involve 2FA setup, activity logs, etc.
    # This is a complex feature and is out of scope for current service completion.
    return {"status": "success_placeholder", "message": "Security settings management is conceptual here."}
