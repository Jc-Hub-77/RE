# backend/config.py
import os
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
# Create a .env file in the backend directory for local development:
# DATABASE_URL="sqlite:///./trading_platform_dev.db"
# JWT_SECRET_KEY="your-super-secret-key-for-jwt-!ChangeME!"
# FRONTEND_URL="http://localhost:3000" # Or your frontend port

# Explicitly load environment variables from the .env file in the backend directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

class Settings:
    PROJECT_NAME: str = "Trading Platform API"
    PROJECT_VERSION: str = "0.1.0"

    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "YOUR_PRODUCTION_DATABASE_URL_CHANGE_ME")
    
    # JWT settings
    # IMPORTANT: This key is crucial for security. It should be a long, random, and unique string.
    # Generate a strong key using: import secrets; secrets.token_hex(32)
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "YOUR_PRODUCTION_JWT_SECRET_KEY_CHANGE_ME")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 48
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = int(os.getenv("PASSWORD_RESET_TOKEN_EXPIRE_HOURS", "1"))

    # API Key Encryption Key (for encrypting sensitive exchange API keys)
    # This key is used to encrypt sensitive data like external API keys stored by users.
    # Generate a strong key using: import secrets; secrets.token_hex(32)
    API_ENCRYPTION_KEY: Optional[str] = os.getenv("API_ENCRYPTION_KEY")

    # Email settings
    SMTP_TLS: bool = True
    SMTP_PORT: int | None = os.getenv("SMTP_PORT", 587)
    SMTP_HOST: str | None = os.getenv("SMTP_HOST", "YOUR_PRODUCTION_SMTP_HOST_CHANGE_ME")
    SMTP_USER: str | None = os.getenv("SMTP_USER", "your_production_smtp_user@example.com")
    SMTP_PASSWORD: str | None = os.getenv("SMTP_PASSWORD") # For production, ensure this is set via env var
    EMAILS_FROM_EMAIL: str | None = os.getenv("EMAILS_FROM_EMAIL", "noreply@yourfrontend.com")
    EMAILS_FROM_NAME: str | None = os.getenv("EMAILS_FROM_NAME", "Your Platform Name")
    
    # Frontend URL for generating links (e.g., email verification, password reset)
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://yourfrontend.com")

    # CORS settings
    # Define the origins allowed to make cross-site requests to your backend.
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "https://yourfrontend.com,https://www.yourfrontend.com").split(',')

    # Referral System Settings
    REFERRAL_COMMISSION_RATE: float = float(os.getenv("REFERRAL_COMMISSION_RATE", "0.10"))  # 10%
    REFERRAL_MINIMUM_PAYOUT_USD: float = float(os.getenv("REFERRAL_MINIMUM_PAYOUT_USD", "20.00"))

    # Default capital for live strategies if not specified in custom parameters.
    # Can be overridden by the 'capital' value in a strategy's custom parameters.
    DEFAULT_STRATEGY_CAPITAL: float = float(os.getenv("DEFAULT_STRATEGY_CAPITAL", "10000.0"))

    # Payment Gateway Settings (Example: Coinbase Commerce)
    # Ensure these are set via environment variables in production.
    COINBASE_COMMERCE_API_KEY: Optional[str] = os.getenv("COINBASE_COMMERCE_API_KEY") # e.g., your_coinbase_commerce_api_key
    COINBASE_COMMERCE_WEBHOOK_SECRET: Optional[str] = os.getenv("COINBASE_COMMERCE_WEBHOOK_SECRET") # e.g., your_coinbase_webhook_secret
    # URLs for payment redirects will use the updated FRONTEND_URL default
    APP_PAYMENT_SUCCESS_URL: str = os.getenv("APP_PAYMENT_SUCCESS_URL", f"{FRONTEND_URL}/payment/success")
    APP_PAYMENT_CANCEL_URL: str = os.getenv("APP_PAYMENT_CANCEL_URL", f"{FRONTEND_URL}/payment/cancel")

    # Directory for user-defined strategies.
    # For production, this MUST be an absolute path to a secure, persistent location outside the application's ephemeral container storage.
    STRATEGIES_DIR: Optional[str] = os.getenv("STRATEGIES_DIR")


settings = Settings()

if not settings.STRATEGIES_DIR:
    # Fallback for development if STRATEGIES_DIR is not set.
    # Assumes 'strategies' directory is one level above 'backend'.
    # For production, STRATEGIES_DIR MUST be explicitly set to an appropriate, persistent path.
    strategies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")
    if os.path.isdir(strategies_path):
        settings.STRATEGIES_DIR = strategies_path
    # No 'else' here; production check below will handle missing STRATEGIES_DIR.

if os.getenv("ENVIRONMENT") == "production":
    # Critical Production Validations
    # Halt startup if essential production configurations are missing or insecure.

    # Validate JWT_SECRET_KEY
    if settings.JWT_SECRET_KEY == "YOUR_PRODUCTION_JWT_SECRET_KEY_CHANGE_ME" or \
       settings.JWT_SECRET_KEY == "a_very_secure_default_secret_key_please_change_me": # Check old default too
        raise ValueError("CRITICAL: JWT_SECRET_KEY is not set or is using a default placeholder. "
                         "Set a strong, unique secret in the environment for production.")

    # Validate API_ENCRYPTION_KEY
    if not settings.API_ENCRYPTION_KEY:
        raise ValueError("CRITICAL: API_ENCRYPTION_KEY is not set. "
                         "Set a strong encryption key in the environment for production.")

    # Validate STRATEGIES_DIR
    if not settings.STRATEGIES_DIR or \
       "backend" in settings.STRATEGIES_DIR or \
       "strategies" == os.path.basename(settings.STRATEGIES_DIR.rstrip('/\\')): # Check if it's just 'strategies'
        raise ValueError("CRITICAL: STRATEGIES_DIR is not configured correctly for production. "
                         "It must be an absolute path to a secure, persistent location "
                         "and not a default development path (e.g., './strategies' or within 'backend').")

    # Validate Database URL
    if settings.DATABASE_URL == "YOUR_PRODUCTION_DATABASE_URL_CHANGE_ME" or \
       settings.DATABASE_URL.startswith("sqlite:"):
        raise ValueError("CRITICAL: DATABASE_URL is not configured for production or is using SQLite. "
                         "Set a valid production database URL (e.g., PostgreSQL, MySQL) in the environment.")

    # Validate Frontend URL
    if settings.FRONTEND_URL == "https://yourfrontend.com" or \
       settings.FRONTEND_URL.startswith("http://localhost"):
        raise ValueError("CRITICAL: FRONTEND_URL is not configured for production or is using a local/placeholder URL. "
                         "Set the correct public frontend URL in the environment.")
    
    # Validate SMTP Settings
    if not settings.SMTP_HOST or settings.SMTP_HOST == "YOUR_PRODUCTION_SMTP_HOST_CHANGE_ME":
        raise ValueError("CRITICAL: SMTP_HOST is not configured for production.")
    if not settings.SMTP_USER or settings.SMTP_USER == "your_production_smtp_user@example.com":
        raise ValueError("CRITICAL: SMTP_USER is not configured for production.")
    if not settings.SMTP_PASSWORD: # SMTP_PASSWORD should not have a default value other than None
        raise ValueError("CRITICAL: SMTP_PASSWORD is not configured for production.")
    if not settings.EMAILS_FROM_EMAIL or settings.EMAILS_FROM_EMAIL == "noreply@yourfrontend.com":
        raise ValueError("CRITICAL: EMAILS_FROM_EMAIL is not configured for production.")

    # Validate Payment Gateway Settings
    if not settings.COINBASE_COMMERCE_API_KEY:
        raise ValueError("CRITICAL: COINBASE_COMMERCE_API_KEY is not configured for production.")
    if not settings.COINBASE_COMMERCE_WEBHOOK_SECRET:
        raise ValueError("CRITICAL: COINBASE_COMMERCE_WEBHOOK_SECRET is not configured for production.")

# General check for DATABASE_URL (applies to all environments)
# This is somewhat redundant due to the production check above, but good as a basic guard.
if not settings.DATABASE_URL: # This check was already here and is fine.
    raise ValueError("DATABASE_URL not set. Please configure it in .env or environment variables.")
