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
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./trading_platform_local.db")
    
    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "a_very_secure_default_secret_key_please_change_me")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 48

    # API Key Encryption Key (for encrypting sensitive exchange API keys)
    API_ENCRYPTION_KEY: Optional[str] = os.getenv("API_ENCRYPTION_KEY")

    # Email settings (placeholders, configure for a real email service)
    SMTP_TLS: bool = True
    SMTP_PORT: int | None = os.getenv("SMTP_PORT", 587)
    SMTP_HOST: str | None = os.getenv("SMTP_HOST")
    SMTP_USER: str | None = os.getenv("SMTP_USER")
    SMTP_PASSWORD: str | None = os.getenv("SMTP_PASSWORD")
    EMAILS_FROM_EMAIL: str | None = os.getenv("EMAILS_FROM_EMAIL")
    EMAILS_FROM_NAME: str | None = os.getenv("EMAILS_FROM_NAME", "Trading Platform")
    
    # Frontend URL for generating links (e.g., email verification)
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8080") # Assuming frontend runs on 8080

    # CORS settings
    # Adjust according to your frontend's origin
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(',')

    # Referral System Settings
    REFERRAL_COMMISSION_RATE: float = float(os.getenv("REFERRAL_COMMISSION_RATE", "0.10"))  # 10%
    REFERRAL_MINIMUM_PAYOUT_USD: float = float(os.getenv("REFERRAL_MINIMUM_PAYOUT_USD", "20.00"))

    # Payment Gateway Settings (Coinbase Commerce Example)
    COINBASE_COMMERCE_API_KEY: Optional[str] = os.getenv("COINBASE_COMMERCE_API_KEY")
    COINBASE_COMMERCE_WEBHOOK_SECRET: Optional[str] = os.getenv("COINBASE_COMMERCE_WEBHOOK_SECRET")
    # URLs for payment redirects (can be overridden in charge creation)
    APP_PAYMENT_SUCCESS_URL: str = os.getenv("APP_PAYMENT_SUCCESS_URL", f"{FRONTEND_URL}/payment/success")
    APP_PAYMENT_CANCEL_URL: str = os.getenv("APP_PAYMENT_CANCEL_URL", f"{FRONTEND_URL}/payment/cancel")

    # Directory for strategies
    STRATEGIES_DIR: Optional[str] = os.getenv("STRATEGIES_DIR")


settings = Settings()

if not settings.STRATEGIES_DIR:
    # Fallback for development if not set, assuming 'strategies' dir is one level above 'backend'
    # For production, this should be explicitly set.
    strategies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")
    if os.path.isdir(strategies_path):
        settings.STRATEGIES_DIR = strategies_path
    else:
        # If neither env var nor default path exists, raise error or log warning
        # For now, let's log a warning, consistent with how API_ENCRYPTION_KEY is handled if missing (though that one is critical)
        print("WARNING: STRATEGIES_DIR environment variable is not set and default path not found.")
        # settings.STRATEGIES_DIR = None # Or some other default if appropriate

# Ensure JWT_SECRET_KEY is not the default in a production-like environment
if settings.JWT_SECRET_KEY == "a_very_secure_default_secret_key_please_change_me" and os.getenv("ENVIRONMENT") == "production":
    raise ValueError("CRITICAL: JWT_SECRET_KEY must be set to a strong, unique secret in production!")

# Ensure API_ENCRYPTION_KEY is set in a production environment
if not settings.API_ENCRYPTION_KEY and os.getenv("ENVIRONMENT") == "production":
    raise ValueError("CRITICAL: API_ENCRYPTION_KEY must be set in a production environment!")

if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL not set. Please configure it in .env or environment variables.")
