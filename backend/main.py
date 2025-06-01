# backend/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import settings
from backend.models import Base, engine, init_db # Removed SessionLocal here if only for get_db
from backend.db import get_db # Import get_db from the new db.py
from backend.api.v1 import auth_router, admin_router, strategy_router, exchange_router, referral_router, payment_router, backtesting_router, live_trading_router # Import routers

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    openapi_url=f"/api/v1/openapi.json" # Standard OpenAPI doc path
)

@app.on_event("startup")
async def startup_event():
    """
    Initializes the database and creates tables on application startup.
    """
    print("Running application startup event...")
    # Initialize database (call init_db from models.py)
    # This sets up the global 'engine' and 'SessionLocal' in models.py
    init_db(settings.DATABASE_URL)

    # Create database tables (For development only. Use Alembic for production migrations)
    # This should be called after init_db has configured the engine.
    try:
        # Ensure engine is not None before calling create_all
        # Access the global engine from models.py
        from backend.models import engine as global_engine
        if global_engine is None:
             print("Error: Database engine is None after init_db call.")
             # Depending on desired behavior, you might raise an exception or exit
             # For now, we'll just print an error and skip table creation
        else:
            Base.metadata.create_all(bind=global_engine)
            print("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        # Depending on the severity, you might want to exit or handle this error.


# CORS Middleware
if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip() for origin in settings.ALLOWED_ORIGINS], # Ensure origins are strings
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Example Root Endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} - Version {settings.PROJECT_VERSION}"}

# API Routers
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication & Users"])
app.include_router(admin_router.router, prefix="/api/v1/admin", tags=["Admin Management"])
app.include_router(strategy_router.router, prefix="/api/v1/strategies", tags=["Strategies & Subscriptions"])
app.include_router(exchange_router.router, prefix="/api/v1/exchanges", tags=["Exchange API Keys"])
app.include_router(referral_router.router, prefix="/api/v1/referrals", tags=["Referrals"])
app.include_router(payment_router.router, prefix="/api/v1/payments", tags=["Payments & Webhooks"])
app.include_router(backtesting_router.router, prefix="/api/v1/backtests", tags=["Backtesting"])
app.include_router(live_trading_router.router, prefix="/api/v1/live-trading", tags=["Live Trading"])


# For running with uvicorn directly (e.g., uvicorn backend.main:app --reload)
if __name__ == "__main__":
    import uvicorn
    # Note: This uvicorn.run is for direct script execution.
    # Production deployments usually use Gunicorn + Uvicorn workers or similar.
    # The --reload flag is for development.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=".")

print(f"{settings.PROJECT_NAME} application startup complete. Listening on configured host/port.")
print(f"Database URL: {settings.DATABASE_URL}")
print(f"Allowed CORS origins: {settings.ALLOWED_ORIGINS}")
