# backend/models.py
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base # Updated import

# DATABASE_URL = "sqlite:///./trading_platform.db" # Example for SQLite
# For production, consider PostgreSQL or MySQL and load from environment variables/config file.
# e.g., DATABASE_URL = "postgresql://user:password@host:port/database"
engine = None # Will be initialized by the main application
SessionLocal = None # Will be initialized by the main application
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False) # Store hashed passwords only
    email_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    referral_code = Column(String, unique=True, index=True, nullable=True) # Unique code for this user to refer others
    referred_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # ID of the user who referred this user
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    email_verification_token = Column(String, unique=True, nullable=True)
    email_verification_token_expires_at = Column(DateTime, nullable=True)
    password_reset_token = Column(String, unique=True, nullable=True) # Add password reset token field
    password_reset_token_expires_at = Column(DateTime, nullable=True) # Add password reset token expiry field
    is_active = Column(Boolean, default=True) # Add is_active field
    last_password_change_at = Column(DateTime, nullable=True, default=None)

    profile = relationship("Profile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    strategy_subscriptions = relationship("UserStrategySubscription", back_populates="user", cascade="all, delete-orphan")
    backtest_results = relationship("BacktestResult", back_populates="user", cascade="all, delete-orphan")
    payment_transactions = relationship("PaymentTransaction", back_populates="user", cascade="all, delete-orphan")
    
    # Relationships for referrals
    referrals_made = relationship("Referral", foreign_keys="Referral.referrer_user_id", back_populates="referrer", cascade="all, delete-orphan")
    # referred_by = relationship("User", remote_side=[id], foreign_keys=[referred_by_user_id]) # This creates a circular dependency if not handled carefully.
                                                                                             # It's often simpler to query for this.

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name = Column(String, index=True, nullable=True)
    bio = Column(Text, nullable=True)
    # Add other profile fields: avatar_url, etc.

    user = relationship("User", back_populates="profile")

    def __repr__(self):
        return f"<Profile(user_id={self.user_id}, full_name='{self.full_name}')>"

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange_name = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True) # User-defined label for the key
    api_key_public_preview = Column(String, nullable=True) # e.g., "abc...xyz"
    
    encrypted_api_key = Column(Text, nullable=False) # Encrypted public API key
    encrypted_secret_key = Column(Text, nullable=False) # Encrypted secret API key
    encrypted_passphrase = Column(Text, nullable=True) # Encrypted passphrase, if applicable

    status = Column(String, default="pending_verification", index=True) # e.g., pending_verification, active, error_authentication, error_decryption
    status_message = Column(Text, nullable=True) # More details on the status, e.g., error message
    last_tested_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="api_keys")
    # subscriptions_using_this_key = relationship("UserStrategySubscription", back_populates="api_key") # If needed

    def __repr__(self):
        return f"<ApiKey(id={self.id}, user_id='{self.user_id}', exchange='{self.exchange_name}', label='{self.label}')>"

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    python_code_path = Column(String, nullable=False) # Path to the .py file containing the strategy logic
    default_parameters = Column(Text) # JSON string or dict of default params
    category = Column(String) # e.g., "Trend Following", "Mean Reversion"
    risk_level = Column(String) # e.g., "Low", "Medium", "High"
    historical_performance_summary = Column(Text, nullable=True) # Text or link to report
    is_active = Column(Boolean, default=True) # Admin can deactivate a strategy
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    subscriptions = relationship("UserStrategySubscription", back_populates="strategy")

    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}')>"

class UserStrategySubscription(Base):
    __tablename__ = "user_strategy_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False) # The exchange account to run this on
    custom_parameters = Column(Text) # JSON string or dict of user-defined params
    is_active = Column(Boolean, default=False) # True if currently active and running
    subscribed_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True) # For time-based subscriptions
    backtest_results_id = Column(Integer, ForeignKey("backtest_results.id"), nullable=True) # Optional link to a specific backtest
    status_message = Column(String, nullable=True) # e.g., "Running", "Error: API key invalid", "Paused"
    celery_task_id = Column(String, nullable=True, index=True) # ID of the associated Celery task for live trading
    
    user = relationship("User", back_populates="strategy_subscriptions")
    strategy = relationship("Strategy", back_populates="subscriptions")
    api_key = relationship("ApiKey") # No back_populates needed if ApiKey doesn't need to list subscriptions directly
    # backtest_result = relationship("BacktestResult") # If a subscription is based on one specific backtest

    def __repr__(self):
        return f"<UserStrategySubscription(id={self.id}, user_id={self.user_id}, strategy_id={self.strategy_id}, active={self.is_active})>"

    # Add relationships to Order and Position models
    orders = relationship("Order", back_populates="subscription", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="subscription", cascade="all, delete-orphan")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, ForeignKey("user_strategy_subscriptions.id"), nullable=False)
    order_id = Column(String, index=True, nullable=True) # Exchange's order ID
    symbol = Column(String, nullable=False)
    order_type = Column(String, nullable=False) # e.g., 'limit', 'market', 'stop'
    side = Column(String, nullable=False) # e.g., 'buy', 'sell'
    amount = Column(Float, nullable=False) # Base currency amount
    price = Column(Float, nullable=True) # Price for limit/stop orders
    cost = Column(Float, nullable=True) # Total cost (amount * price)
    filled = Column(Float, nullable=True) # Filled amount
    remaining = Column(Float, nullable=True) # Remaining amount
    status = Column(String, default="open", index=True) # e.g., 'open', 'closed', 'canceled', 'expired'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    closed_at = Column(DateTime, nullable=True) # Timestamp when order was closed/filled/canceled
    raw_order_data = Column(Text, nullable=True) # JSON for exchange-specific details

    subscription = relationship("UserStrategySubscription", back_populates="orders")

    def __repr__(self):
        return f"<Order(id={self.id}, sub_id={self.subscription_id}, symbol='{self.symbol}', side='{self.side}', status='{self.status}')>"

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, ForeignKey("user_strategy_subscriptions.id"), nullable=False)
    symbol = Column(String, nullable=False)
    exchange_name = Column(String, nullable=False) # Exchange where the position is held
    side = Column(String, nullable=False) # 'long' or 'short'
    amount = Column(Float, nullable=False) # Size of the position
    entry_price = Column(Float, nullable=True) # Average entry price
    current_price = Column(Float, nullable=True) # Current market price (needs periodic update)
    is_open = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow) # Time position was opened
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow) # Last updated time
    closed_at = Column(DateTime, nullable=True) # Time position was closed
    pnl = Column(Float, nullable=True) # Profit/Loss when closed

    subscription = relationship("UserStrategySubscription", back_populates="positions")

    def __repr__(self):
        return f"<Position(id={self.id}, sub_id={self.subscription_id}, symbol='{self.symbol}', side='{self.side}', is_open={self.is_open})>"


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # Storing the strategy name/identifier used for the backtest.
    # If strategies are dynamic or versioned, storing strategy_id (FK) might be complex if strategies change.
    # Storing the path or a unique strategy identifier string might be more robust for historical backtests.
    strategy_name_used = Column(String, index=True, nullable=False)
    strategy_code_snapshot = Column(Text, nullable=True) # Optional: snapshot of the strategy code at time of backtest

    custom_parameters_json = Column(Text) # JSON string of parameters
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    timeframe = Column(String, nullable=False)
    symbol = Column(String, index=True, nullable=False)

    # Performance Metrics
    pnl = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)

    trades_log_json = Column(Text) # JSON string for list of trades
    equity_curve_json = Column(Text) # JSON string for equity curve data

    status = Column(String, default="queued", index=True) # e.g., queued, running, completed, failed, cancelled
    celery_task_id = Column(String, nullable=True, index=True) # ID of the associated Celery task
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow) # Add updated_at

    user = relationship("User", back_populates="backtest_results")
    # If UserStrategySubscription can link to a backtest result:
    # subscriptions_using_this_backtest = relationship("UserStrategySubscription", back_populates="backtest_result")


    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy_name_used='{self.strategy_name_used}', user_id={self.user_id}, status='{self.status}', pnl={self.pnl})>"

class PaymentTransaction(Base):
    __tablename__ = "payment_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_strategy_subscription_id = Column(Integer, ForeignKey("user_strategy_subscriptions.id"), nullable=True) # Can be null if it's a general platform payment

    amount_crypto = Column(Float, nullable=False)
    crypto_currency = Column(String, nullable=False)
    usd_equivalent = Column(Float, nullable=True) # Approx USD value at time of transaction

    payment_gateway = Column(String) # e.g., "CoinbaseCommerce", "BitPay", "Manual"
    gateway_transaction_id = Column(String, unique=True, index=True, nullable=True) # ID from the payment gateway
    internal_reference = Column(String, unique=True, index=True, nullable=True) # For manual or internal tracking
    status = Column(String, default="pending", index=True) # pending, completed, failed, refunded
    description = Column(Text, nullable=True) # e.g., "Subscription to EMA Crossover strategy for 1 month"

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="payment_transactions")
    # strategy_subscription = relationship("UserStrategySubscription") # If needed

    def __repr__(self):
        return f"<PaymentTransaction(id={self.id}, user_id={self.user_id}, status='{self.status}')>"

class Referral(Base):
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, index=True)
    referrer_user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # User who made the referral
    referred_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True) # User who was referred (a user can only be referred once)

    signed_up_at = Column(DateTime, default=datetime.datetime.utcnow)
    first_payment_at = Column(DateTime, nullable=True) # Timestamp of first qualifying payment by referred user

    # Commission tracking for this specific referral link
    commission_earned_total = Column(Float, default=0.0) # Total ever earned from this referred user
    commission_pending_payout = Column(Float, default=0.0) # Current amount owed to referrer
    commission_paid_out_total = Column(Float, default=0.0) # Total ever paid out for this referral
    last_payout_date = Column(DateTime, nullable=True)
    is_active_for_commission = Column(Boolean, default=True) # Can be turned off if referral terms change

    referrer = relationship("User", foreign_keys=[referrer_user_id], back_populates="referrals_made")
    # referred_user = relationship("User", foreign_keys=[referred_user_id]) # Avoids issues with multiple relationships to User

    def __repr__(self):
        return f"<Referral(id={self.id}, referrer_id={self.referrer_user_id}, referred_id={self.referred_user_id})>"

# Function to initialize database engine and session (call this from your main app setup)
def init_db(database_url: str):
    global engine, SessionLocal
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # Base.metadata.create_all(bind=engine) # Creates tables. Be cautious with this in production.
                                          # Use migrations (e.g., Alembic) for schema changes.
    return engine

# Example of how to create tables (typically done once or via migrations)
# if __name__ == "__main__":
#     # This is for demonstration; in a real app, manage DB creation/migration carefully.
#     db_url = "sqlite:///./trading_platform_dev.db"
#     print(f"Initializing database at {db_url} and creating tables if they don't exist...")
#     init_db(db_url)
#     Base.metadata.create_all(bind=engine)
#     print("Database tables should be created (if they weren't already).")
#     print("Note: `engine` and `SessionLocal` are None until `init_db` is called from your application.")

print("models.py loaded. Call init_db(DATABASE_URL) from your main application to set up the database engine.")
