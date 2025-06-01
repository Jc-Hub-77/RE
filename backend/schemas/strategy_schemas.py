# backend/schemas/strategy_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime

# --- Strategy Schemas (User-Facing) ---
class StrategyAvailableView(BaseModel):
    id: int # Database ID
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    risk_level: Optional[str] = None
    historical_performance_summary: Optional[str] = None

    class Config:
        model_config = { "from_attributes": True }

class StrategyAvailableListResponse(BaseModel):
    status: str
    strategies: List[StrategyAvailableView]

class PaymentOption(BaseModel):
    months: int
    price_usd: float
    description: Optional[str] = None

class StrategyParameterDefinition(BaseModel): # Describes a single parameter
    # This structure depends on how strategy classes define their params.
    # Example:
    type: str # e.g., "int", "float", "str", "bool", "choice"
    label: str
    default: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    choices: Optional[List[Any]] = None # For "choice" type
    description: Optional[str] = None

class StrategyDetailView(StrategyAvailableView):
    parameters_definition: Dict[str, StrategyParameterDefinition] = {} # Defines structure and types of params
    default_parameters_db: Dict[str, Any] = {} # Actual default values from DB (JSON parsed)
    payment_options: Optional[List[PaymentOption]] = None

class StrategyDetailResponse(BaseModel):
    status: str
    details: Optional[StrategyDetailView] = None
    message: Optional[str] = None # For errors

# --- User Strategy Subscription Schemas ---
class UserStrategySubscriptionCreateRequest(BaseModel):
    strategy_db_id: int = Field(..., alias="strategyId") # ID from the Strategy database table
    api_key_id: int = Field(..., alias="apiKeyId")
    custom_parameters: Dict[str, Any] = Field(..., alias="customParameters")
    subscription_months: int = Field(1, ge=1, le=12, alias="subscriptionMonths")

class UserStrategySubscriptionResponseData(BaseModel):
    subscription_id: int
    strategy_id: int # DB ID of the strategy
    strategy_name: str
    api_key_id: int
    custom_parameters: Dict[str, Any]
    is_active: bool
    status_message: Optional[str] = None
    subscribed_at: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None # Or str if formatted
    time_remaining_seconds: Optional[int] = None

    class Config:
        model_config = { "from_attributes": True } # If mapping from UserStrategySubscription model directly

class UserStrategySubscriptionActionResponse(BaseModel):
    status: str
    message: str
    subscription_id: Optional[int] = None
    expires_at: Optional[str] = None # ISO format string

class UserStrategySubscriptionListResponse(BaseModel):
    status: str
    subscriptions: List[UserStrategySubscriptionResponseData]

class BacktestResultResponse(BaseModel):
    status: str
    message: str
    backtest_id: int
    strategy_id: int
    symbol: str
    timeframe: str
    period: str
    initial_capital: float
    final_equity: float
    pnl: float
    pnl_percentage: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    custom_parameters_used: Dict[str, Any]
    trades_log: List[Dict[str, Any]]
    equity_curve: List[List[Any]]

class AdminBacktestListResponse(BaseModel):
    status: str
    backtests: List[BacktestResultResponse] # Assuming admin list returns full results
    # Could add pagination/total fields if the service layer supports it
