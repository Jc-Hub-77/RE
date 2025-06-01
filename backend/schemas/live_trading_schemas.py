# backend/schemas/live_trading_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime

class LiveStrategyStatus(BaseModel):
    subscription_id: int
    strategy_name: str
    symbol: str
    is_alive: bool
    status_message: Optional[str] = None # Could be 'Running', 'Error', 'Stopped', etc.

class RunningStrategiesResponse(BaseModel):
    status: str
    running_strategies: List[LiveStrategyStatus]
    message: Optional[str] = None

class StrategyActionResponse(BaseModel): # General response for deploy/stop actions
    status: str
    message: str
    subscription_id: Optional[int] = None

# No specific request schemas needed if actions are based on subscription ID path parameter.
