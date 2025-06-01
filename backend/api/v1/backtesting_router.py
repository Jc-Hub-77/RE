# backend/api/v1/backtesting_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from ...schemas import strategy_schemas
from ...services import backtesting_service
from backend.models import User
from backend.db import get_db
from .auth_router import get_current_active_user # Dependency for protected routes
from .admin_router import get_current_active_admin_user # Import admin dependency

router = APIRouter()

@router.post("/backtests", response_model=Dict[str, Any]) # Define a more specific response model if possible
async def run_backtest_endpoint(
    backtest_params: strategy_schemas.UserStrategySubscriptionCreateRequest, # Re-use the subscription request
    start_date: str,
    end_date: str,
    symbol: str, # Make symbol configurable
    timeframe: str, # Make timeframe configurable
    initial_capital: float = 10000.0,
    exchange_id: str = 'binance',
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Runs a backtest for a given strategy.
    """
    # The UserStrategySubscriptionCreateRequest already contains strategy_id, api_key_id, custom_parameters
    # We can pass these directly to the backtesting service.
    result = backtesting_service.run_backtest(
        db_session=db,
        user_id=current_user.id,
        strategy_id=backtest_params.strategy_db_id,
        custom_parameters=backtest_params.custom_parameters,
        symbol=symbol, # Use the configurable symbol
        timeframe=timeframe, # Use the configurable timeframe
        start_date_str=start_date,
        end_date_str=end_date,
        initial_capital=initial_capital,
        exchange_id=exchange_id
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.get("/backtests/{backtest_id}", response_model=strategy_schemas.BacktestResultResponse) # Define a specific response model
async def get_user_backtest_result(
    backtest_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves a specific backtest result for the authenticated user by ID.
    """
    result = backtesting_service.get_backtest_result_by_id(db, backtest_id, user_id=current_user.id)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
    return result

@router.get("/admin/backtests", response_model=strategy_schemas.AdminBacktestListResponse, dependencies=[Depends(get_current_active_admin_user)]) # Define a specific response model
async def admin_list_all_backtest_results(
    db: Session = Depends(get_db),
    # Add pagination/filtering/sorting queries if needed
):
    """
    Admin endpoint to list all backtest results.
    """
    result = backtesting_service.list_all_backtest_results(db)
    if result["status"] == "error": # Should not happen if service layer is robust
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error listing backtest results"))
    return result
