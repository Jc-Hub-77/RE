# backend/api/v1/backtesting_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Any # Changed from Dict, Any

from ...schemas import strategy_schemas # Keep this as is
# Import the new schemas directly
from ...schemas.strategy_schemas import BacktestRunRequest, BacktestRunResponse
from ...services import backtesting_service
from backend.models import User
from backend.db import get_db
from .auth_router import get_current_active_user # Dependency for protected routes
from .admin_router import get_current_active_admin_user # Import admin dependency

router = APIRouter()

@router.post("/backtests", response_model=BacktestRunResponse) # Use the new response model
async def run_backtest_endpoint(
    payload: BacktestRunRequest, # Use the new request schema for the payload
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Runs a backtest for a given strategy using parameters from the request body.
    """
    result = backtesting_service.run_backtest(
        db_session=db,
        user_id=current_user.id,
        strategy_id=payload.strategy_db_id, # strategy_id from payload
        custom_parameters=payload.custom_parameters,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date_str=payload.start_date_str,
        end_date_str=payload.end_date_str,
        initial_capital=payload.initial_capital,
        exchange_id=payload.exchange_id
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])

    # The service returns a dict that should match BacktestRunResponse fields
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
    page: int = Query(1, alias="page", ge=1),
    per_page: int = Query(20, alias="per_page", ge=1, le=100)
    # Add pagination/filtering/sorting queries if needed
):
    """
    Admin endpoint to list all backtest results.
    """
    # Call the new generic list_backtest_results with user_id=None for admin view
    result = backtesting_service.list_backtest_results(
        db=db,
        user_id=None,
        page=page,
        per_page=per_page
    )
    if result["status"] == "error": # Should not happen if service layer is robust
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error listing backtest results"))
    return result
