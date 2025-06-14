# backend/services/backtesting_service.py
import datetime
import json
import ccxt
import importlib.util
import os
import pandas as pd # For data handling and calculations
import numpy as np  # For numerical operations
import logging

from backend.models import BacktestResult, Strategy as StrategyModel
from backend.utils import _load_strategy_class_from_db_obj, get_system_setting # Import from utils
from backend.services.exchange_service import fetch_historical_data
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError # Added for DB errors
from backend.celery_app import celery_app # Import celery app
from backend.tasks import run_backtest_task # Import the Celery task
from backend.models import BacktestResult, Strategy as StrategyModel, User # Added User for potential joins
from sqlalchemy import desc # Added for ordering
from typing import Optional, Dict, Any, List # Ensure these are imported


# Initialize logger
logger = logging.getLogger(__name__)
# MAX_BACKTEST_DAYS is now fetched dynamically

# --- Core Backtesting Logic (Helper Function) ---
def _perform_backtest_logic(db_session: Session,
                            backtest_result_id: int, # Added backtest_result_id
                            user_id: int,
                            strategy_id: int,
                            custom_parameters: dict,
                            symbol: str,
                            timeframe: str,
                            start_date_str: str,
                            end_date_str: str,
                            initial_capital: float, # Default removed, will be passed by run_backtest
                            exchange_id: str # Default removed, will be passed by run_backtest
                           ):
    """
    Performs the core backtesting logic. Designed to be called by a Celery task.
    Updates the BacktestResult status in the database.
    """
    backtest_record = db_session.query(BacktestResult).filter(BacktestResult.id == backtest_result_id).first()
    if not backtest_record:
        logger.error(f"BacktestResult record with ID {backtest_result_id} not found.")
        # Cannot update status if record is not found
        return {"status": "error", "message": f"BacktestResult record with ID {backtest_result_id} not found."}

    try:
        # Update status to running
        backtest_record.status = "running"
        db_session.commit()

        start_date = datetime.datetime.fromisoformat(start_date_str)
        end_date = datetime.datetime.fromisoformat(end_date_str)
    except ValueError:
        logger.error(f"Invalid date format for backtest: {start_date_str} to {end_date_str}")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": "Invalid date format."}

    # Fetch MAX_BACKTEST_DAYS from system settings
    max_days_str = get_system_setting(db_session, "MAX_BACKTEST_DAYS_SYSTEM", str(366))
    try:
        current_max_backtest_days = int(max_days_str)
    except ValueError:
        logger.warning(f"Could not parse MAX_BACKTEST_DAYS_SYSTEM ('{max_days_str}') from SystemSetting. Using default 366.")
        current_max_backtest_days = 366

    if (end_date - start_date).days > current_max_backtest_days:
        logger.error(f"Backtest period exceeds max allowed days ({current_max_backtest_days}): {start_date_str} to {end_date_str}")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": f"Backtest period cannot exceed {current_max_backtest_days} days."}
    if start_date >= end_date:
        logger.error(f"Start date is not before end date: {start_date_str} to {end_date_str}")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": "Start date must be before end date."}

    # 1. Load Strategy Class
    strategy_db_obj = db_session.query(StrategyModel).filter(StrategyModel.id == strategy_id, StrategyModel.is_active == True).first()
    if not strategy_db_obj:
        logger.error(f"Strategy with ID '{strategy_id}' not found or is not active for backtest.")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": f"Strategy with ID '{strategy_id}' not found or is not active."}

    StrategyClass = _load_strategy_class_from_db_obj(strategy_db_obj)
    if not StrategyClass:
        logger.error(f"Could not load strategy class for {strategy_db_obj.name} (ID: {strategy_id}).")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": f"Could not load strategy class for {strategy_db_obj.name}."}

    # Update strategy name in the record
    backtest_record.strategy_name_used = strategy_db_obj.name
    db_session.commit()


    # 2. Fetch Historical Data
    try:
        historical_df = fetch_historical_data(exchange_id, symbol, timeframe, start_date, end_date)
        if historical_df.empty:
            logger.warning(f"No historical data found for {symbol}@{timeframe} from {start_date_str} to {end_date_str} on {exchange_id}.")
            backtest_record.status = "no_data" # Use 'no_data' status
            db_session.commit()
            return {"status": "error", "message": "No historical data found for the given parameters."}
    except Exception as e: # fetch_historical_data has its own error handling, this is a fallback
        logger.exception(f"Unexpected error calling fetch_historical_data for backtest BR_ID {backtest_result_id}: {e}")
        backtest_record.status = "failed"
        backtest_record.status_message = f"Data Fetch Error: {str(e)[:150]}"
        db_session.commit()
        return {"status": "error", "message": f"Failed to fetch historical data: {str(e)}"}

    # 3. Instantiate Strategy
    strategy_params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "capital": initial_capital,
        **custom_parameters
    }
    try:
        strategy_instance = StrategyClass(**strategy_params)
    except Exception as e:
        logger.exception(f"Error initializing strategy '{strategy_db_obj.name}' (ID: {strategy_id}) for backtest BR_ID {backtest_result_id}: {e}")
        backtest_record.status = "failed"
        backtest_record.status_message = f"Strategy Init Error: {str(e)[:150]}"
        db_session.commit()
        return {"status": "error", "message": f"Error initializing strategy: {str(e)}"}

    # 4. Run the strategy's backtest method
    try:
        backtest_output = strategy_instance.run_backtest(historical_df)
    except Exception as e:
        logger.exception(f"Error during strategy's run_backtest method for '{strategy_db_obj.name}' (ID: {strategy_id}), BR_ID {backtest_result_id}: {e}")
        backtest_record.status = "failed"
        backtest_record.status_message = f"Strategy Execution Error: {str(e)[:150]}"
        db_session.commit()
        return {"status": "error", "message": f"Error executing strategy backtest: {str(e)}"}

    # 5. Process results from the strategy's output
    pnl = backtest_output.get("pnl", 0.0)
    trades_log = backtest_output.get("trades", [])
    sharpe_ratio = backtest_output.get("sharpe_ratio", 0.0)
    max_drawdown = backtest_output.get("max_drawdown", 0.0)
    total_trades = len(trades_log)
    winning_trades = sum(1 for t in trades_log if t.get("pnl", 0) > 0)
    losing_trades = total_trades - winning_trades

    # Generate equity curve
    equity_curve = []
    if not historical_df.empty:
        equity_timestamps = historical_df.index.astype(np.int64) // 10**6 # Milliseconds

        pnl_at_time = {}
        cumulative_pnl = 0
        sorted_trades_for_equity = sorted(trades_log, key=lambda t: t.get('exit_time', t.get('entry_time', 0)))

        for trade in sorted_trades_for_equity:
            trade_pnl = trade.get("pnl", 0)
            cumulative_pnl += trade_pnl
            time_key = trade.get('exit_time', trade.get('entry_time'))
            if time_key:
                 pnl_at_time[int(time_key * 1000 if isinstance(time_key, (int, float)) and time_key < 1e12 else time_key)] = cumulative_pnl

        last_recorded_pnl = 0
        for ts_millis in equity_timestamps:
            relevant_pnl_times = [t for t in pnl_at_time.keys() if t <= ts_millis]
            if relevant_pnl_times:
                last_recorded_pnl = pnl_at_time[max(relevant_pnl_times)]

            equity_curve.append([ts_millis, round(initial_capital + last_recorded_pnl, 2)])

        if not trades_log and not historical_df.empty:
            equity_curve = [[ts_millis, round(initial_capital, 2)] for ts_millis in equity_timestamps]

    # 6. Update results in the existing record
    backtest_record.pnl = pnl
    backtest_record.sharpe_ratio = sharpe_ratio
    backtest_record.max_drawdown = max_drawdown
    backtest_record.total_trades = total_trades
    backtest_record.winning_trades = winning_trades
    backtest_record.losing_trades = losing_trades
    backtest_record.trades_log_json = json.dumps(trades_log)
    backtest_record.equity_curve_json = json.dumps(equity_curve)
    backtest_record.status = "completed" # Set status to completed
    backtest_record.updated_at = datetime.datetime.utcnow() # Update timestamp

    try:
        db_session.commit()
        logger.info(f"Backtest result updated for ID: {backtest_record.id} for user {user_id}, strategy {strategy_id}.")
        return {
            "status": "success",
            "message": "Backtest completed and results stored.",
            "backtest_id": backtest_record.id,
            # Include other summary data if needed by the task result
        }
    except SQLAlchemyError as sae:
        db_session.rollback()
        logger.exception(f"SQLAlchemyError updating backtest results for BR_ID {backtest_record.id}: {sae}")
        # Attempt to update status to failed, but this might also fail if DB is down
        try:
            backtest_record.status = "failed" 
            backtest_record.status_message = f"DB Error Saving Results: {str(sae)[:100]}"
            db_session.commit()
        except Exception: # Re-rollback if status update fails
            db_session.rollback()
        return {"status": "error", "message": "Database error updating backtest results."}
    except Exception as e: # General fallback for unexpected errors during result processing/commit
        db_session.rollback()
        logger.exception(f"Unexpected error updating backtest results for BR_ID {backtest_record.id}: {e}")
        try:
            backtest_record.status = "failed"
            backtest_record.status_message = f"Unexpected Error Saving Results: {str(e)[:100]}"
            db_session.commit()
        except Exception:
            db_session.rollback()
        return {"status": "error", "message": "Unexpected error updating backtest results."}


# --- Service Function to Queue Backtest ---
def run_backtest(db_session: Session, # Use DB session directly
                 user_id: int,
                 strategy_id: int,
                 custom_parameters: dict,
                 symbol: str,
                 timeframe: str,
                 start_date_str: str,
                 end_date_str: str,
                 initial_capital: Optional[float] = None,
                 exchange_id: Optional[str] = None
                ):
    """
    Queues a backtest task to be run by a Celery worker.
    Creates a BacktestResult record with 'queued' status.
    """
    # Basic validation before queuing
    try:
        start_date = datetime.datetime.fromisoformat(start_date_str)
        end_date = datetime.datetime.fromisoformat(end_date_str)
    except ValueError:
        return {"status": "error", "message": "Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)."}

    # Fetch MAX_BACKTEST_DAYS from system settings
    max_days_str = get_system_setting(db_session, "MAX_BACKTEST_DAYS_SYSTEM", str(366))
    try:
        current_max_backtest_days = int(max_days_str)
    except ValueError:
        logger.warning(f"Could not parse MAX_BACKTEST_DAYS_SYSTEM ('{max_days_str}') from SystemSetting for pre-queue check. Using default 366.")
        current_max_backtest_days = 366

    if (end_date - start_date).days > current_max_backtest_days:
        return {"status": "error", "message": f"Backtest period cannot exceed {current_max_backtest_days} days."}
    if start_date >= end_date:
        return {"status": "error", "message": "Start date must be before end date."}

    # Get dynamic defaults for initial_capital and exchange_id
    final_initial_capital = initial_capital
    if final_initial_capital is None:
        try:
            default_initial_capital_str = get_system_setting(db_session, "DEFAULT_BACKTEST_INITIAL_CAPITAL", "10000.0")
            final_initial_capital = float(default_initial_capital_str)
        except ValueError:
            logger.warning(f"Could not parse DEFAULT_BACKTEST_INITIAL_CAPITAL ('{default_initial_capital_str}') from SystemSetting. Using hardcoded default 10000.0.")
            final_initial_capital = 10000.0
    
    final_exchange_id = exchange_id
    if final_exchange_id is None:
        final_exchange_id = get_system_setting(db_session, "DEFAULT_BACKTEST_EXCHANGE_ID", "binance")


    # Create a BacktestResult record in the DB with status 'queued'
    backtest_record = BacktestResult(
        user_id=user_id,
        strategy_name_used=str(strategy_id), # Placeholder, will be updated by task
        custom_parameters_json=json.dumps(custom_parameters),
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        symbol=symbol,
        status="queued"
    )
    db_session.add(backtest_record)
    db_session.commit()
    db_session.refresh(backtest_record)

    try:
        # Send the task to the Celery queue
        task = run_backtest_task.delay(
            backtest_result_id=backtest_record.id, # Pass the new record ID to the task
            user_id=user_id,
            strategy_id=strategy_id,
            custom_parameters=custom_parameters,
            symbol=symbol,
            timeframe=timeframe,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            initial_capital=final_initial_capital,
            exchange_id=final_exchange_id
        )
        logger.info(f"Queued backtest task for user {user_id}, strategy {strategy_id}. Task ID: {task.id}")

        # Update the BacktestResult record with the task ID
        backtest_record.celery_task_id = task.id
        db_session.commit()

        return {"status": "success", "message": "Backtest task queued.", "backtest_id": backtest_record.id, "task_id": task.id}
    except CeleryOperationalError as coe: # More specific error for Celery
        db_session.rollback() 
        logger.error(f"Celery OperationalError queuing backtest task for user {user_id}, strategy {strategy_id}: {coe}", exc_info=True)
        backtest_record.status = "failed_to_queue"; backtest_record.status_message = f"Celery Error: {str(coe)[:100]}"
        db_session.commit() # Try to commit status update
        return {"status": "error", "message": f"Failed to queue backtest task due to Celery operational error: {coe}"}
    except Exception as e: # General fallback
        db_session.rollback() 
        logger.exception(f"Unexpected error queuing backtest task for user {user_id}, strategy {strategy_id}: {e}")
        backtest_record.status = "failed_to_queue"; backtest_record.status_message = f"Queueing Error: {str(e)[:100]}"
        db_session.commit() # Try to commit status update
        return {"status": "error", "message": f"Failed to queue backtest task: {e}"}

# Note: The _load_strategy_class helper function is assumed to be defined elsewhere or needs to be added.
# Based on live_trading_service, it seems _load_strategy_class_from_db_obj is the correct function to use.
# Let's ensure that's used consistently.

# IMPORTANT: API_ENCRYPTION_KEY is critical for securing sensitive data. In production environments, 
# ensure this key (and other secrets like database credentials, JWT secrets) is managed via a secure 
# secrets management system (e.g., environment variables injected from HashiCorp Vault, 
# AWS Secrets Manager, Azure Key Vault, or similar) and is NOT hardcoded or committed to version control.
# While this service is for backtesting, API_ENCRYPTION_KEY is a system-wide concern.

def get_backtest_result_by_id(db: Session, backtest_id: int, user_id: Optional[int] = None) -> Dict[str, Any]:
    #"""
    #Retrieves a specific backtest result by its ID.
    #If user_id is provided, it also ensures the backtest belongs to that user.
    #"""
    query = db.query(BacktestResult).filter(BacktestResult.id == backtest_id)
    if user_id is not None:
        query = query.filter(BacktestResult.user_id == user_id)

    result = query.first()

    if not result:
        logger.warning(f"Backtest result ID {backtest_id} not found {'for user ' + str(user_id) if user_id else ''}.")
        return {"status": "error", "message": "Backtest result not found or access denied."}

    # Serialize the result. Assuming BacktestResultResponse schema expects these fields.
    # Adjust serialization as needed based on the actual Pydantic schema.
    serialized_result = {
        "id": result.id,
        "user_id": result.user_id,
        "strategy_name_used": result.strategy_name_used,
        "custom_parameters_json": result.custom_parameters_json, # Keep as JSON string
        "start_date": result.start_date.isoformat() if result.start_date else None,
        "end_date": result.end_date.isoformat() if result.end_date else None,
        "timeframe": result.timeframe,
        "symbol": result.symbol,
        "pnl": result.pnl,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "trades_log_json": result.trades_log_json, # Keep as JSON string
        "equity_curve_json": result.equity_curve_json, # Keep as JSON string
        "status": result.status,
        "celery_task_id": result.celery_task_id,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
    }
    logger.info(f"Retrieved backtest result ID {backtest_id} {'for user ' + str(user_id) if user_id else ''}.")
    return {"status": "success", "result": serialized_result}


def list_backtest_results(db: Session, user_id: Optional[int] = None, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
    #"""
    #Lists backtest results with pagination.
    #If user_id is provided, filters results for that user. Otherwise, lists all (for admin).
    #"""
    query = db.query(BacktestResult)
    if user_id is not None:
        query = query.filter(BacktestResult.user_id == user_id)

    total_items = query.count()

    results_page = query.order_by(desc(BacktestResult.created_at)) \
                        .offset((page - 1) * per_page) \
                        .limit(per_page) \
                        .all()

    serialized_list = []
    for result in results_page:
        serialized_list.append({
            "id": result.id,
            "user_id": result.user_id,
            "strategy_name_used": result.strategy_name_used,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "start_date": result.start_date.isoformat() if result.start_date else None,
            "end_date": result.end_date.isoformat() if result.end_date else None,
            "pnl": result.pnl,
            "status": result.status,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "updated_at": result.updated_at.isoformat() if result.updated_at else None,
        })

    total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 0

    logger.info(f"Listed backtest results page {page} {'for user ' + str(user_id) if user_id else '(all users)'}. Found {total_items} total.")
    return {
        "status": "success",
        "results": serialized_list,
        "total_items": total_items,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }
