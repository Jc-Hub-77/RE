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
from backend.utils import _load_strategy_class_from_db_obj # Import from utils
from backend.services.exchange_service import fetch_historical_data
from sqlalchemy.orm import Session
from backend.celery_app import celery_app # Import celery app
from backend.tasks import run_backtest_task # Import the Celery task

# --- Configuration ---
MAX_BACKTEST_DAYS = 366 # Maximum backtest period allowed (e.g., 1 year)

# Initialize logger
logger = logging.getLogger(__name__)

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
                            initial_capital: float = 10000.0,
                            exchange_id: str = 'binance'
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

    if (end_date - start_date).days > MAX_BACKTEST_DAYS:
        logger.error(f"Backtest period exceeds max allowed days ({MAX_BACKTEST_DAYS}): {start_date_str} to {end_date_str}")
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": f"Backtest period cannot exceed {MAX_BACKTEST_DAYS} days."}
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
    except Exception as e:
        logger.error(f"Failed to fetch historical data for backtest: {e}", exc_info=True)
        backtest_record.status = "failed"
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
        logger.error(f"Error initializing strategy '{strategy_db_obj.name}' (ID: {strategy_id}) for backtest: {e}", exc_info=True)
        backtest_record.status = "failed"
        db_session.commit()
        return {"status": "error", "message": f"Error initializing strategy: {str(e)}"}

    # 4. Run the strategy's backtest method
    try:
        backtest_output = strategy_instance.run_backtest(historical_df)
    except Exception as e:
        logger.error(f"Error during strategy's run_backtest method for '{strategy_db_obj.name}' (ID: {strategy_id}): {e}", exc_info=True)
        backtest_record.status = "failed"
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
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error updating backtest results for ID {backtest_record.id}: {e}", exc_info=True)
        backtest_record.status = "failed" # Set status to failed on error
        db_session.commit()
        return {"status": "error", "message": "Database error updating backtest results."}

# --- Service Function to Queue Backtest ---
def run_backtest(db_session: Session, # Use DB session directly
                 user_id: int,
                 strategy_id: int,
                 custom_parameters: dict,
                 symbol: str,
                 timeframe: str,
                 start_date_str: str,
                 end_date_str: str,
                 initial_capital: float = 10000.0,
                 exchange_id: str = 'binance'
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

    if (end_date - start_date).days > MAX_BACKTEST_DAYS:
        return {"status": "error", "message": f"Backtest period cannot exceed {MAX_BACKTEST_DAYS} days."}
    if start_date >= end_date:
        return {"status": "error", "message": "Start date must be before end date."}

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
            initial_capital=initial_capital,
            exchange_id=exchange_id
        )
        logger.info(f"Queued backtest task for user {user_id}, strategy {strategy_id}. Task ID: {task.id}")

        # Update the BacktestResult record with the task ID
        backtest_record.celery_task_id = task.id
        db_session.commit()

        return {"status": "success", "message": "Backtest task queued.", "backtest_id": backtest_record.id, "task_id": task.id}

    except Exception as e:
        db_session.rollback() # Rollback the initial record creation if task queuing fails
        logger.error(f"Failed to queue backtest task for user {user_id}, strategy {strategy_id}: {e}", exc_info=True)
        # Update the BacktestResult record status to 'failed_to_queue'
        backtest_record.status = "failed_to_queue"
        db_session.commit()
        return {"status": "error", "message": f"Failed to queue backtest task: {e}"}

# Note: The _load_strategy_class helper function is assumed to be defined elsewhere or needs to be added.
# Based on live_trading_service, it seems _load_strategy_class_from_db_obj is the correct function to use.
# Let's ensure that's used consistently.

# Correcting the internal strategy loading function call
# The original code had `_load_strategy_class`, but the import is `_load_strategy_class_from_db_obj`.
# I will use `_load_strategy_class_from_db_obj`.

# TODO: Implement a secure secrets management solution for API_ENCRYPTION_KEY in production.
# This key is critical for decrypting API keys for live trading.
