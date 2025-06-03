import datetime
import time
import ccxt
import json
import os
import importlib.util
import logging
import pandas as pd # Added for DataFrame conversion
import smtplib # Added for email
import socket # For socket.gaierror in email task
from email.mime.text import MIMEText # Added for email

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from celery.exceptions import OperationalError as CeleryOperationalError # Though tasks usually don't handle their own send errors
from backend.celery_app import celery_app
from backend.models import UserStrategySubscription, ApiKey, User, Strategy as StrategyModel, BacktestResult # Added BacktestResult
from backend.utils import _load_strategy_class_from_db_obj # Import from utils
from backend.services.exchange_service import _decrypt_data # Assuming this is preferred over full service for this direct action
from backend.config import settings 
from backend.db import SessionLocal 

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Celery Task Definitions ---

@celery_app.task(bind=True)
def run_live_strategy(self, user_sub_id: int):
    """
    Celery task to run a live trading strategy for a specific subscription.
    """
    db_session = None
    try:
        db_session = SessionLocal() 

        user_sub = db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_sub_id).first()
        if not user_sub or not user_sub.is_active or \
           (user_sub.expires_at and user_sub.expires_at <= datetime.datetime.utcnow()):
            logger.info(f"[SubID {user_sub_id}] Subscription not found, inactive, or expired. Stopping task.")
            if user_sub and user_sub.is_active: 
                user_sub.is_active = False
                user_sub.status_message = "Stopped: Subscription expired or deactivated."
                db_session.commit()
            return {"status": "stopped", "message": "Subscription inactive or expired."}

        strategy_db_obj = db_session.query(StrategyModel).filter(StrategyModel.id == user_sub.strategy_id).first()
        if not strategy_db_obj:
            logger.error(f"[SubID {user_sub_id}] Strategy DB object ID {user_sub.strategy_id} not found.")
            user_sub.status_message = "Error: Strategy not found."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Strategy not found."}

        StrategyClass = _load_strategy_class_from_db_obj(strategy_db_obj)
        if not StrategyClass:
            logger.error(f"[SubID {user_sub_id}] Could not load strategy class for {strategy_db_obj.name}.")
            user_sub.status_message = "Error: Could not load strategy class."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Could not load strategy class."}

        try:
            # Ensure custom_params is a dict, even if None in DB
            custom_params = json.loads(user_sub.custom_parameters) if isinstance(user_sub.custom_parameters, str) else (user_sub.custom_parameters or {})
        except json.JSONDecodeError:
            logger.error(f"[SubID {user_sub_id}] Invalid JSON in custom_parameters: {user_sub.custom_parameters}.")
            user_sub.status_message = "Error: Invalid strategy parameters."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Invalid strategy parameters."}
        
        # Resolve capital: Use from custom_params if present, else use a default from settings or a fallback.
        capital_for_strategy = custom_params.get("capital", getattr(settings, 'DEFAULT_STRATEGY_CAPITAL', 10000))
        
        # Ensure symbol and timeframe are present in custom_params
        symbol = custom_params.get("symbol")
        timeframe = custom_params.get("timeframe")

        if not symbol:
            logger.error(f"[SubID {user_sub_id}] 'symbol' is missing from custom_parameters.")
            user_sub.status_message = "Error: Strategy parameter 'symbol' is missing."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Strategy parameter 'symbol' is missing."}
        
        if not timeframe:
            logger.error(f"[SubID {user_sub_id}] 'timeframe' is missing from custom_parameters.")
            user_sub.status_message = "Error: Strategy parameter 'timeframe' is missing."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Strategy parameter 'timeframe' is missing."}
        
        # Prepare init_params for StrategyClass constructor
        # Pass all custom_parameters, plus resolved capital, symbol, and timeframe.
        init_params = {
            "symbol": symbol, 
            "timeframe": timeframe,
            "capital": capital_for_strategy, 
            **custom_params 
        }
        
        try:
            strategy_instance = StrategyClass(**init_params)
            # strategy_instance.name = strategy_db_obj.name # Strategy class should set its own name
        except Exception as e:
            logger.error(f"[SubID {user_sub_id}] Error initializing strategy class '{strategy_db_obj.name}': {e}", exc_info=True)
            user_sub.status_message = f"Error initializing strategy: {str(e)[:150]}"; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": f"Error initializing strategy: {e}"}

        api_key_record = db_session.query(ApiKey).filter(ApiKey.id == user_sub.api_key_id, ApiKey.user_id == user_sub.user_id).first()
        if not api_key_record or api_key_record.status != "active":
            logger.error(f"[SubID {user_sub_id}] API Key ID {user_sub.api_key_id} not found or not active.")
            user_sub.status_message = "Error: API Key not found or inactive."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "API Key not found or inactive."}

        try:
            decrypted_key_public = _decrypt_data(api_key_record.encrypted_api_key)
            decrypted_secret = _decrypt_data(api_key_record.encrypted_secret_key)
            decrypted_passphrase = _decrypt_data(api_key_record.encrypted_passphrase) if api_key_record.encrypted_passphrase else None
        except ValueError as e:
            logger.error(f"[SubID {user_sub_id}] Failed to decrypt API credentials for key ID {user_sub.api_key_id}: {e}.")
            user_sub.status_message = "Error: Failed to decrypt API credentials."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Failed to decrypt API credentials."}

        exchange_id_str = api_key_record.exchange_name.lower()
        if not hasattr(ccxt, exchange_id_str):
            logger.error(f"[SubID {user_sub_id}] Exchange {exchange_id_str} not supported by CCXT.")
            user_sub.status_message = "Error: Exchange not supported."; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": "Exchange not supported."}

        exchange_class = getattr(ccxt, exchange_id_str)
        ccxt_config = {
            'apiKey': decrypted_key_public, 'secret': decrypted_secret,
            'options': {'adjustForTimeDifference': True}, 'enableRateLimit': True,
        }
        if decrypted_passphrase: ccxt_config['password'] = decrypted_passphrase

        try:
            exchange_ccxt = exchange_class(ccxt_config) 
            exchange_ccxt.check_required_credentials()
            logger.info(f"[SubID {user_sub_id}] Initialized CCXT exchange '{exchange_id_str}' for strategy '{strategy_instance.name}'.")
        except Exception as e:
            logger.error(f"[SubID {user_sub_id}] Failed to initialize CCXT for '{exchange_id_str}': {e}", exc_info=True)
            user_sub.status_message = f"Error initializing exchange: {str(e)[:150]}"; user_sub.is_active = False; db_session.commit()
            return {"status": "error", "message": f"Failed to initialize exchange: {e}"}

        logger.info(f"[SubID {user_sub_id}] Task started for strategy '{strategy_instance.name}' on symbol '{init_params['symbol']}'.")

        while not self.request.is_terminated:
            current_sub_for_loop = db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_sub_id).first() 
            if not current_sub_for_loop or not current_sub_for_loop.is_active or \
               (current_sub_for_loop.expires_at and current_sub_for_loop.expires_at <= datetime.datetime.utcnow()):
                logger.info(f"[SubID {user_sub_id}] Subscription loop: Inactive or expired. Stopping.")
                if current_sub_for_loop and current_sub_for_loop.is_active:
                    current_sub_for_loop.is_active = False; current_sub_for_loop.status_message = "Stopped: Expired/deactivated during run."; db_session.commit()
                break
            
            current_sub_for_loop.status_message = f"Running - Last cycle check: {datetime.datetime.utcnow().isoformat()}"
            db_session.commit() 

            logger.debug(f"[SubID {user_sub_id}] Starting strategy cycle for {init_params['symbol']}@{init_params['timeframe']}.")

            market_data_df = None 
            try:
                ohlcv = exchange_ccxt.fetch_ohlcv(init_params['symbol'], init_params['timeframe'], limit=200) 
                if ohlcv:
                    market_data_df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']) 
                    market_data_df['timestamp'] = pd.to_datetime(market_data_df['timestamp'], unit='ms')
                    market_data_df.set_index('timestamp', inplace=True)
                    logger.debug(f"[SubID {user_sub_id}] Fetched {len(market_data_df)} candles for {init_params['symbol']}.")
                else: 
                    logger.warning(f"[SubID {user_sub_id}] No OHLCV data fetched for {init_params['symbol']}@{init_params['timeframe']}.")
            except ccxt.NetworkError as ne: # More specific CCXT errors
                logger.error(f"[SubID {user_sub_id}] CCXT NetworkError fetching market data for {init_params['symbol']}: {ne}", exc_info=True)
                current_sub_for_loop.status_message = f"Running - Data fetch network error: {str(ne)[:100]}"; db_session.commit()
            except ccxt.ExchangeError as ee: # More specific CCXT errors
                logger.error(f"[SubID {user_sub_id}] CCXT ExchangeError fetching market data for {init_params['symbol']}: {ee}", exc_info=True)
                current_sub_for_loop.status_message = f"Running - Data fetch exchange error: {str(ee)[:100]}"; db_session.commit()
            except ccxt.BaseError as be: # Catch other CCXT base errors
                logger.error(f"[SubID {user_sub_id}] CCXT BaseError fetching market data for {init_params['symbol']}: {be}", exc_info=True)
                current_sub_for_loop.status_message = f"Running - Data fetch CCXT error: {str(be)[:100]}"; db_session.commit()
            except Exception as e: 
                logger.exception(f"[SubID {user_sub_id}] Unexpected error fetching market data for {init_params['symbol']}: {e}")
                current_sub_for_loop.status_message = f"Running - Data fetch error: {str(e)[:100]}"; db_session.commit()

            try:
                logger.debug(f"[SubID {user_sub_id}] Calling execute_live_signal for '{strategy_instance.name}'.")
                strategy_instance.execute_live_signal(
                    db_session=db_session,            
                    user_sub_obj=current_sub_for_loop, 
                    market_data_df=market_data_df,    
                    exchange_ccxt=exchange_ccxt      
                )
                current_sub_for_loop.status_message = f"Running - Last successful cycle: {datetime.datetime.utcnow().isoformat()}"
                db_session.commit()
                logger.debug(f"[SubID {user_sub_id}] Strategy execute_live_signal completed for {init_params['symbol']}.")
            except Exception as e: # This catches errors from within the strategy's own code
                logger.exception(f"[SubID {user_sub_id}] Error in strategy execute_live_signal for '{strategy_instance.name}': {e}")
                current_sub_for_loop.status_message = f"Error in execution: {str(e)[:150]}" 
                db_session.commit()
            
            sleep_duration_seconds = 60 
            try:
                tf_val_str = ''.join(filter(str.isdigit, init_params['timeframe']))
                tf_unit = ''.join(filter(str.isalpha, init_params['timeframe'])).lower()
                if tf_val_str:
                    tf_val = int(tf_val_str)
                    if tf_unit == 'm': base_sleep = tf_val * 60
                    elif tf_unit == 'h': base_sleep = tf_val * 3600
                    elif tf_unit == 'd': base_sleep = tf_val * 86400
                    else: base_sleep = 60 
                    sleep_duration_seconds = max(1, int(base_sleep * 0.95)) 
                logger.debug(f"[SubID {user_sub_id}] Calculated sleep duration: {sleep_duration_seconds}s based on timeframe {init_params['timeframe']}.")
            except ValueError: 
                logger.warning(f"[SubID {user_sub_id}] Could not parse timeframe '{init_params['timeframe']}'. Defaulting sleep to 60s.")
            
            time.sleep(sleep_duration_seconds)

    except SQLAlchemyError as sae:
        logger.exception(f"[SubID {user_sub_id}] SQLAlchemyError in task run_live_strategy setup: {sae}")
        # Attempt to update status if possible, but session might be compromised
        try:
            if db_session and user_sub: # Check if user_sub was fetched
                user_sub.status_message = f"DB Error during task setup: {str(sae)[:100]}"
                user_sub.is_active = False
                db_session.commit()
        except Exception as db_err_sae:
            logger.error(f"[SubID {user_sub_id}] Further DB error updating status after SQLAlchemyError: {db_err_sae}", exc_info=True)
        return {"status": "error", "message": f"Database error in task setup: {sae}"}
    except ccxt.BaseError as cbe: # Errors during exchange init or initial checks
        logger.exception(f"[SubID {user_sub_id}] CCXT BaseError in task run_live_strategy setup: {cbe}")
        try:
            if db_session and user_sub:
                user_sub.status_message = f"Exchange Init Error: {str(cbe)[:100]}"
                user_sub.is_active = False
                db_session.commit()
        except Exception as db_err_cbe:
            logger.error(f"[SubID {user_sub_id}] Further DB error updating status after CCXT Error: {db_err_cbe}", exc_info=True)
        return {"status": "error", "message": f"CCXT error in task setup: {cbe}"}
    except Exception as e: # General fallback for setup errors
        logger.exception(f"[SubID {user_sub_id}] Critical error in task run_live_strategy setup phase: {e}")
        try:
            if db_session and user_sub: # Check if user_sub was fetched
                sub_to_update = db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == user_sub_id).first() # Re-fetch for safety
                if sub_to_update: 
                    sub_to_update.status_message = f"Critical Task Setup Error: {str(e)[:150]}"
                    sub_to_update.is_active = False 
                    db_session.commit()
        except Exception as db_err: 
            logger.error(f"[SubID {user_sub_id}] DB error while updating status on critical task setup error: {db_err}", exc_info=True)
        return {"status": "error", "message": f"Critical error in task setup: {e}"} 
    finally:
        if db_session: db_session.close()
        logger.info(f"[SubID {user_sub_id}] Task run_live_strategy finished one execution cycle or stopped.")
        return {"status": "completed", "message": "Task run_live_strategy cycle finished or stopped."}


@celery_app.task(bind=True)
def run_backtest_task(self, backtest_result_id: int, user_id: int, strategy_id: int,
                      custom_parameters: dict, symbol: str, timeframe: str,
                      start_date_str: str, end_date_str: str,
                      initial_capital: float = 10000.0, # Default initial capital for backtests.
                                                        # Consider making this a parameter from the API if varied inputs are common.
                      exchange_id: str = 'binance'):    # Default exchange for backtests.
                                                        # Consider making this a parameter from the API for broader exchange support.
    from backend.services.backtesting_service import _perform_backtest_logic # Import inside function
    db_session = None
    try:
        db_session = SessionLocal()
        logger.info(f"Starting backtest task {self.request.id} for User {user_id}, Strategy {strategy_id}, BR_ID {backtest_result_id}.")
        
        result = _perform_backtest_logic(
            db_session=db_session, backtest_result_id=backtest_result_id, user_id=user_id,
            strategy_id=strategy_id, custom_parameters=custom_parameters, symbol=symbol,
            timeframe=timeframe, start_date_str=start_date_str, end_date_str=end_date_str,
            initial_capital=initial_capital, exchange_id=exchange_id
        )
        
        if result.get("status") == "success":
            logger.info(f"Backtest task {self.request.id} for BR_ID {backtest_result_id} reported success from _perform_backtest_logic.")
        else:
            logger.error(f"Backtest task {self.request.id} for BR_ID {backtest_result_id} reported failure from _perform_backtest_logic: {result.get('message')}")
        
        return result
    except SQLAlchemyError as sae: # Error with SessionLocal() or initial DB ops if any before _perform_backtest_logic
        logger.exception(f"SQLAlchemyError in run_backtest_task {self.request.id} for BR_ID {backtest_result_id} (before main logic): {sae}")
        # Attempt to mark BacktestResult as failed if possible
        try:
            if db_session: # db_session might not be valid if SessionLocal() failed
                br_record = db_session.query(BacktestResult).filter(BacktestResult.id == backtest_result_id).first()
                if br_record and br_record.status not in ["completed", "failed"]:
                    br_record.status = "failed"; br_record.status_message = f"Task DB Setup Error: {str(sae)[:100]}"; db_session.commit()
        except Exception as db_update_err:
            logger.error(f"Further DB error updating BacktestResult {backtest_result_id} after task setup SQLAlchemyError: {db_update_err}", exc_info=True)
        return {"status": "error", "message": f"Database setup error in backtest task: {sae}"}
    except Exception as e: # General errors not caught by _perform_backtest_logic or initial setup
        logger.exception(f"Critical unhandled error in backtest task {self.request.id} for BR_ID {backtest_result_id}: {e}")
        try:
            if db_session:
                br_record = db_session.query(BacktestResult).filter(BacktestResult.id == backtest_result_id).first()
                if br_record and br_record.status not in ["completed", "failed"]: 
                    br_record.status = "failed"
                    br_record.status_message = f"Critical task error: {str(e)[:250]}" 
                    br_record.pnl = 0 
                    br_record.updated_at = datetime.datetime.utcnow()
                    db_session.commit()
        except Exception as db_err: # Nested try-except for status update
            logger.error(f"DB error updating BacktestResult {backtest_result_id} on critical task error: {db_err}", exc_info=True)
        return {"status": "error", "message": f"Critical error during backtest task execution: {e}"}
    finally:
        if db_session: db_session.close()
        logger.info(f"Backtest task {self.request.id} for BR_ID {backtest_result_id} finished processing.")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_email_task(self, to_email: str, subject:str, body: str):
    logger.info(f"Celery task send_email_task received for {to_email} with subject '{subject}'")
    if not settings.SMTP_HOST or not settings.SMTP_USER or not settings.SMTP_PASSWORD or not settings.EMAILS_FROM_EMAIL:
        logger.warning(f"Email not sent by Celery task to {to_email}: SMTP settings not fully configured. Simulating email.")
        logger.info(f"--- Celery Simulated Email to {to_email} ---")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body:\n{body}")
        logger.info(f"--- End of Celery Simulated Email ---")
        return {"status": "warning_simulated", "message": "Email settings not configured; email simulated by Celery."}

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = settings.EMAILS_FROM_EMAIL
        msg['To'] = to_email

        smtp_port = settings.SMTP_PORT if settings.SMTP_PORT is not None else 587

        with smtplib.SMTP(settings.SMTP_HOST, smtp_port) as server:
            if settings.SMTP_TLS:
                server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.EMAILS_FROM_EMAIL, to_email, msg.as_string())
        
        logger.info(f"Celery task successfully sent email to {to_email} with subject '{subject}'.")
        return {"status": "success", "message": "Email sent successfully by Celery."}
    except (smtplib.SMTPAuthenticationError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, socket.gaierror) as smtp_e:
        logger.error(f"SMTP error sending email to {to_email}: {smtp_e}", exc_info=True)
        try:
            self.retry(exc=smtp_e)
        except Exception as retry_exc: # Catch MaxRetriesExceededError
             logger.error(f"Celery task send_email_task failed after retries for {to_email} due to SMTP error: {retry_exc}", exc_info=True)
             raise # Re-raise the exception to mark the task as failed
        raise # Re-raise the exception to mark the task as failed if not retrying
    except Exception as e: # General fallback for other errors
        logger.exception(f"Unexpected error in send_email_task for {to_email}: {e}")
        try:
            self.retry(exc=e)
        except Exception as retry_exc: # Catch MaxRetriesExceededError
             logger.error(f"Celery task send_email_task failed after retries for {to_email} due to unexpected error: {retry_exc}", exc_info=True)
             raise
        raise
